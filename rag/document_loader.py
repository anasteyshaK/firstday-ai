
import hashlib
import os
import re
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"
load_dotenv(PROJECT_ROOT / ".env", override=False)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5434")),
    "dbname": os.getenv("DB_NAME", "firstday_db"),
    "user": os.getenv("DB_USER", "firstday_user"),
    "password": os.getenv("DB_PASSWORD", ""),
}

OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "http://localhost:11434"
)
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSIONS = 768


def parse_document(text: str):
    metadata = {}
    body = text

    if text.startswith("---"):
        match = re.match(
            r"\A---\s*\n(.*?)\n---\s*\n",
            text,
            re.DOTALL,
        )

        if match:
            header = match.group(1)
            body = text[match.end():]

            for line in header.splitlines():
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                metadata[key.strip()] = (
                    value.strip().strip('"').strip("'")
                )

    return metadata, body.strip()


def split_into_chunks(text: str, max_chars=1200):
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            for start in range(0, len(paragraph), max_chars):
                chunks.append(
                    paragraph[start:start + max_chars]
                )
            continue

        if len(current) + len(paragraph) + 2 > max_chars:
            if current:
                chunks.append(current)
            current = paragraph
        else:
            current = (
                current + "\n\n" + paragraph
            ).strip()

    if current:
        chunks.append(current)

    return chunks


def create_embedding(text: str):
    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": text,
    }).encode("utf-8")

    request = Request(
        url=f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except URLError as error:
        raise RuntimeError(
            "Не удалось подключиться к Ollama. "
            "Проверь, что Ollama запущена."
        ) from error

    embeddings = result.get("embeddings", [])

    if not embeddings:
        raise RuntimeError(
            "Ollama не вернула embedding."
        )

    embedding = embeddings[0]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Ожидалось {EMBEDDING_DIMENSIONS} значений, "
            f"получено {len(embedding)}"
        )

    return "[" + ",".join(map(str, embedding)) + "]"


def load_document(connection, file_path):
    raw_text = file_path.read_text(encoding="utf-8")
    metadata, content = parse_document(raw_text)

    if not content:
        return 0

    relative_path = str(
        file_path.relative_to(KNOWLEDGE_BASE)
    ).replace("\\", "/")

    title = metadata.get("title", file_path.stem)
    category = metadata.get(
        "category",
        file_path.parent.name,
    )
    department = metadata.get("department") or None
    access_level = metadata.get(
        "access_level", "EMPLOYEE"
    ).upper()

    if access_level not in {"EMPLOYEE", "HR", "ADMIN"}:
        raise ValueError(
            f"Некорректный access_level: {relative_path}"
        )

    content_hash = hashlib.sha256(
        raw_text.encode("utf-8")
    ).hexdigest()

    chunks = split_into_chunks(content)

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, content_hash
            FROM documents
            WHERE file_path = %s
            """,
            (relative_path,),
        )
        existing = cursor.fetchone()

        if existing and existing[1] == content_hash:
            print(f"[SKIP] {relative_path}")
            return 0

        # Получаем embeddings до изменения базы.
        vectors = [
            create_embedding(chunk)
            for chunk in chunks
        ]

        cursor.execute(
            """
            INSERT INTO documents (
                title,
                file_path,
                category,
                department,
                access_level,
                content_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path)
            DO UPDATE SET
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                department = EXCLUDED.department,
                access_level = EXCLUDED.access_level,
                content_hash = EXCLUDED.content_hash,
                updated_at = NOW()
            RETURNING id
            """,
            (
                title,
                relative_path,
                category,
                department,
                access_level,
                content_hash,
            ),
        )

        document_id = cursor.fetchone()[0]

        # Обновлённый документ заменяет старые фрагменты.
        cursor.execute(
            """
            DELETE FROM document_chunks
            WHERE document_id = %s
            """,
            (document_id,),
        )

        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors)
        ):
            cursor.execute(
                """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s::vector)
                """,
                (
                    document_id,
                    index,
                    chunk,
                    vector,
                ),
            )

    connection.commit()
    print(
        f"[OK] {relative_path}: {len(chunks)} chunks"
    )
    return len(chunks)


def main():
    if not KNOWLEDGE_BASE.exists():
        raise FileNotFoundError(
            f"Папка не найдена: {KNOWLEDGE_BASE}"
        )

    files = sorted(KNOWLEDGE_BASE.rglob("*.md"))
    files = [
        path for path in files
        if path.name.lower() != "readme.md"
    ]

    print(f"Найдено документов: {len(files)}")

    if not files:
        print("Нет Markdown-файлов для загрузки.")
        return

    total_chunks = 0
    processed = 0
    failed = 0

    with psycopg2.connect(**DB_CONFIG) as connection:
        for file_path in files:
            try:
                count = load_document(
                    connection,
                    file_path,
                )
                total_chunks += count
                processed += 1
            except Exception as error:
                connection.rollback()
                failed += 1
                print(
                    f"[ERROR] {file_path.name}: {error}"
                )

    print("\nЗагрузка завершена!")
    print(f"Обработано документов: {processed}")
    print(f"Ошибок: {failed}")
    print(f"Новых фрагментов: {total_chunks}")


if __name__ == "__main__":
    main()

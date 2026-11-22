"""Document retrieval for FirstDay AI.

Features:
- pgvector semantic search
- keyword reranking
- role-aware filtering: Java, QA, DevOps
- tool-aware filtering: Slack, Jira, GitHub, MFA, etc.
- special handling for first-day onboarding questions
- document deduplication
- access-level filtering

Compatible with rag/answer.py.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from collections import defaultdict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg2
from psycopg2.extras import RealDictCursor
from query_policy import stack_question, technology_evidence


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5434")),
    "dbname": os.getenv("DB_NAME", "firstday_db"),
    "user": os.getenv("DB_USER", "firstday_user"),
    "password": os.getenv("DB_PASSWORD", ""),
}

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
).rstrip("/")

EMBEDDING_MODEL = "nomic-embed-text"

MIN_SIMILARITY = 0.55

STOP_WORDS = {
    "что", "как", "где", "когда", "какие",
    "какой", "какая", "нужно", "можно",
    "должен", "должна", "сделать", "мне",
    "для", "это", "или", "при", "его",
    "мои", "все", "если", "есть", "про",
    "the", "how", "what", "when", "where",
    "can", "should", "does", "get", "my",
    "to", "in", "on", "for", "and", "is",
    "are", "an", "a",
}


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def create_embedding(text: str) -> str:
    """Generate an embedding with Ollama."""

    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": text,
    }).encode("utf-8")

    request = Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=180,
        ) as response:
            result = json.load(response)

    except HTTPError as error:
        raise RuntimeError(
            f"Ollama embedding HTTP {error.code}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Не удалось подключиться к Ollama."
        ) from error

    embeddings = result.get("embeddings", [])

    if not embeddings:
        raise RuntimeError(
            "Ollama не вернула embedding."
        )

    if len(embeddings[0]) != 768:
        raise RuntimeError(
            "Некорректная размерность embedding."
        )

    return json.dumps(embeddings[0])


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text: str) -> str:
    """Lowercase text and normalize Russian ё."""

    return text.casefold().replace("ё", "е")


def meaningful_words(text: str) -> set[str]:
    """Extract meaningful words for lexical ranking."""

    words = re.findall(
        r"[a-zа-я0-9]+",
        normalize(text),
    )

    return {
        word
        for word in words
        if len(word) >= 3
        and word not in STOP_WORDS
    }


def keyword_score(
    question: str,
    result: dict,
) -> float:
    """Compare question terms with document title/content."""

    question_words = meaningful_words(question)

    if not question_words:
        return 0.0

    title_words = meaningful_words(
        result["title"]
    )

    content_words = meaningful_words(
        result["content"]
    )

    title_matches = len(
        question_words & title_words
    )

    content_matches = len(
        question_words & content_words
    )

    title_score = (
        title_matches / len(question_words)
    )

    content_score = (
        content_matches / len(question_words)
    )

    return min(
        1.0,
        0.7 * title_score
        + 0.3 * content_score,
    )


# ============================================================
# ROLE AND TOOL IDENTIFICATION
# ============================================================

ROLE_PATTERNS = {
    "java": (
        r"\bjava\b",
        r"\bджава\b",
    ),
    "qa": (
        r"\bqa\b",
        r"\bтестировщик\w*\b",
    ),
    "devops": (
        r"\bdevops\b",
        r"\bsre\b",
    ),
}

TOOL_PATTERNS = {
    "slack": (
        r"\bslack\b",
    ),
    "jira": (
        r"\bjira\b",
        r"\bджира\b",
    ),
    "confluence": (
        r"\bconfluence\b",
    ),
    "figma": (
        r"\bfigma\b",
    ),
    "gitlab": (
        r"\bgitlab\b",
        r"\bгитлаб\b",
    ),
    "github": (
        r"\bgithub\b",
        r"\bгитхаб\b",
    ),
    "vpn": (
        r"\bvpn\b",
        r"\bвпн\b",
    ),
    "mfa": (
        r"\bmfa\b",
        r"\b2fa\b",
        r"\bмногофакторн\w*\b",
    ),
}


def detected_names(
    question: str,
    patterns: dict[str, tuple[str, ...]],
) -> set[str]:
    """Detect explicitly mentioned roles or tools."""

    text = normalize(question)
    found: set[str] = set()

    for name, expressions in patterns.items():
        if any(
            re.search(expression, text)
            for expression in expressions
        ):
            found.add(name)

    return found


def topic_match(
    text: str,
    topic: str,
) -> bool:
    """Match a known role/tool in a document identity."""

    # A file path like 06_devops/oncall.md must still match the role
    # "devops".  Regex word boundaries do not split underscores because
    # underscore is considered a word character by Python's re module.
    normalized = re.sub(r"[_./\\-]+", " ", normalize(text))

    patterns = (
        ROLE_PATTERNS.get(topic)
        or TOOL_PATTERNS.get(topic)
        or ()
    )

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


# ============================================================
# FIRST-DAY CONTEXT
# ============================================================

def is_first_day_question(question: str) -> bool:
    """Recognize different Russian/English first-day forms.

    Examples:
        первый рабочий день
        первого рабочего дня
        в первый день
        в первый рабочий день
        first day
        day one
        day 1

    This prevents MFA mentioned inside a longer
    first-day question from being treated as the
    only topic of that question.
    """

    text = normalize(question)

    russian = re.search(
        r"\bперв\w*\s+"
        r"(?:рабоч\w*\s+)?"
        r"(?:день|дня|дне|днем|дню)\b",
        text,
    )

    english = re.search(
        r"\b(?:first[\s-]*day|day[\s-]*one|day[\s-]*1)\b",
        text,
    )

    return bool(russian or english)


# ============================================================
# QUERY INTENT HELPERS (GENERAL, NOT DOCUMENT-ID HARDCODING)
# ============================================================

def is_access_setup_question(question: str) -> bool:
    """Distinguish requesting initial access from fixing an incident.

    A request for GitHub access is not answered by an incident runbook
    about a broken GitHub Actions workflow, even though both mention GitHub.
    """
    text = normalize(question)
    request = bool(re.search(
        r"(?:\bкак\s+(?:получить|запросить|оформить)\s+доступ\b"
        r"|\bнужен\s+доступ\b"
        r"|\bhow\s+(?:do\s+i\s+)?(?:get|request)\s+access\b)",
        text,
    ))
    incident = bool(re.search(
        r"\b(?:если|ошибк\w*|не\s+работает|сломал\w*|потер\w*|"
        r"заблокир\w*|failed|broken|lost|error)\b",
        text,
    ))
    return request and not incident


def is_first_day_overview_question(question: str) -> bool:
    """General first-day checklist, not a particular first-day problem.

    Detailed questions about broken laptops, VPN, MFA or other incidents
    must be free to retrieve the relevant operational instructions.
    """
    text = normalize(question)
    if not is_first_day_question(text):
        return False
    if detected_names(text, TOOL_PATTERNS) or detected_names(text, ROLE_PATTERNS):
        return False
    if re.search(
        r"\b(?:если|ошибк\w*|сломал\w*|не\s+работает|"
        r"ноутбук\w*|парол\w*|заблокир\w*|lost|broken|failed)\b",
        text,
    ):
        return False
    return bool(re.search(
        r"\b(?:что\s+(?:мне\s+)?(?:делать|нужно)\b|"
        r"какие\s+задач\w*\b|"
        r"план\s+перв\w*\s+дн\w*\b|"
        r"what\s+(?:should|do)\s+i\s+do\b)",
        text,
    ))


def is_first_day_overview_document(result: dict) -> bool:
    """Prefer day-one and week-one instructions, not preparation beforehand."""
    identity = re.sub(
        r"[_./\\-]+", " ",
        normalize(f"{result['title']} {result['file_path']}"),
    )
    # Documents about activities *before* the first day are relevant for
    # advance preparation, but not for what to do *during* the first day.
    if re.search(
        r"(?:\bbefore\s+first\s+day\b|"
        r"\b(?:подготовк\w*\s+)?до\s+перв\w*\s+(?:рабоч\w*\s+)?дн\w*\b)",
        identity,
    ):
        return False
    return bool(re.search(
        r"\b(?:day\s+one|first\s+day|week\s+one|first\s+week|"
        r"перв\w*\s+(?:рабоч\w*\s+)?д(?:ень|ня|не|ню|нем)|"
        r"план\s+перв\w*\s+недел\w*)\b",
        identity,
    ))


def is_role_first_week_overview_question(question: str) -> bool:
    """True for a general role's first-week plan, not a how-to question.

    Detailed questions (for example, how to write a QA bug report) should
    still be able to retrieve specialized documentation.
    """
    text = normalize(question)
    roles = detected_names(text, ROLE_PATTERNS)
    if len(roles) != 1:
        return False
    if re.search(
        r"\b(?:как\s+настроить|как\s+оформить|как\s+создать|"
        r"ошибк\w*|баг[\s-]*репорт\w*|bug[\s-]*report\w*|"
        r"how\s+to|how\s+do\s+i)\b",
        text,
    ):
        return False
    return bool(re.search(
        r"\b(?:перв\w*\s+недел\w*|недел\w*\s*1|"
        r"first[\s-]*week|week\s*1|"
        r"дни\s*1\s*[-–—]\s*5|days?\s*1\s*[-–—]\s*5)\b",
        text,
    ))


def is_role_first_week_overview_document(question: str, result: dict) -> bool:
    """For role-week overviews, prefer schedules and onboarding guides.

    Use the document's *scope* (title/path) rather than an allowlist of
    known filenames. Detailed operational runbooks remain eligible for
    questions about those procedures.
    """
    role = next(iter(detected_names(question, ROLE_PATTERNS)))
    identity = re.sub(
        r"[_./\\-]+", " ",
        normalize(f"{result['title']} {result['file_path']}"),
    )
    if not topic_match(identity, role):
        return False
    return bool(re.search(
        r"\b(?:onboarding|learning|curriculum|"
        r"адаптац\w*|обучени\w*|учебн\w*|трек\w*|"
        r"перв\w*\s+недел\w*|first\s+week)\b",
        identity,
    ))


# ============================================================
# DOCUMENT TOPIC FILTERING
# ============================================================

def passes_topic_filter(
    question: str,
    result: dict,
) -> bool:
    """Filter documents by explicitly named topics.

    Different professions should not be mixed.

    For ordinary questions about one named tool,
    require the document title/path to match it.

    Exception:
        A first-day onboarding question may mention
        MFA as one task among several. In that case,
        do not require every document to be about MFA.
    """

    roles = detected_names(
        question,
        ROLE_PATTERNS,
    )

    tools = detected_names(
        question,
        TOOL_PATTERNS,
    )

    document_identity = (
        f"{result['title']} "
        f"{result['file_path']}"
    )

    # An explicit initial-access question should not cite a troubleshooting
    # incident, even when the incident mentions the same product.
    path = normalize(result["file_path"]).replace("\\", "/")
    if (
        is_access_setup_question(question)
        and "11_incidents/" in path
    ):
        return False

    # Broad first-day checklists need first-day sources, not generic
    # development guides that happen to rank well by embedding similarity.
    if (
        is_first_day_overview_question(question)
        and not is_first_day_overview_document(result)
    ):
        return False

    # A broad week-one curriculum question needs the role's learning track
    # or onboarding schedule, not isolated procedure guides.
    if (
        is_role_first_week_overview_question(question)
        and not is_role_first_week_overview_document(question, result)
    ):
        return False

    # ----------------------------------------------
    # ROLE FILTERING
    # ----------------------------------------------

    if len(roles) == 1:
        requested_role = next(iter(roles))

        other_roles = (
            set(ROLE_PATTERNS)
            - {requested_role}
        )

        if any(
            topic_match(document_identity, other)
            for other in other_roles
        ):
            return False

    # ----------------------------------------------
    # FIRST-DAY EXCEPTION
    # ----------------------------------------------

    first_day_context = is_first_day_question(
        question
    )

    # ----------------------------------------------
    # TOOL FILTERING
    # ----------------------------------------------

    if len(tools) == 1 and not first_day_context:
        requested_tool = next(iter(tools))

        if not topic_match(
            document_identity,
            requested_tool,
        ):
            return False

    return True


def document_topic_bonus(
    question: str,
    result: dict,
) -> float:
    """Reward documents whose title/path matches the topic."""

    identity = (
        f"{result['title']} "
        f"{result['file_path']}"
    )

    requested_topics = (
        detected_names(
            question,
            ROLE_PATTERNS,
        )
        | detected_names(
            question,
            TOOL_PATTERNS,
        )
    )

    if not requested_topics:
        return 0.0

    matches = sum(
        1
        for topic in requested_topics
        if topic_match(identity, topic)
    )

    return min(
        0.15,
        matches * 0.10,
    )


# ============================================================
# HYBRID DOCUMENT SEARCH
# ============================================================

def search_documents(
    question: str,
    limit: int = 5,
    access_level: str = "EMPLOYEE",
) -> list[dict]:
    """Search documents using vector and lexical relevance.

    Compatibility:
        Keeps the existing function signature.

    Security:
        Document access-level filtering is performed
        inside the database query.

    Ranking:
        75% semantic similarity
        25% keyword relevance
        + exact topic bonus
    """

    if not question.strip():
        raise ValueError(
            "Вопрос не может быть пустым."
        )

    if not 1 <= limit <= 20:
        raise ValueError(
            "limit должен быть от 1 до 20."
        )

    allowed_levels = {
        "EMPLOYEE": [
            "EMPLOYEE",
        ],
        "HR": [
            "EMPLOYEE",
            "HR",
        ],
        "ADMIN": [
            "EMPLOYEE",
            "HR",
            "ADMIN",
        ],
    }

    if access_level not in allowed_levels:
        raise ValueError(
            "Неизвестный уровень доступа."
        )

    # ----------------------------------------------
    # STEP 1. GENERATE QUERY EMBEDDING
    # ----------------------------------------------

    embedding = create_embedding(
        question + (' локальная разработка backend окружение programming languages frameworks database' if stack_question(question) else '')
    )

    candidate_limit = max(
        100,
        limit * 12,
    )

    # ----------------------------------------------
    # STEP 2. VECTOR SEARCH IN POSTGRESQL
    # ----------------------------------------------

    query = """
        SELECT
            d.title,
            d.file_path,
            d.category,
            d.access_level,
            c.content,
            c.chunk_index,
            1 - (c.embedding <=> %s::vector)
                AS similarity
        FROM document_chunks AS c
        JOIN documents AS d
            ON d.id = c.document_id
        WHERE d.access_level = ANY(%s)
          AND c.embedding IS NOT NULL
        ORDER BY CASE WHEN %s AND c.content ~* %s THEN 0 ELSE 1 END,
                 c.embedding <=> %s::vector
        LIMIT %s;
    """

    with psycopg2.connect(
        **DB_CONFIG
    ) as connection:

        with connection.cursor(
            cursor_factory=RealDictCursor,
        ) as cursor:

            cursor.execute(
                query,
                (
                    embedding,
                    allowed_levels[access_level],
                    stack_question(question),
                    r'(backend|стек|stack).*(использу|uses|Java|Python|Spring)',
                    embedding,
                    candidate_limit,
                ),
            )

            candidates = [
                dict(row)
                for row in cursor.fetchall()
            ]

    # ----------------------------------------------
    # STEP 3. FILTER AND RERANK
    # ----------------------------------------------

    filtered: list[dict] = []

    requested_topics = (
        detected_names(
            question,
            ROLE_PATTERNS,
        )
        | detected_names(
            question,
            TOOL_PATTERNS,
        )
    )

    for result in candidates:
        similarity = float(
            result["similarity"]
        )

        vector_score = max(
            0.0,
            min(1.0, similarity),
        )

        lexical_score = keyword_score(
            question,
            result,
        )

        result["keyword_score"] = lexical_score

        # Remove documents that do not match
        # the explicitly requested role/tool.
        if not passes_topic_filter(
            question,
            result,
        ):
            continue

        identity = (
            f"{result['title']} "
            f"{result['file_path']}"
        )

        has_exact_topic = any(
            topic_match(identity, topic)
            for topic in requested_topics
        )

        # Reject weak semantic matches unless
        # the document clearly matches the topic.
        if (
            similarity < MIN_SIMILARITY
            and not has_exact_topic
            and not (stack_question(question) and technology_evidence(result['content']))
        ):
            continue

        bonus = document_topic_bonus(
            question,
            result,
        )

        result["rerank_score"] = (
            0.75 * vector_score
            + 0.25 * lexical_score
            + bonus
            + (2.0 if stack_question(question) and technology_evidence(result['content']) else 0.0)
        )

        filtered.append(result)

    # ----------------------------------------------
    # STEP 4. SORT RESULTS
    # ----------------------------------------------

    filtered.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    # ----------------------------------------------
    # STEP 5. LIMIT CHUNKS PER DOCUMENT
    # ----------------------------------------------

    document_counts = defaultdict(int)
    selected: list[dict] = []

    for result in filtered:
        path = result["file_path"]

        if document_counts[path] >= 2:
            continue

        selected.append(result)
        document_counts[path] += 1

        if len(selected) >= limit:
            break

    return selected


# ============================================================
# CLI OUTPUT
# ============================================================

def main():
    question = " ".join(
        sys.argv[1:]
    ).strip()

    if not question:
        question = input(
            "Введите вопрос: "
        ).strip()

    if not question:
        print(
            "Вопрос не указан."
        )
        return

    print(
        "\nFirstDay AI ищет документы...\n"
    )

    results = search_documents(
        question=question,
    )

    print(
        f"Вопрос: {question}"
    )

    print("=" * 65)

    if not results:
        print(
            "Подходящих документов не найдено. "
            "Нельзя формировать инструкцию "
            "на основании нерелевантных источников."
        )
        return

    unique_paths: set[str] = set()

    for index, result in enumerate(
        results,
        start=1,
    ):
        unique_paths.add(
            result["file_path"]
        )

        print(
            f"\nРезультат #{index}"
        )

        print(
            f"Документ: {result['title']}"
        )

        print(
            f"Источник: {result['file_path']}"
        )

        print(
            "Vector similarity: "
            f"{result['similarity']:.3f}"
        )

        print(
            "Keyword score: "
            f"{result['keyword_score']:.3f}"
        )

        print(
            "Rerank score: "
            f"{result['rerank_score']:.3f}"
        )

        print(
            "Фрагмент:\n"
            f"{result['content'][:500]}"
        )

    print(
        "\nНайдено подходящих документов:",
        len(unique_paths),
    )


if __name__ == "__main__":
    main()

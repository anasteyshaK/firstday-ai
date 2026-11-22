"""Apply the additive project migration and seed synthetic demo documents only.

Run: python -m tools.setup_project_onboarding
Never reads employee rows or prints connection credentials. Re-running preserves progress.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag"))


PROJECTS = {
    "orbitflow": {
        "name": "OrbitFlow",
        "description": "Подключение к платформе рабочих процессов: продукт, сервисы, среда и первый вклад.",
        "docs": {
            "overview.md": ("OrbitFlow: продукт и рабочие сценарии", "Контекст", ["07_product/orbitflow-overview.md", "07_product/workflow-lifecycle.md"]),
            "architecture.md": ("OrbitFlow: архитектура, сервисы и ответственные", "Контекст", ["04_engineering/service-catalog.md", "04_engineering/engineering-overview.md"]),
            "environment.md": ("OrbitFlow: окружение, доступы и запуск", "Окружение", ["04_engineering/local-development.md", "02_it_support/github-access.md", "05_qa/qa-onboarding.md", "06_devops/environments.md"]),
            "delivery.md": ("OrbitFlow: первая задача, PR и review", "Рабочий процесс", ["12_process/first-contribution.md", "04_engineering/pull-request.md", "12_process/definition-of-done.md"]),
        },
    },
    "beacon": {
        "name": "Beacon Observability",
        "description": "Подключение к направлению наблюдаемости: сигналы, окружения, runbooks и безопасные изменения.",
        "docs": {
            "overview.md": ("Beacon: назначение и сигналы мониторинга", "Контекст", ["06_devops/monitoring.md"]),
            "architecture.md": ("Beacon: навигация по сервисам и ответственным", "Контекст", ["06_devops/monitoring.md", "06_devops/oncall.md"]),
            "environment.md": ("Beacon: окружения и безопасный доступ", "Окружение", ["06_devops/environments.md", "06_devops/docker-basics.md", "03_security/least-privilege.md"]),
            "delivery.md": ("Beacon: первая задача и согласование изменений", "Рабочий процесс", ["06_devops/oncall.md", "06_devops/rollback.md", "12_process/definition-of-done.md"]),
        },
    },
}


def demo_document(paths: list[str]) -> str:
    sections = ["> Синтетический демо-проект NovaTech. Инструкции и системы вымышлены.\n"
                "> Маршрут подходит специалистам любого уровня; детали уточняются у команды проекта."]
    for path in paths:
        text = (ROOT / "knowledge_base" / path).read_text(encoding="utf-8-sig")
        if text.startswith("---"):
            text = text.split("---", 2)[2].strip()
        text = text.split("## Контрольный результат", 1)[0].strip()
        sections.append(f"Исходный учебный материал: `{path}`\n\n{text}")
    return "\n\n---\n\n".join(sections)


def main() -> int:
    # Environment must be loaded before search constructs DB_CONFIG.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    import psycopg2
    from search import DB_CONFIG
    try:
        with psycopg2.connect(**DB_CONFIG, connect_timeout=5) as conn:
            with conn.cursor() as cursor:
                cursor.execute((ROOT / "db/005_project_onboarding.sql").read_text(encoding="utf-8"))
            # The SQL migration has its own transaction. Demo seeding is atomic below.
            with conn.cursor() as cursor:
                for slug, project in PROJECTS.items():
                    cursor.execute(
                        """INSERT INTO projects (slug, name, description, is_demo) VALUES (%s,%s,%s,TRUE)
                           ON CONFLICT (slug) DO NOTHING""", (slug, project["name"], project["description"]),
                    )
                    cursor.execute("SELECT id FROM projects WHERE slug=%s AND is_demo=TRUE", (slug,))
                    row = cursor.fetchone()
                    if row is None:
                        raise RuntimeError("Demo slug is already used by a private project")
                    for filename, (title, category, paths) in project["docs"].items():
                        cursor.execute(
                            """INSERT INTO project_documents (project_id, file_path, title, category, content)
                               VALUES (%s,%s,%s,%s,%s) ON CONFLICT (project_id, file_path) DO NOTHING""",
                            (row[0], f"projects/{slug}/{filename}", title, category, demo_document(paths)),
                        )
        print("Project migration ready. Two synthetic demo projects seeded; admin assignment required. Apply migration 006 before starting the app.")
        return 0
    except Exception as exc:
        print(f"Project setup failed: {type(exc).__name__}. Check PostgreSQL and local configuration.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

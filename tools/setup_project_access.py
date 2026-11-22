"""Apply only migration 006; never reads employee data or prints credentials."""
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    from dotenv import load_dotenv
    load_dotenv(root / '.env', override=False)
    import psycopg2
    from backend.auth_routes import DB_CONFIG
    try:
        with psycopg2.connect(**DB_CONFIG, connect_timeout=5) as conn:
            with conn.cursor() as cursor:
                cursor.execute((root / 'db/006_project_assignments.sql').read_text(encoding='utf-8'))
        print('Project access migration ready. Existing profiles and progress preserved.')
        return 0
    except Exception as exc:
        print(f'Project access setup failed: {type(exc).__name__}. Check PostgreSQL and migration 005.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

# Local development setup

These PowerShell commands prepare a fresh local installation. Use Python 3.11 or newer, a Node.js version supported by Vite 6, Docker Compose, and Ollama. PostgreSQL and the optional pgAdmin interface are bound to localhost.

## 1. Get the source and dependencies

```powershell
git clone https://github.com/anasteyshaK/firstday-ai.git
Set-Location firstday-ai
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
npm --prefix frontend ci
```

`requirements.txt` contains runtime dependencies; `requirements-dev.txt` adds the test dependencies. The listed versions match the prepared development environment.

## 2. Configure your local environment

```powershell
Copy-Item .env.example .env
```

Edit `.env` locally and replace **both** password placeholders: `DB_PASSWORD` and `PGADMIN_PASSWORD`. Never commit this file. If you already have a configured `.env`, keep it and add the missing settings instead of replacing it. For an existing database, retain its current database password.

After changing `.env`, fully stop and restart the API process. The running Uvicorn reload process can retain the environment loaded at startup; a successful health response alone does not confirm database connectivity. Changing this local file does not require deleting data or recreating database volumes.

The defaults use PostgreSQL on port `5434`, Ollama on port `11434`, the `qwen3:1.7b` chat model, and `nomic-embed-text` embeddings. To use another installed chat model, update `CHAT_MODEL`.

```powershell
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
docker compose --env-file .env -f .\db\docker-compose.yml up -d
```

Keep Ollama running while indexing documentation and using generated answers. Model inference performance depends on your hardware.

## 3. Prepare a new database

Wait for PostgreSQL to become healthy:

```powershell
docker compose --env-file .env -f .\db\docker-compose.yml ps
```

For a **new development database**, apply the base schema and compatibility updates. These commands do not delete volumes or drop tables. Review SQL changes before applying them to an existing installation.

```powershell
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Get-Content .\db\schema.sql -Raw -Encoding utf8 | docker exec -i firstday_postgres psql -U firstday_user -d firstday_db -v ON_ERROR_STOP=1
Get-Content .\db\002_auth_sessions.sql -Raw -Encoding utf8 | docker exec -i firstday_postgres psql -U firstday_user -d firstday_db -v ON_ERROR_STOP=1
Get-Content .\db\004_chat_messages_compat.sql -Raw -Encoding utf8 | docker exec -i firstday_postgres psql -U firstday_user -d firstday_db -v ON_ERROR_STOP=1
.\venv\Scripts\python.exe -m tools.setup_project_onboarding
.\venv\Scripts\python.exe -m tools.setup_project_access
```

The two setup modules apply project schema updates and prepare synthetic demonstration projects. They preserve existing project progress.

Index the fictional company documentation. The wrapper loads local configuration before importing the document loader:

```powershell
.\venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); import runpy; runpy.run_module('rag.document_loader', run_name='__main__')"
```

The embedding model must be available for this step. Confirm that the loader reports no failed documents before testing knowledge questions.

## 4. Start the application

API terminal, from the repository root:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --env-file .env --host 127.0.0.1 --port 8001
```

Frontend terminal:

```powershell
npm --prefix frontend run dev
```

Open [the application](http://127.0.0.1:5173) and create your own account. API documentation is available at [localhost:8001/docs](http://127.0.0.1:8001/docs). pgAdmin is an optional database administration interface at [localhost:779](http://127.0.0.1:779), using `admin@example.com` and your local `PGADMIN_PASSWORD`.

Registration creates a regular employee account. Selecting a journey does not grant administrator rights or project membership. To administer your own development installation, run the interactive administrator utility and follow its prompts:

```powershell
.\venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); import runpy; runpy.run_module('backend.promote_admin', run_name='__main__')"
```

An administrator can then assign project memberships through the application. Read [Project Onboarding](PROJECT_ONBOARDING.md) for the permission model.

## 5. Verify the installation

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
npm --prefix frontend run build
```

The [answer quality guide](ANSWER_QUALITY.md) covers source checks, answer scenarios, and integration verification. Run database integration checks sequentially against your local development instance.

## Restarting and data persistence

Database state lives in Docker named volumes. Use the same Compose project and local credentials when restarting. Changing `DB_PASSWORD` in `.env` does not rotate the password inside an existing database volume. Never remove volumes as an ordinary restart step.

Langfuse telemetry is optional and disabled in `.env.example`. A public deployment needs separate operational and security preparation; this setup is intended for local development.

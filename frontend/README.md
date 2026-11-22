# FirstDay AI frontend

React + TypeScript + Vite. Локальная типографика Manrope; изображения находятся в `public/assets`.

## Запуск

Из папки frontend:

```powershell
npm install
npm run dev
```

Интерфейс работает на http://127.0.0.1:5173. Vite проксирует `/api` на FastAPI по адресу http://127.0.0.1:8001. Настройка backend описана в [README проекта](../README.md).

## Сборка

```powershell
npm run build
```

## Данные

Dashboard, Internship, Project Onboarding, Knowledge Hub, Access Center и история чата используют аутентифицированные API. Сотрудник видит только назначенные ему проекты. Статусы задач и блокеры сохраняются в PostgreSQL.

Чат отправляет `question`, `conversation_id` и выбранный `project_id` в `POST /api/v1/chat/stream`. Формат потока — NDJSON: status/delta/done/error. Обычный `/api/v1/chat` также поддерживается. Документы и переписка разделяются по сотруднику и проекту.

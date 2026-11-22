"""Local-only onboarding demo API.

Uses the existing employees / onboarding_plans / onboarding_tasks schema.
No authentication is implemented: this demo router MUST NOT be exposed publicly.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Literal
from uuid import UUID, uuid5, NAMESPACE_URL

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = PROJECT_ROOT / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
from search import DB_CONFIG

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/demo/onboarding", tags=["Onboarding demo"])
Role = Literal["JAVA", "QA", "DEVOPS"]
Status = Literal["TODO", "IN_PROGRESS", "DONE"]

ROLE_NAMES: dict[str, str] = {
    "JAVA": "Junior Java Developer",
    "QA": "QA Engineer",
    "DEVOPS": "Junior DevOps",
}

# Based on the fictional NovaTech knowledge-base documents
# 01_hr/week-one.md and 08_learning/{java,qa,devops}-track.md.
# Day placement inside the documented ranges is a demo scheduling choice.
COMMON_TASKS = [
    (1, "Настроить корпоративную учётную запись и MFA", "Знакомство с buddy; источник: 01_hr/week-one.md"),
    (2, "Подготовить ноутбук, VPN и менеджер паролей", "Источник: 01_hr/week-one.md"),
    (3, "Изучить инструменты команды и продукт", "Источник: 01_hr/week-one.md"),
    (4, "Подготовить локальное или учебное окружение", "Источник: 01_hr/week-one.md"),
    (5, "Показать первый результат и получить обратную связь", "Источник: 01_hr/week-one.md"),
]
ROLE_TASKS = {
    "JAVA": [
        (5, "Установить JDK 21, запустить сервис и изучить архитектуру", "Дни 1–5; 08_learning/java-track.md"),
        (10, "Изучить REST API, DTO и форматы ошибок", "Дни 6–10; 08_learning/java-track.md"),
        (18, "Реализовать небольшой endpoint с тестами", "Дни 11–20; 08_learning/java-track.md"),
        (30, "Подготовить PR, review, документацию и демо", "Дни 21–30; 08_learning/java-track.md"),
    ],
    "QA": [
        (5, "Изучить продукт и staging", "Дни 1–5; 08_learning/qa-track.md"),
        (10, "Оформить тест-кейсы и проверки API", "Дни 6–10; 08_learning/qa-track.md"),
        (18, "Создать Selenium или API automation test", "Дни 11–20; 08_learning/qa-track.md"),
        (30, "Добавить проверку в CI и представить результаты", "Дни 21–30; 08_learning/qa-track.md"),
    ],
    "DEVOPS": [
        (5, "Изучить Docker Compose и окружения", "Дни 1–5; 08_learning/devops-track.md"),
        (10, "Настроить безопасный учебный CI", "Дни 6–10; 08_learning/devops-track.md"),
        (18, "Изучить Beacon, сигналы мониторинга и runbooks", "Дни 11–20; 08_learning/devops-track.md"),
        (30, "Провести контролируемое расследование и демо", "Дни 21–30; 08_learning/devops-track.md"),
    ],
}


def demo_employee_id(role: Role) -> UUID:
    return uuid5(NAMESPACE_URL, f"firstday-ai-local-demo-employee-v1/{role}")


def _connection():
    return psycopg2.connect(**DB_CONFIG)


def _week(day: int) -> int:
    if day <= 5:
        return 1
    if day <= 10:
        return 2
    if day <= 20:
        return 3
    return 4


class TaskOut(BaseModel):
    id: UUID
    title: str
    description: str | None
    day_number: int
    week: int
    status: Status


class PlanOut(BaseModel):
    role: Role
    job_title: str
    plan_id: UUID
    title: str
    status: str
    total_tasks: int
    completed_tasks: int
    progress_percent: int
    tasks: list[TaskOut]
    demo: bool = True


class TaskUpdate(BaseModel):
    status: Status


def _get_plan(conn, role: Role) -> PlanOut:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """SELECT p.id, p.title, p.status
               FROM onboarding_plans p
               WHERE p.employee_id = %s""",
            (str(demo_employee_id(role)),),
        )
        plan = cur.fetchone()
        if not plan:
            raise HTTPException(status_code=404, detail="Демо-план ещё не создан. Выполните POST /init.")
        cur.execute(
            """SELECT id, title, description, day_number, status
               FROM onboarding_tasks
               WHERE plan_id = %s
               ORDER BY day_number, created_at, id""",
            (plan["id"],),
        )
        rows = cur.fetchall()
    tasks = [TaskOut(**row, week=_week(row["day_number"])) for row in rows]
    completed = sum(task.status == "DONE" for task in tasks)
    total = len(tasks)
    return PlanOut(
        role=role,
        job_title=ROLE_NAMES[role],
        plan_id=plan["id"],
        title=plan["title"],
        status=plan["status"],
        total_tasks=total,
        completed_tasks=completed,
        progress_percent=round(completed * 100 / total) if total else 0,
        tasks=tasks,
    )


@router.post("/init", response_model=PlanOut)
def initialize_demo_plan(role: Role = Query(default="JAVA")):
    """Idempotently create one isolated demo employee and task plan per role."""
    employee_id = demo_employee_id(role)
    try:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO employees
                       (id, full_name, email, role, job_title, department, start_date)
                       VALUES (%s, %s, %s, 'EMPLOYEE', %s, %s, %s)
                       ON CONFLICT (id) DO NOTHING""",
                    (
                        str(employee_id),
                        f"FirstDay Demo {role}",
                        f"firstday-demo-{role.lower()}@example.invalid",
                        ROLE_NAMES[role],
                        "Engineering" if role != "QA" else "Quality Engineering",
                        date.today(),
                    ),
                )
                cur.execute(
                    """INSERT INTO onboarding_plans (employee_id, title, status)
                       VALUES (%s, %s, 'NOT_STARTED')
                       ON CONFLICT (employee_id) DO NOTHING""",
                    (str(employee_id), f"30-дневная адаптация — {ROLE_NAMES[role]}"),
                )
                cur.execute("SELECT id FROM onboarding_plans WHERE employee_id = %s FOR UPDATE", (str(employee_id),))
                plan_id = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM onboarding_tasks WHERE plan_id = %s", (plan_id,))
                if cur.fetchone()[0] == 0:
                    for day, title, description in COMMON_TASKS + ROLE_TASKS[role]:
                        cur.execute(
                            """INSERT INTO onboarding_tasks
                               (plan_id, title, description, day_number, status)
                               VALUES (%s, %s, %s, %s, 'TODO')""",
                            (plan_id, title, description, day),
                        )
            return _get_plan(conn, role)
    except psycopg2.Error as exc:
        logger.exception("Failed to initialize onboarding demo")
        raise HTTPException(status_code=503, detail="База данных недоступна") from exc


@router.get("", response_model=PlanOut)
def get_demo_plan(role: Role = Query(default="JAVA")):
    try:
        with _connection() as conn:
            return _get_plan(conn, role)
    except psycopg2.Error as exc:
        logger.exception("Failed to get onboarding demo")
        raise HTTPException(status_code=503, detail="База данных недоступна") from exc


@router.patch("/tasks/{task_id}", response_model=PlanOut)
def update_demo_task(task_id: UUID, body: TaskUpdate, role: Role = Query(default="JAVA")):
    try:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM onboarding_plans WHERE employee_id = %s FOR UPDATE",
                    (str(demo_employee_id(role)),),
                )
                plan = cur.fetchone()
                if not plan:
                    raise HTTPException(status_code=404, detail="Демо-план не найден")
                plan_id = plan[0]
                cur.execute(
                    """UPDATE onboarding_tasks SET status = %s,
                       completed_at = CASE WHEN %s = 'DONE' THEN NOW() ELSE NULL END
                       WHERE id = %s AND plan_id = %s RETURNING id""",
                    (body.status, body.status, str(task_id), plan_id),
                )
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Задача не найдена в выбранном демо-плане")
                cur.execute(
                    """SELECT COUNT(*), COUNT(*) FILTER (WHERE status = 'DONE')
                       FROM onboarding_tasks WHERE plan_id = %s""",
                    (plan_id,),
                )
                total, done = cur.fetchone()
                new_status = "COMPLETED" if total and done == total else ("IN_PROGRESS" if done else "NOT_STARTED")
                cur.execute("UPDATE onboarding_plans SET status = %s WHERE id = %s", (new_status, plan_id))
            return _get_plan(conn, role)
    except psycopg2.Error as exc:
        logger.exception("Failed to update onboarding task")
        raise HTTPException(status_code=503, detail="База данных недоступна") from exc

"""Project onboarding. Membership is checked on the server for every project read/write."""
from __future__ import annotations

from typing import Literal
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from backend.auth_routes import DB_CONFIG, UserOut, current_user

router = APIRouter(prefix="/api/v1/projects", tags=["Project onboarding"])
ProjectRole = Literal["ENGINEERING", "QA", "DEVOPS", "GENERAL"]
TaskStatus = Literal["TODO", "IN_PROGRESS", "BLOCKED", "DONE"]


class ProjectOut(BaseModel):
    id: UUID
    slug: str
    name: str
    description: str
    is_demo: bool
    joined: bool
    project_role: str | None = None


class JoinBody(BaseModel):
    project_role: ProjectRole = "ENGINEERING"


class ProjectTask(BaseModel):
    id: UUID
    position: int
    stage: str
    title: str
    description: str
    document_path: str | None
    status: TaskStatus
    blocker_note: str


class ProjectPlan(BaseModel):
    project_id: UUID
    project_name: str
    project_role: str
    tasks: list[ProjectTask]
    total_tasks: int
    completed_tasks: int
    blocked_tasks: int
    progress_percent: int


class TaskUpdate(BaseModel):
    status: TaskStatus
    blocker_note: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def check_blocker(self):
        self.blocker_note = self.blocker_note.strip()
        if self.status == "BLOCKED" and not self.blocker_note:
            raise ValueError("Опишите, что мешает продолжить работу")
        if self.status != "BLOCKED":
            self.blocker_note = ""
        return self


def require_membership(cursor, project_id: UUID, employee_id: UUID) -> dict:
    cursor.execute(
        """SELECT p.id, p.slug, p.name, m.project_role
           FROM projects p JOIN project_memberships m ON m.project_id=p.id
           WHERE p.id=%s AND m.employee_id=%s AND m.is_active=TRUE""",
        (str(project_id), str(employee_id)),
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(404, "Проект не найден или недоступен")
    return dict(row)


def check_project_access(project_id: UUID, employee_id: UUID) -> None:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            require_membership(cursor, project_id, employee_id)


def project_plan(cursor, project_id: UUID, employee_id: UUID) -> ProjectPlan:
    member = require_membership(cursor, project_id, employee_id)
    cursor.execute(
        """SELECT id, position, stage, title, description, document_path, status, blocker_note
           FROM project_onboarding_tasks WHERE project_id=%s AND employee_id=%s
           ORDER BY position""", (str(project_id), str(employee_id)),
    )
    tasks = [ProjectTask(**row) for row in cursor.fetchall()]
    done = sum(task.status == "DONE" for task in tasks)
    return ProjectPlan(
        project_id=project_id, project_name=member["name"], project_role=member["project_role"],
        tasks=tasks, total_tasks=len(tasks), completed_tasks=done,
        blocked_tasks=sum(task.status == "BLOCKED" for task in tasks),
        progress_percent=round(100 * done / len(tasks)) if tasks else 0,
    )


def task_template(project_role: str) -> list[tuple[str, str, str, str]]:
    """Demo project milestones are appropriate for experienced hires as well as juniors."""
    tasks = [
        ("CONTEXT", "Разобраться в назначении проекта", "Прочитайте обзор и объясните основной пользовательский сценарий.", "overview.md"),
        ("CONTEXT", "Найти компоненты и ответственных", "Изучите карту компонентов. Уточните владельцев через Atlas; не угадывайте имена.", "architecture.md"),
        ("ENVIRONMENT", "Проверить необходимые доступы", "Сверьте доступы с инструкцией. При необходимости создайте заявку в Access Center.", "environment.md"),
    ]
    role_task = {
        "ENGINEERING": ("Подготовить рабочее окружение", "Следуйте README выбранного сервиса; проверьте тесты и health endpoint."),
        "QA": ("Подготовить тестовую среду", "Проверьте staging, тестовые данные и smoke tests по инструкции проекта."),
        "DEVOPS": ("Проверить окружения и наблюдаемость", "Найдите dashboard и runbook, проверьте read-only доступ."),
        "GENERAL": ("Проверить рабочий сценарий", "Пройдите основной сценарий в sandbox и запишите открытые вопросы."),
    }[project_role]
    tasks.append(("ENVIRONMENT", *role_task, "environment.md"))
    tasks.extend([
        ("DELIVERY", "Изучить процесс изменений", "Уточните правила review, проверки и согласования изменений.", "delivery.md"),
        ("DELIVERY", "Согласовать первую задачу", "Вместе с ответственным выберите небольшую задачу и критерии приёмки. Отметка здесь означает ваш отчёт, а не автоматическое одобрение.", "delivery.md"),
    ])
    return tasks


@router.get("", response_model=list[ProjectOut])
def list_projects(user: UserOut = Depends(current_user)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT p.id, p.slug, p.name, p.description, p.is_demo,
                              m.employee_id IS NOT NULL AS joined, m.project_role
                       FROM projects p LEFT JOIN project_memberships m
                         ON m.project_id=p.id AND m.employee_id=%s
                       WHERE m.is_active=TRUE ORDER BY p.name""", (str(user.id),),
                )
                return [ProjectOut(**row) for row in cursor.fetchall()]
    except psycopg2.Error as exc:
        raise HTTPException(503, "Проекты недоступны. Проверьте БД и миграции 005/006.") from exc


@router.post("/{project_id}/join", response_model=ProjectPlan)
def join_demo_project(project_id: UUID, body: JoinBody, user: UserOut = Depends(current_user)):
    """Compatibility endpoint: only return an already assigned plan; never grant access."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                return project_plan(cursor, project_id, user.id)
    except psycopg2.Error as exc:
        raise HTTPException(503, "Не удалось сохранить подключение к проекту") from exc


@router.get("/{project_id}/onboarding", response_model=ProjectPlan)
def get_project_plan(project_id: UUID, user: UserOut = Depends(current_user)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                return project_plan(cursor, project_id, user.id)
    except psycopg2.Error as exc:
        raise HTTPException(503, "План проекта недоступен") from exc


@router.patch("/{project_id}/onboarding/tasks/{task_id}", response_model=ProjectPlan)
def update_project_task(project_id: UUID, task_id: UUID, body: TaskUpdate, user: UserOut = Depends(current_user)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                require_membership(cursor, project_id, user.id)
                cursor.execute(
                    """UPDATE project_onboarding_tasks SET status=%s, blocker_note=%s, updated_at=NOW()
                       WHERE id=%s AND project_id=%s AND employee_id=%s RETURNING id""",
                    (body.status, body.blocker_note, str(task_id), str(project_id), str(user.id)),
                )
                if cursor.fetchone() is None:
                    raise HTTPException(404, "Задача не найдена")
                return project_plan(cursor, project_id, user.id)
    except psycopg2.Error as exc:
        raise HTTPException(503, "Не удалось сохранить задачу") from exc


def read_project_documents(project_id: UUID, employee_id: UUID, *, path: str | None = None) -> list[dict]:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            require_membership(cursor, project_id, employee_id)
            cursor.execute(
                """SELECT title, file_path, category, content, updated_at
                   FROM project_documents WHERE project_id=%s AND (%s IS NULL OR file_path=%s)
                   ORDER BY category, title""", (str(project_id), path, path),
            )
            return [dict(row) for row in cursor.fetchall()]


@router.get("/{project_id}/documents")
def list_project_documents(project_id: UUID, q: str = Query(default="", max_length=150),
                           category: str = Query(default="", max_length=100), user: UserOut = Depends(current_user)):
    try:
        documents = read_project_documents(project_id, user.id)
    except psycopg2.Error as exc:
        raise HTTPException(503, "Документы проекта недоступны") from exc
    categories: dict[str, int] = {}
    for doc in documents:
        categories[doc["category"]] = categories.get(doc["category"], 0) + 1
    term = q.strip().casefold()
    items = [{key: value for key, value in doc.items() if key != "content"} for doc in documents
             if (not category or doc["category"] == category)
             and (not term or term in f"{doc['title']} {doc['file_path']} {doc['content']}".casefold())]
    return {"items": items, "total": len(items), "categories": [{"name": key, "count": count} for key, count in categories.items()]}

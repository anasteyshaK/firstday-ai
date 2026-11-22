"""Administrator-only reporting and request review for FirstDay AI.

All endpoints require a valid authenticated ADMIN role. All data is local.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth_routes import UserOut, current_user, DB_CONFIG, EmploymentType, OnboardingMode
from backend.private_routes import ReviewBody
from backend.project_routes import ProjectRole, task_template

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["Admin Panel"])


def require_admin(user: UserOut = Depends(current_user)) -> UserOut:
    if user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Доступ только для администратора")
    return user


class Overview(BaseModel):
    employees_total: int
    plans_total: int
    completed_plans: int
    tasks_total: int
    tasks_done: int
    average_progress: int
    requests_pending: int
    requests_approved: int
    requests_rejected: int


class EmployeeRow(BaseModel):
    id: UUID
    full_name: str
    email: str
    role: str
    job_title: str
    department: str
    start_date: date
    employment_type: EmploymentType = 'EMPLOYEE'
    onboarding_mode: OnboardingMode = 'COMPANY'
    plan_status: str | None = None
    tasks_total: int
    tasks_done: int
    progress_percent: int


class RequestRow(BaseModel):
    id: UUID
    employee_id: UUID
    employee_name: str
    employee_email: str
    system_name: str
    reason: str
    status: str
    requested_at: datetime
    resolved_at: datetime | None = None


@router.get("/overview", response_model=Overview)
def overview(_: UserOut = Depends(require_admin)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""SELECT
                  (SELECT COUNT(*) FROM employees) AS employees_total,
                  (SELECT COUNT(*) FROM onboarding_plans) AS plans_total,
                  (SELECT COUNT(*) FROM onboarding_plans WHERE status='COMPLETED') AS completed_plans,
                  (SELECT COUNT(*) FROM onboarding_tasks) AS tasks_total,
                  (SELECT COUNT(*) FROM onboarding_tasks WHERE status='DONE') AS tasks_done,
                  (SELECT COUNT(*) FROM access_requests WHERE status='PENDING') AS requests_pending,
                  (SELECT COUNT(*) FROM access_requests WHERE status='APPROVED') AS requests_approved,
                  (SELECT COUNT(*) FROM access_requests WHERE status='REJECTED') AS requests_rejected
                """)
                data = dict(cur.fetchone())
                cur.execute("""SELECT COALESCE(ROUND(AVG(progress)), 0)::int AS average_progress
                    FROM (
                      SELECT 100.0 * COUNT(t.id) FILTER (WHERE t.status='DONE') /
                        NULLIF(COUNT(t.id), 0) AS progress
                      FROM onboarding_plans p
                      LEFT JOIN onboarding_tasks t ON t.plan_id=p.id
                      GROUP BY p.id
                    ) progress_per_plan""")
                data['average_progress'] = cur.fetchone()['average_progress']
        return Overview(**data)
    except psycopg2.Error as exc:
        logger.exception('Admin overview error')
        raise HTTPException(status_code=503, detail='База данных недоступна') from exc


@router.get("/employees", response_model=list[EmployeeRow])
def employees(_: UserOut = Depends(require_admin)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""SELECT e.id, e.full_name, e.email, e.role, e.job_title,
                       e.department, e.start_date, e.employment_type, e.onboarding_mode, p.status AS plan_status,
                       COUNT(t.id)::int AS tasks_total,
                       COUNT(t.id) FILTER (WHERE t.status='DONE')::int AS tasks_done,
                       COALESCE(ROUND(100.0 * COUNT(t.id) FILTER (WHERE t.status='DONE') /
                           NULLIF(COUNT(t.id), 0)), 0)::int AS progress_percent
                    FROM employees e
                    LEFT JOIN onboarding_plans p ON p.employee_id=e.id
                    LEFT JOIN onboarding_tasks t ON t.plan_id=p.id
                    GROUP BY e.id, e.full_name, e.email, e.role, e.job_title,
                             e.department, e.start_date, p.id, p.status
                    ORDER BY e.start_date DESC, e.full_name
                    LIMIT 500""")
                return [EmployeeRow(**row) for row in cur.fetchall()]
    except psycopg2.Error as exc:
        logger.exception('Admin employee list error')
        raise HTTPException(status_code=503, detail='База данных недоступна') from exc


@router.get("/access-requests", response_model=list[RequestRow])
def all_requests(_: UserOut = Depends(require_admin)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""SELECT r.id, r.employee_id, e.full_name AS employee_name,
                       e.email AS employee_email, r.system_name, r.reason, r.status,
                       r.requested_at, r.resolved_at
                    FROM access_requests r
                    JOIN employees e ON e.id=r.employee_id
                    ORDER BY (r.status='PENDING') DESC, r.requested_at DESC
                    LIMIT 500""")
                return [RequestRow(**row) for row in cur.fetchall()]
    except psycopg2.Error as exc:
        logger.exception('Admin requests error')
        raise HTTPException(status_code=503, detail='База данных недоступна') from exc


@router.patch("/access-requests/{request_id}/review", response_model=RequestRow)
def review(request_id: UUID, body: ReviewBody, _: UserOut = Depends(require_admin)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""UPDATE access_requests SET status=%s, resolved_at=NOW()
                    WHERE id=%s AND status='PENDING' RETURNING id""",
                    (body.status, str(request_id)))
                if not cur.fetchone():
                    raise HTTPException(status_code=409, detail='Заявка уже обработана или не найдена')
                cur.execute("""SELECT r.id, r.employee_id, e.full_name AS employee_name,
                       e.email AS employee_email, r.system_name, r.reason, r.status,
                       r.requested_at, r.resolved_at
                    FROM access_requests r JOIN employees e ON e.id=r.employee_id
                    WHERE r.id=%s""", (str(request_id),))
                row = cur.fetchone()
        return RequestRow(**row)
    except psycopg2.Error as exc:
        logger.exception('Admin review error')
        raise HTTPException(status_code=503, detail='База данных недоступна') from exc


class ProfileUpdate(BaseModel):
    employment_type: EmploymentType
    onboarding_mode: OnboardingMode


class AssignmentBody(BaseModel):
    project_role: ProjectRole = 'ENGINEERING'
    is_active: bool = True


class AdminProject(BaseModel):
    id: UUID
    name: str
    slug: str
    is_demo: bool


class AssignmentOut(BaseModel):
    project_id: UUID
    name: str
    project_role: ProjectRole
    is_active: bool


@router.patch('/employees/{employee_id}/profile', response_model=UserOut)
def update_profile(employee_id: UUID, body: ProfileUpdate, _: UserOut = Depends(require_admin)):
    """Employment/scenario fields never change the security role or project permissions."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('''UPDATE employees SET employment_type=%s, onboarding_mode=%s WHERE id=%s
                    RETURNING id, full_name, email, role, job_title, department, start_date,
                              employment_type, onboarding_mode''',
                    (body.employment_type, body.onboarding_mode, str(employee_id)))
                row = cur.fetchone()
                if row is None:
                    raise HTTPException(404, 'Сотрудник не найден')
                return UserOut(**row)
    except psycopg2.Error as exc:
        raise HTTPException(503, 'Не удалось сохранить профиль') from exc


@router.get('/projects', response_model=list[AdminProject])
def project_catalog(_: UserOut = Depends(require_admin)):
    """Admin catalog contains metadata only, not documents or another employee's chat."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT id, name, slug, is_demo FROM projects ORDER BY name')
                return [AdminProject(**row) for row in cur.fetchall()]
    except psycopg2.Error as exc:
        raise HTTPException(503, 'Каталог проектов недоступен') from exc


@router.get('/employees/{employee_id}/projects', response_model=list[AssignmentOut])
def employee_projects(employee_id: UUID, _: UserOut = Depends(require_admin)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT id FROM employees WHERE id=%s', (str(employee_id),))
                if cur.fetchone() is None:
                    raise HTTPException(404, 'Сотрудник не найден')
                cur.execute('''SELECT p.id AS project_id, p.name, m.project_role, m.is_active
                    FROM project_memberships m JOIN projects p ON p.id=m.project_id
                    WHERE m.employee_id=%s ORDER BY p.name''', (str(employee_id),))
                return [AssignmentOut(**row) for row in cur.fetchall()]
    except psycopg2.Error as exc:
        raise HTTPException(503, 'Назначения недоступны') from exc


@router.put('/employees/{employee_id}/projects/{project_id}', response_model=AssignmentOut)
def assign_project(employee_id: UUID, project_id: UUID, body: AssignmentBody,
                   admin: UserOut = Depends(require_admin)):
    """Grant/reactivate/revoke membership atomically; retain all personal progress/history."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT id, slug, name FROM projects WHERE id=%s FOR UPDATE', (str(project_id),))
                project = cur.fetchone()
                if project is None:
                    raise HTTPException(404, 'Проект не найден')
                cur.execute('SELECT id FROM employees WHERE id=%s FOR UPDATE', (str(employee_id),))
                if cur.fetchone() is None:
                    raise HTTPException(404, 'Сотрудник не найден')
                cur.execute('''SELECT project_role, is_active FROM project_memberships
                    WHERE project_id=%s AND employee_id=%s FOR UPDATE''', (str(project_id), str(employee_id)))
                existing = cur.fetchone()
                if not body.is_active:
                    if existing is None:
                        raise HTTPException(404, 'Назначение не найдено')
                    cur.execute('''UPDATE project_memberships SET is_active=FALSE, assigned_by=%s, updated_at=NOW()
                        WHERE project_id=%s AND employee_id=%s''', (str(admin.id), str(project_id), str(employee_id)))
                    return AssignmentOut(project_id=project_id, name=project['name'],
                                         project_role=existing['project_role'], is_active=False)
                if existing is not None and existing['project_role'] != body.project_role:
                    raise HTTPException(409, 'Роль уже закреплена за маршрутом. Для восстановления выберите прежнюю роль; прогресс сохранён.')
                cur.execute('''INSERT INTO project_memberships
                    (project_id, employee_id, project_role, is_active, assigned_by)
                    VALUES (%s,%s,%s,TRUE,%s) ON CONFLICT (project_id, employee_id)
                    DO UPDATE SET is_active=TRUE, assigned_by=EXCLUDED.assigned_by, updated_at=NOW()''',
                    (str(project_id), str(employee_id), body.project_role, str(admin.id)))
                for index, (stage, title, description, filename) in enumerate(task_template(body.project_role), 1):
                    cur.execute('''INSERT INTO project_onboarding_tasks
                        (project_id, employee_id, position, stage, title, description, document_path)
                        VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (project_id, employee_id, position) DO NOTHING''',
                        (str(project_id), str(employee_id), index, stage, title, description,
                         f"projects/{project['slug']}/{filename}"))
                return AssignmentOut(project_id=project_id, name=project['name'],
                                     project_role=body.project_role, is_active=True)
    except psycopg2.Error as exc:
        raise HTTPException(503, 'Не удалось изменить назначение') from exc

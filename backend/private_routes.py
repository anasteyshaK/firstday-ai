"""Private onboarding and access request endpoints bound to authenticated employee id."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.auth_routes import UserOut, current_user
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / 'rag'
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
from search import DB_CONFIG

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/v1', tags=['My Data'])
TaskStatus = Literal['TODO', 'IN_PROGRESS', 'DONE']
ReqStatus = Literal['PENDING', 'APPROVED', 'REJECTED']


class TaskOut(BaseModel):
    id: UUID
    title: str
    description: str | None
    day_number: int
    week: int
    status: TaskStatus


class PlanOut(BaseModel):
    job_title: str
    plan_id: UUID
    title: str
    status: str
    total_tasks: int
    completed_tasks: int
    progress_percent: int
    tasks: list[TaskOut]
    demo: bool = False


class TaskUpdate(BaseModel):
    status: TaskStatus


class AccessRequestCreate(BaseModel):
    system_name: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=1000)


class AccessRequestOut(BaseModel):
    id: UUID
    system_name: str
    reason: str
    status: ReqStatus
    requested_at: datetime
    resolved_at: datetime | None = None


class AccessRequestsResponse(BaseModel):
    demo: bool = False
    items: list[AccessRequestOut]
    total: int


class ReviewBody(BaseModel):
    status: Literal['APPROVED', 'REJECTED']


def _week(day: int) -> int:
    return 1 if day <= 5 else 2 if day <= 10 else 3 if day <= 20 else 4


def _plan(conn, employee_id: UUID, job_title: str) -> PlanOut:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            'SELECT id, title, status FROM onboarding_plans WHERE employee_id=%s',
            (str(employee_id),),
        )
        plan = cur.fetchone()
        if not plan:
            raise HTTPException(404, 'Персональный план не найден')
        cur.execute(
            '''SELECT id, title, description, day_number, status
               FROM onboarding_tasks WHERE plan_id=%s ORDER BY day_number, created_at, id''',
            (str(plan['id']),),
        )
        tasks = [TaskOut(**row, week=_week(row['day_number'])) for row in cur.fetchall()]
    completed = sum(t.status == 'DONE' for t in tasks)
    return PlanOut(
        job_title=job_title, plan_id=plan['id'], title=plan['title'], status=plan['status'],
        total_tasks=len(tasks), completed_tasks=completed,
        progress_percent=round(100 * completed / len(tasks)) if tasks else 0, tasks=tasks,
    )


@router.get('/onboarding', response_model=PlanOut)
def get_my_plan(user: UserOut = Depends(current_user)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            return _plan(conn, user.id, user.job_title)
    except psycopg2.Error as exc:
        logger.exception('Get onboarding failed')
        raise HTTPException(503, 'База данных недоступна') from exc


@router.patch('/onboarding/tasks/{task_id}', response_model=PlanOut)
def update_my_task(task_id: UUID, body: TaskUpdate, user: UserOut = Depends(current_user)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                # Authorization enforced in UPDATE via employee_id join.
                cur.execute(
                    '''UPDATE onboarding_tasks t SET
                           status=%s,
                           completed_at=CASE WHEN %s='DONE' THEN NOW() ELSE NULL END
                       FROM onboarding_plans p
                       WHERE t.plan_id=p.id AND t.id=%s AND p.employee_id=%s
                       RETURNING t.id''',
                    (body.status, body.status, str(task_id), str(user.id)),
                )
                if not cur.fetchone():
                    raise HTTPException(404, 'Задача не найдена')
                cur.execute(
                    '''SELECT p.id FROM onboarding_plans p WHERE p.employee_id=%s FOR UPDATE''',
                    (str(user.id),),
                )
                plan_id = cur.fetchone()[0]
                cur.execute(
                    '''SELECT COUNT(*), COUNT(*) FILTER (WHERE status='DONE')
                       FROM onboarding_tasks WHERE plan_id=%s''',
                    (plan_id,),
                )
                total, done = cur.fetchone()
                status = 'COMPLETED' if total and done == total else 'IN_PROGRESS' if done else 'NOT_STARTED'
                cur.execute('UPDATE onboarding_plans SET status=%s WHERE id=%s', (status, plan_id))
            return _plan(conn, user.id, user.job_title)
    except psycopg2.Error as exc:
        logger.exception('Update task failed')
        raise HTTPException(503, 'База данных недоступна') from exc


@router.get('/access-requests', response_model=AccessRequestsResponse)
def my_requests(user: UserOut = Depends(current_user)):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    '''SELECT id, system_name, reason, status, requested_at, resolved_at
                       FROM access_requests WHERE employee_id=%s
                       ORDER BY requested_at DESC, id DESC''',
                    (str(user.id),),
                )
                items = [AccessRequestOut(**row) for row in cur.fetchall()]
        return AccessRequestsResponse(items=items, total=len(items))
    except psycopg2.Error as exc:
        logger.exception('List requests failed')
        raise HTTPException(503, 'База данных недоступна') from exc


@router.post('/access-requests', response_model=AccessRequestOut, status_code=201)
def create_my_request(body: AccessRequestCreate, user: UserOut = Depends(current_user)):
    if not body.system_name.strip() or not body.reason.strip():
        raise HTTPException(422, 'Заполните инструмент и причину')
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    '''INSERT INTO access_requests (employee_id, system_name, reason, status)
                       VALUES (%s,%s,%s,'PENDING')
                       RETURNING id, system_name, reason, status, requested_at, resolved_at''',
                    (str(user.id), body.system_name.strip(), body.reason.strip()),
                )
                row = cur.fetchone()
        return AccessRequestOut(**row)
    except psycopg2.Error as exc:
        logger.exception('Create request failed')
        raise HTTPException(503, 'База данных недоступна') from exc


@router.get('/admin/access-requests', response_model=list[AccessRequestOut])
def admin_requests(user: UserOut = Depends(current_user)):
    if user.role != 'ADMIN':
        raise HTTPException(403, 'Доступ только для администратора')
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    '''SELECT id, system_name, reason, status, requested_at, resolved_at
                       FROM access_requests ORDER BY requested_at DESC'''
                )
                return [AccessRequestOut(**r) for r in cur.fetchall()]
    except psycopg2.Error as exc:
        raise HTTPException(503, 'База данных недоступна') from exc


@router.patch('/admin/access-requests/{request_id}/review', response_model=AccessRequestOut)
def review_request(request_id: UUID, body: ReviewBody, user: UserOut = Depends(current_user)):
    if user.role != 'ADMIN':
        raise HTTPException(403, 'Доступ только для администратора')
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    '''UPDATE access_requests SET status=%s, resolved_at=NOW()
                       WHERE id=%s AND status='PENDING'
                       RETURNING id, system_name, reason, status, requested_at, resolved_at''',
                    (body.status, str(request_id)),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(404, 'Ожидающая заявка не найдена')
        return AccessRequestOut(**row)
    except psycopg2.Error as exc:
        logger.exception('Review request failed')
        raise HTTPException(503, 'База данных недоступна') from exc

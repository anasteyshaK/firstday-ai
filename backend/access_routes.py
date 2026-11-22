"""Local demonstration access-request API. No authentication: NEVER expose publicly."""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / 'rag'
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
from search import DB_CONFIG
from backend.onboarding_routes import demo_employee_id, ROLE_NAMES

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/v1/demo/access-requests', tags=['Access Center demo'])
Role = Literal['JAVA', 'QA', 'DEVOPS']
RequestStatus = Literal['PENDING', 'APPROVED', 'REJECTED']


class AccessRequestCreate(BaseModel):
    system_name: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=1000)

    @field_validator('system_name', 'reason')
    @classmethod
    def strip_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('Field must not be blank')
        return value


class AccessRequestStatusUpdate(BaseModel):
    status: Literal['APPROVED', 'REJECTED']


class AccessRequestOut(BaseModel):
    id: UUID
    system_name: str
    reason: str
    status: RequestStatus
    requested_at: datetime
    resolved_at: datetime | None = None


class AccessRequestsResponse(BaseModel):
    role: Role
    demo: bool = True
    items: list[AccessRequestOut]
    total: int


def _ensure_demo_employee(cursor, role: Role) -> None:
    """Same deterministic demo identity as My Onboarding; no real accounts."""
    cursor.execute(
        '''INSERT INTO employees (id, full_name, email, role, job_title, department, start_date)
           VALUES (%s, %s, %s, 'EMPLOYEE', %s, %s, %s)
           ON CONFLICT (id) DO NOTHING''',
        (str(demo_employee_id(role)), f'FirstDay Demo {role}',
         f'firstday-demo-{role.lower()}@example.invalid', ROLE_NAMES[role],
         'Quality Engineering' if role == 'QA' else 'Engineering', date.today()),
    )


def _list_requests(cursor, role: Role) -> AccessRequestsResponse:
    cursor.execute(
        '''SELECT id, system_name, reason, status, requested_at, resolved_at
           FROM access_requests WHERE employee_id = %s
           ORDER BY requested_at DESC, id DESC''',
        (str(demo_employee_id(role)),),
    )
    items = [AccessRequestOut(**row) for row in cursor.fetchall()]
    return AccessRequestsResponse(role=role, items=items, total=len(items))


@router.get('', response_model=AccessRequestsResponse)
def get_requests(role: Role = Query(default='JAVA')):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                return _list_requests(cursor, role)
    except psycopg2.Error as exc:
        logger.exception('Failed to list access requests')
        raise HTTPException(503, 'Database is unavailable') from exc


@router.post('', response_model=AccessRequestOut, status_code=201)
def create_request(body: AccessRequestCreate, role: Role = Query(default='JAVA')):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                _ensure_demo_employee(cursor, role)
                cursor.execute(
                    '''INSERT INTO access_requests (employee_id, system_name, reason, status)
                       VALUES (%s, %s, %s, 'PENDING')
                       RETURNING id, system_name, reason, status, requested_at, resolved_at''',
                    (str(demo_employee_id(role)), body.system_name, body.reason),
                )
                return AccessRequestOut(**cursor.fetchone())
    except psycopg2.Error as exc:
        logger.exception('Failed to create access request')
        raise HTTPException(503, 'Database is unavailable') from exc


@router.patch('/{request_id}/demo-review', response_model=AccessRequestOut)
def demo_review_request(
    request_id: UUID,
    body: AccessRequestStatusUpdate,
    role: Role = Query(default='JAVA'),
):
    """Simulated decision only: does not grant access to any external tool."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    '''UPDATE access_requests
                       SET status = %s, resolved_at = NOW()
                       WHERE id = %s AND employee_id = %s AND status = 'PENDING'
                       RETURNING id, system_name, reason, status, requested_at, resolved_at''',
                    (body.status, str(request_id), str(demo_employee_id(role))),
                )
                row = cursor.fetchone()
                if row is None:
                    raise HTTPException(404, 'Pending demo request not found')
                return AccessRequestOut(**row)
    except psycopg2.Error as exc:
        logger.exception('Failed to simulate review')
        raise HTTPException(503, 'Database is unavailable') from exc

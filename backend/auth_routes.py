"""FirstDay AI local auth: opaque bearer sessions + PBKDF2 password hashing.

For a public deployment: add rate limiting, email verification, password reset,
TLS and a full security audit. Never expose legacy /demo routes alongside these.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import date, datetime, timedelta, timezone
from uuid import UUID
from typing import Literal

import psycopg2
from psycopg2 import errors
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from backend.onboarding_routes import ROLE_NAMES, COMMON_TASKS, ROLE_TASKS
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
from search import DB_CONFIG

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)
SESSION_LIFETIME = timedelta(days=7)
ITERATIONS = 600_000
EmploymentType = Literal['INTERN', 'EMPLOYEE']
OnboardingMode = Literal['COMPANY', 'PROJECT']


class RegisterBody(BaseModel):
    full_name: str = Field(min_length=2, max_length=150)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=12, max_length=128)
    job_track: str = Field(pattern=r"^(JAVA|QA|DEVOPS)$")
    employment_type: EmploymentType = 'EMPLOYEE'
    onboarding_mode: OnboardingMode | None = None

    @field_validator("email")
    @classmethod
    def email_format(cls, value: str) -> str:
        value = value.strip().lower()
        if value.count("@") != 1 or not value.split("@", 1)[1].count("."):
            raise ValueError("Введите корректный email")
        return value


class LoginBody(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: UUID
    full_name: str
    email: str
    role: str
    job_title: str
    department: str
    start_date: date
    employment_type: EmploymentType = 'EMPLOYEE'
    onboarding_mode: OnboardingMode = 'COMPANY'


class AuthOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str | None) -> bool:
    if not encoded:
        return False
    try:
        algorithm, count, salt_hex, digest_hex = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        count_int = int(count)
        if not 300_000 <= count_int <= 1_000_000:
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), count_int
        )
        return hmac.compare_digest(expected, actual)
    except (ValueError, TypeError):
        return False


def issue_session(cursor, employee_id: UUID) -> str:
    raw_token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    expires = datetime.now(timezone.utc) + SESSION_LIFETIME
    cursor.execute(
        "INSERT INTO auth_sessions (employee_id, token_hash, expires_at) VALUES (%s,%s,%s)",
        (str(employee_id), token_hash, expires),
    )
    return raw_token


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> UserOut:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Войдите в аккаунт", headers={"WWW-Authenticate": "Bearer"})
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT e.id, e.full_name, e.email, e.role, e.job_title,
                              e.department, e.start_date, e.employment_type, e.onboarding_mode
                       FROM auth_sessions s JOIN employees e ON e.id = s.employee_id
                       WHERE s.token_hash = %s AND s.revoked_at IS NULL
                         AND s.expires_at > NOW()""",
                    (token_hash,),
                )
                row = cursor.fetchone()
    except psycopg2.Error as exc:
        log.exception("Auth database error")
        raise HTTPException(503, "База данных недоступна") from exc
    if not row:
        raise HTTPException(401, "Сессия недействительна или истекла")
    return UserOut(**row)


@router.post("/register", response_model=AuthOut, status_code=201)
def register(body: RegisterBody):
    # Registration selects a profession, not the employee's seniority.
    role_name = ROLE_NAMES[body.job_track].removeprefix("Junior ")
    department = "Quality Engineering" if body.job_track == "QA" else "Engineering"
    onboarding_mode = body.onboarding_mode or ('COMPANY' if body.employment_type == 'INTERN' else 'PROJECT')
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """INSERT INTO employees
                       (full_name, email, password_hash, role, job_title, department, start_date, employment_type, onboarding_mode)
                       VALUES (%s,%s,%s,'EMPLOYEE',%s,%s,CURRENT_DATE,%s,%s)
                       RETURNING id, full_name, email, role, job_title, department, start_date, employment_type, onboarding_mode""",
                    (body.full_name.strip(), body.email, hash_password(body.password), role_name, department,
                     body.employment_type, onboarding_mode),
                )
                user = UserOut(**cursor.fetchone())
                # One private plan per employee, not shared by job title.
                cursor.execute(
                    """INSERT INTO onboarding_plans (employee_id, title, status)
                       VALUES (%s,%s,'NOT_STARTED') RETURNING id""",
                    (str(user.id), f"30-дневная адаптация — {role_name}"),
                )
                plan_id = cursor.fetchone()["id"]
                for day, title, description in COMMON_TASKS + ROLE_TASKS[body.job_track]:
                    cursor.execute(
                        """INSERT INTO onboarding_tasks (plan_id, title, description, day_number, status)
                           VALUES (%s,%s,%s,%s,'TODO')""",
                        (str(plan_id), title, description, day),
                    )
                token = issue_session(cursor, user.id)
        return AuthOut(access_token=token, user=user)
    except errors.UniqueViolation as exc:
        raise HTTPException(409, "Аккаунт с таким email уже существует") from exc
    except psycopg2.Error as exc:
        log.exception("Registration failed")
        raise HTTPException(503, "База данных недоступна или миграция не выполнена") from exc


@router.post("/login", response_model=AuthOut)
def login(body: LoginBody):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """SELECT id, full_name, email, role, job_title, department,
                              start_date, employment_type, onboarding_mode, password_hash FROM employees WHERE email=%s""",
                    (body.email.strip().lower(),),
                )
                row = cursor.fetchone()
                if not row or not verify_password(body.password, row.get("password_hash")):
                    raise HTTPException(401, "Неверный email или пароль")
                user = UserOut(**row)
                token = issue_session(cursor, user.id)
        return AuthOut(access_token=token, user=user)
    except psycopg2.Error as exc:
        log.exception("Login database error")
        raise HTTPException(503, "База данных недоступна") from exc


@router.get("/me", response_model=UserOut)
def me(user: UserOut = Depends(current_user)):
    return user


@router.post("/logout", status_code=204)
def logout(credentials: HTTPAuthorizationCredentials | None = Depends(security),
           _: UserOut = Depends(current_user)):
    token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE auth_sessions SET revoked_at=NOW() WHERE token_hash=%s", (token_hash,))
    except psycopg2.Error as exc:
        raise HTTPException(503, "База данных недоступна") from exc

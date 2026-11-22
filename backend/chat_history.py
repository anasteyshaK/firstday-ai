
"""Authenticated chat history for FirstDay AI.

All database operations are scoped to the authenticated employee.
The existing PostgreSQL schema uses uppercase sender values:
USER and ASSISTANT.
"""

from __future__ import annotations

import json
import re
from uuid import UUID

import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from backend.auth_routes import DB_CONFIG, UserOut, current_user
from backend.project_routes import require_membership


router = APIRouter(
    prefix="/api/v1/chat/conversations",
    tags=["AI Assistant history"],
)


class ConversationInfo(BaseModel):
    id: UUID
    title: str
    created_at: str
    updated_at: str
    project_id: UUID | None = None


class StoredMessage(BaseModel):
    id: UUID
    role: str
    content: str
    sources: list[dict]
    created_at: str


class ConversationDetail(BaseModel):
    id: UUID
    title: str
    messages: list[StoredMessage]
    project_id: UUID | None = None


def same_project(stored: UUID | str | None, requested: UUID | None) -> bool:
    """psycopg2 UUID columns may be strings if no UUID adapter is registered."""
    return (str(stored) if stored is not None else None) == (str(requested) if requested is not None else None)


def _find_owned(
    cursor,
    conversation_id: UUID,
    employee_id: UUID,
):
    """Find a conversation belonging to the current employee."""

    cursor.execute(
        """
        SELECT id, title, project_id
        FROM chat_conversations
        WHERE id = %s
          AND employee_id = %s
        """,
        (
            str(conversation_id),
            str(employee_id),
        ),
    )

    row = cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    if row["project_id"] is not None:
        require_membership(cursor, row["project_id"], employee_id)

    return row


def recent_questions(
    conversation_id: UUID,
    employee_id: UUID,
    project_id: UUID | None = None,
) -> list[str]:
    """Return recent user questions for the current employee."""

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:

            conversation = _find_owned(
                cur,
                conversation_id,
                employee_id,
            )
            if not same_project(conversation["project_id"], project_id):
                raise HTTPException(409, "Начните новый чат для выбранного проекта")

            cur.execute(
                """
                SELECT content
                FROM chat_messages
                WHERE conversation_id = %s
                  AND employee_id = %s
                  AND sender = 'USER'
                ORDER BY created_at DESC, id DESC
                LIMIT 6
                """,
                (
                    str(conversation_id),
                    str(employee_id),
                ),
            )

            rows = cur.fetchall()

    return [
        row["content"]
        for row in reversed(rows)
    ]


def save_exchange(
    employee_id: UUID,
    conversation_id: UUID | None,
    question: str,
    answer: str,
    sources: list[dict],
    project_id: UUID | None = None,
) -> UUID:
    """Save one user question and one assistant answer atomically."""

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:

            if project_id is not None:
                require_membership(cur, project_id, employee_id)

            if conversation_id is None:

                title = (
                    re.sub(r"\s+", " ", question)
                    .strip()[:120]
                    or "New conversation"
                )

                cur.execute(
                    """
                    INSERT INTO chat_conversations (
                        employee_id,
                        title,
                        project_id
                    )
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (
                        str(employee_id),
                        title,
                        str(project_id) if project_id is not None else None,
                    ),
                )

                conversation_id = cur.fetchone()["id"]

            else:
                conversation = _find_owned(
                    cur,
                    conversation_id,
                    employee_id,
                )
                if not same_project(conversation["project_id"], project_id):
                    raise HTTPException(409, "Начните новый чат для выбранного проекта")

            # Save the employee's message.
            # PostgreSQL CHECK constraint requires 'USER'.

            cur.execute(
                """
                INSERT INTO chat_messages (
                    employee_id,
                    conversation_id,
                    sender,
                    content,
                    sources
                )
                VALUES (
                    %s,
                    %s,
                    'USER',
                    %s,
                    '[]'::jsonb
                )
                """,
                (
                    str(employee_id),
                    str(conversation_id),
                    question,
                ),
            )

            # Save the AI response.
            # PostgreSQL CHECK constraint requires 'ASSISTANT'.

            cur.execute(
                """
                INSERT INTO chat_messages (
                    employee_id,
                    conversation_id,
                    sender,
                    content,
                    sources
                )
                VALUES (
                    %s,
                    %s,
                    'ASSISTANT',
                    %s,
                    %s::jsonb
                )
                """,
                (
                    str(employee_id),
                    str(conversation_id),
                    answer,
                    json.dumps(
                        sources,
                        ensure_ascii=False,
                    ),
                ),
            )

            cur.execute(
                """
                UPDATE chat_conversations
                SET updated_at = NOW()
                WHERE id = %s
                  AND employee_id = %s
                """,
                (
                    str(conversation_id),
                    str(employee_id),
                ),
            )

    return conversation_id


@router.get(
    "",
    response_model=list[ConversationInfo],
)
def list_conversations(
    project_id: UUID | None = None,
    user: UserOut = Depends(current_user),
):
    """List conversations of the authenticated employee."""

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(
                cursor_factory=RealDictCursor
            ) as cur:

                if project_id is not None:
                    require_membership(cur, project_id, user.id)

                cur.execute(
                    """
                    SELECT
                        id,
                        title,
                        project_id,
                        created_at,
                        updated_at
                    FROM chat_conversations
                    WHERE employee_id = %s
                      AND project_id IS NOT DISTINCT FROM %s::uuid
                    ORDER BY updated_at DESC, id DESC
                    LIMIT 100
                    """,
                    (str(user.id), str(project_id) if project_id is not None else None),
                )

                rows = cur.fetchall()

        return [
            ConversationInfo(
                **{
                    **row,
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                }
            )
            for row in rows
        ]

    except psycopg2.Error as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat history unavailable",
        ) from exc


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
)
def get_conversation(
    conversation_id: UUID,
    user: UserOut = Depends(current_user),
):
    """Read messages of a conversation owned by the employee."""

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(
                cursor_factory=RealDictCursor
            ) as cur:

                row = _find_owned(
                    cur,
                    conversation_id,
                    user.id,
                )

                cur.execute(
                    """
                    SELECT
                        id,
                        LOWER(sender) AS role,
                        content,
                        COALESCE(
                            sources,
                            '[]'::jsonb
                        ) AS sources,
                        created_at
                    FROM chat_messages
                    WHERE conversation_id = %s
                      AND employee_id = %s
                    ORDER BY created_at, id
                    LIMIT 300
                    """,
                    (
                        str(conversation_id),
                        str(user.id),
                    ),
                )

                messages = cur.fetchall()

        return ConversationDetail(
            id=row["id"],
            title=row["title"],
            project_id=row["project_id"],
            messages=[
                StoredMessage(
                    **{
                        **message,
                        "created_at": message[
                            "created_at"
                        ].isoformat(),
                    }
                )
                for message in messages
            ],
        )

    except psycopg2.Error as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat history unavailable",
        ) from exc


@router.delete(
    "/{conversation_id}",
    status_code=204,
)
def delete_conversation(
    conversation_id: UUID,
    user: UserOut = Depends(current_user),
):
    """Delete a conversation owned by the employee."""

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(
                cursor_factory=RealDictCursor
            ) as cur:

                _find_owned(
                    cur,
                    conversation_id,
                    user.id,
                )

                # Explicitly delete child messages first.
                # Works even if the FK has no ON DELETE CASCADE.
                cur.execute(
                    """
                    DELETE FROM chat_messages
                    WHERE conversation_id = %s
                      AND employee_id = %s
                    """,
                    (
                        str(conversation_id),
                        str(user.id),
                    ),
                )

                cur.execute(
                    """
                    DELETE FROM chat_conversations
                    WHERE id = %s
                      AND employee_id = %s
                    """,
                    (
                        str(conversation_id),
                        str(user.id),
                    ),
                )

    except psycopg2.Error as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat history unavailable",
        ) from exc

"""Authenticated document listing and viewing; employee-level content only."""
import re
from datetime import datetime
from uuid import UUID

import psycopg2
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from search import DB_CONFIG
from backend.auth_routes import current_user
from backend.auth_routes import UserOut
from backend.project_routes import read_project_documents


document_router = APIRouter(prefix="/api/v1/documents", tags=["Knowledge Hub"], dependencies=[Depends(current_user)])


class DocumentResponse(BaseModel):
    title: str
    file_path: str
    category: str
    content: str


class DocumentSummary(BaseModel):
    title: str
    file_path: str
    category: str
    department: str | None
    updated_at: datetime


class DocumentCategory(BaseModel):
    name: str
    count: int


class DocumentListResponse(BaseModel):
    items: list[DocumentSummary]
    total: int
    categories: list[DocumentCategory]


@document_router.get("", response_model=DocumentListResponse)
def list_documents(
    q: str = Query(default="", max_length=150),
    category: str = Query(default="", max_length=100),
):
    """Search permitted documents by title, filename, category or chunk text."""
    term = q.strip()
    category = category.strip()
    # Escape LIKE wildcards so the query is a literal text search.
    escaped = term.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    pattern = f"%{escaped}%"

    try:
        with psycopg2.connect(**DB_CONFIG) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT category, COUNT(*)
                    FROM documents
                    WHERE access_level = 'EMPLOYEE'
                    GROUP BY category
                    ORDER BY category
                    """
                )
                categories = [DocumentCategory(name=row[0], count=row[1]) for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT d.title, d.file_path, d.category, d.department, d.updated_at
                    FROM documents AS d
                    WHERE d.access_level = 'EMPLOYEE'
                      AND (%s = '' OR d.category = %s)
                      AND (
                        %s = ''
                        OR d.title ILIKE %s ESCAPE '!'
                        OR d.file_path ILIKE %s ESCAPE '!'
                        OR d.category ILIKE %s ESCAPE '!'
                        OR EXISTS (
                            SELECT 1 FROM document_chunks AS c
                            WHERE c.document_id = d.id
                              AND c.content ILIKE %s ESCAPE '!'
                        )
                      )
                    ORDER BY d.category, d.title, d.file_path
                    LIMIT 200
                    """,
                    (category, category, term, pattern, pattern, pattern, pattern),
                )
                rows = cursor.fetchall()

    except psycopg2.Error as error:
        raise HTTPException(status_code=503, detail="Document storage unavailable") from error

    items = [
        DocumentSummary(
            title=row[0], file_path=row[1], category=row[2],
            department=row[3], updated_at=row[4]
        )
        for row in rows
    ]
    return DocumentListResponse(items=items, total=len(items), categories=categories)


@document_router.get("/content", response_model=DocumentResponse)
def get_document_content(path: str = Query(min_length=5, max_length=400),
                         project_id: UUID | None = None,
                         user: UserOut = Depends(current_user)):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*\.md", path):
        raise HTTPException(status_code=400, detail="Invalid document path")

    try:
        if project_id is not None:
            documents = read_project_documents(project_id, user.id, path=path)
            if not documents:
                raise HTTPException(404, "Document not found")
            return DocumentResponse(**documents[0])
        with psycopg2.connect(**DB_CONFIG) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.title, d.file_path, d.category,
                           string_agg(c.content, E'\n\n' ORDER BY c.chunk_index)
                    FROM documents AS d
                    JOIN document_chunks AS c ON c.document_id = d.id
                    WHERE d.file_path = %s AND d.access_level = 'EMPLOYEE'
                    GROUP BY d.id, d.title, d.file_path, d.category
                    """,
                    (path,),
                )
                row = cursor.fetchone()
    except psycopg2.Error as error:
        raise HTTPException(status_code=503, detail="Document storage unavailable") from error

    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(title=row[0], file_path=row[1], category=row[2], content=row[3])

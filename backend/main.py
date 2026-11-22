"""FirstDay AI API.

FastAPI endpoints for:
- Authentication
- Employee onboarding
- Knowledge Hub
- Access Center
- Admin Panel
- AI Assistant
- Conversation history
- Real Ollama streaming
"""

from __future__ import annotations

import json
import logging
import sys

from pathlib import Path
from uuid import UUID

import psycopg2

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "rag"

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


# ============================================================
# IMPORTS
# ============================================================

from answer import ask_firstday, stream_firstday
from telemetry import start_observation, observed

from backend.auth_routes import (
    router as auth_router,
    UserOut,
    current_user,
)

from backend.document_routes import document_router

from backend.private_routes import (
    router as private_router,
)

from backend.admin_routes import (
    router as admin_router,
)

from backend.chat_history import (
    router as chat_history_router,
    recent_questions,
    save_exchange,
)

from backend.chat_context import contextual_question, is_followup, extract_topics
from backend.project_routes import router as project_router, check_project_access
from backend.project_chat import project_personal_answer, ask_project, stream_project

from backend.personal_chat import (
    personal_answer,
    resolve_personal_question,
    _intent,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("firstday-api")


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="FirstDay AI API",
    version="0.5.0",
    description=(
        "Authenticated employee onboarding assistant "
        "with real Ollama streaming"
    ),
)

app.include_router(auth_router)
app.include_router(document_router)
app.include_router(private_router)
app.include_router(admin_router)
app.include_router(chat_history_router)
app.include_router(project_router)


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    conversation_id: UUID | None = None
    project_id: UUID | None = None


class SourceResponse(BaseModel):
    reference: int
    title: str
    file_path: str
    similarity: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    conversation_id: UUID


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/api/v1/health",
    tags=["Health"],
)
def health():
    return {
        "status": "ok",
        "service": "FirstDay AI API",
        "version": "0.5.0",
        "streaming": "ollama",
    }


# ============================================================
# CHAT CONTEXT
# ============================================================

def build_document_questions(
    original_question: str,
    previous_questions: list[str],
    project_context: bool = False,
) -> tuple[str, str, str | None]:
    """Separate retrieval question from generation question.

    Retrieval can use an expanded question.

    Generation receives the original employee question and
    the previous documentation topic for genuine follow-ups.

    Employee records and private database answers are never
    added to the RAG prompt.
    """

    original = original_question.strip()

    retrieval = contextual_question(
        original,
        previous_questions,
    )

    previous_topic = None

    # Project questions about architecture/owners need context too, even when
    # no corporate tool/role keyword was mentioned. Never inherit a personal intent.
    if (project_context and retrieval == original and previous_questions
            and is_followup(original) and not extract_topics(original)
            and _intent(original) is None and _intent(previous_questions[-1]) is None):
        retrieval = f"Предыдущий вопрос: {previous_questions[-1][:250]}\nУточнение: {original}"

    if (
        retrieval != original
        and previous_questions
        and _intent(original) is None
    ):
        previous = previous_questions[-1].strip()

        if previous and _intent(previous) is None:
            previous_topic = previous

    return retrieval, original, previous_topic


def choose_personal_question(
    question: str,
    previous_questions: list[str],
    retrieval_question: str,
) -> str:
    """Prefer the explicit personal intent before RAG contextual expansion.

    This keeps chained task follow-ups in PostgreSQL instead of
    accidentally routing them through unrelated knowledge documents.
    """
    direct = resolve_personal_question(question, previous_questions)
    if _intent(direct) is not None:
        return direct

    return resolve_personal_question(
        question,
        previous_questions,
        retrieval_question,
    )


# ============================================================
# STANDARD CHAT
# ============================================================

@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["AI Assistant"],
)
async def chat(
    request: ChatRequest,
    user: UserOut = Depends(current_user),
):
    """Non-streaming endpoint.

    Kept for compatibility with existing API clients.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty",
        )

    trace = start_observation(
        "firstday-chat", metadata={
            "endpoint": "chat", "question_chars": len(question),
        },
    )
    parent = trace.child_context
    try:
        previous_questions: list[str] = []

        if request.project_id is not None:
            await run_in_threadpool(check_project_access, request.project_id, user.id)

        if request.conversation_id is not None:
            previous_questions = await run_in_threadpool(
                recent_questions,
                request.conversation_id,
                user.id,
                request.project_id,
            )

        (
            retrieval_question,
            generation_question,
            previous_topic,
        ) = build_document_questions(
            question,
            previous_questions,
            project_context=request.project_id is not None,
        )

        personal_question = choose_personal_question(
            question,
            previous_questions,
            retrieval_question,
        )

        with observed("personal-query", parent=parent) as span:
            if request.project_id is None:
                personal = await run_in_threadpool(personal_answer, personal_question, user)
            else:
                personal = await run_in_threadpool(
                    project_personal_answer, personal_question, user, request.project_id,
                )
            span.update(output={"handled": personal is not None})

        if personal is not None:
            answer = personal
            sources = []

        else:
            if request.project_id is not None:
                result = await run_in_threadpool(
                    ask_project, generation_question, request.project_id, user.id,
                    retrieval_question, previous_topic, parent,
                )
            elif parent is None:
                result = await run_in_threadpool(
                    ask_firstday, generation_question, "EMPLOYEE",
                    retrieval_question, previous_topic,
                )
            else:
                result = await run_in_threadpool(
                    ask_firstday, generation_question, "EMPLOYEE",
                    retrieval_question, previous_topic, parent,
                )

            answer = result["answer"]
            sources = result["sources"]

        with observed("chat-history-save", parent=parent):
            conversation_id = await run_in_threadpool(
                save_exchange, user.id, request.conversation_id,
                question, answer, sources,
                request.project_id,
            )
        trace.update(output={
            "route": "personal" if personal is not None else "rag",
            "sources_count": len(sources),
            "response_chars": len(answer),
        })

        return ChatResponse(
            answer=answer,
            sources=[
                SourceResponse(**source)
                for source in sources
            ],
            conversation_id=conversation_id,
        )

    except HTTPException as exc:
        trace.mark_error(exc)
        raise

    except psycopg2.Error as exc:
        trace.mark_error(exc)
        logger.exception(
            "Chat database failure"
        )

        raise HTTPException(
            status_code=503,
            detail="Chat database unavailable",
        ) from exc

    except Exception as exc:
        trace.mark_error(exc)
        logger.exception(
            "Chat processing failed"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "AI unavailable. "
                "Check Ollama and PostgreSQL."
            ),
        ) from exc

    finally:
        trace.end()


# ============================================================
# NDJSON HELPERS
# ============================================================

def encode_event(event: dict) -> str:
    """Encode one streaming event as an NDJSON line."""

    return (
        json.dumps(
            event,
            ensure_ascii=False,
        )
        + "\n"
    )


# ============================================================
# REAL OLLAMA STREAMING
# ============================================================

@app.post(
    "/api/v1/chat/stream",
    tags=["AI Assistant"],
)
def stream_chat(
    request: ChatRequest,
    user: UserOut = Depends(current_user),
):
    """Stream Ollama-generated text directly to React.

    Workflow:
        1. Validate authenticated user.
        2. Retrieve previous questions.
        3. Resolve follow-up context.
        4. Check personal PostgreSQL data.
        5. Otherwise retrieve RAG documents.
        6. Stream live Ollama chunks.
        7. Save completed exchange to PostgreSQL.
        8. Send completion event with sources.

    A failed or incomplete generation is not saved.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty",
        )

    def events():
        trace = start_observation(
            "firstday-chat", metadata={
                "endpoint": "stream", "question_chars": len(question),
            },
        )
        parent = trace.child_context
        try:
            yield from traced_events(trace, parent)
        finally:
            trace.end()

    def traced_events(trace, parent):
        yield encode_event({
            "type": "status",
            "message": (
                "Searching documents and preparing an answer…"
            ),
        })

        full_answer_parts: list[str] = []
        sources: list[dict] = []

        try:
            # --------------------------------------------
            # 1. Load previous USER questions.
            # --------------------------------------------

            previous_questions: list[str] = []

            if request.project_id is not None:
                check_project_access(request.project_id, user.id)

            if request.conversation_id is not None:
                previous_questions = recent_questions(
                    request.conversation_id,
                    user.id,
                    request.project_id,
                )

            # --------------------------------------------
            # 2. Resolve follow-up context.
            # --------------------------------------------

            (
                retrieval_question,
                generation_question,
                previous_topic,
            ) = build_document_questions(
                question,
                previous_questions,
                project_context=request.project_id is not None,
            )

            # --------------------------------------------
            # 3. Handle personal-data questions.
            # --------------------------------------------

            personal_question = choose_personal_question(
                question,
                previous_questions,
                retrieval_question,
            )

            with observed("personal-query", parent=parent) as span:
                personal = (
                    personal_answer(personal_question, user) if request.project_id is None
                    else project_personal_answer(personal_question, user, request.project_id)
                )
                span.update(output={"handled": personal is not None})

            if personal is not None:
                full_answer_parts.append(personal)

                yield encode_event({
                    "type": "delta",
                    "text": personal,
                })

            # --------------------------------------------
            # 4. Stream corporate-document answers.
            # --------------------------------------------

            else:
                stream_kwargs = dict(
                    question=generation_question,
                    access_level="EMPLOYEE",
                    retrieval_question=retrieval_question,
                    previous_topic=previous_topic,
                )
                if parent is not None:
                    stream_kwargs["telemetry_parent"] = parent
                if request.project_id is not None:
                    stream_kwargs.pop("access_level")
                    stream_kwargs.update(project_id=request.project_id, employee_id=user.id)
                    response_stream = stream_project(**stream_kwargs)
                else:
                    response_stream = stream_firstday(**stream_kwargs)
                for event in response_stream:
                    event_type = event.get("type")

                    if event_type == "sources":
                        sources = event.get(
                            "sources",
                            [],
                        )

                    elif event_type == "delta":
                        text = event.get(
                            "text",
                            "",
                        )

                        if not text:
                            continue

                        full_answer_parts.append(text)

                        # Text is forwarded as soon as
                        # Ollama produces it.
                        yield encode_event({
                            "type": "delta",
                            "text": text,
                        })

            # --------------------------------------------
            # 5. Validate completed answer.
            # --------------------------------------------

            full_answer = "".join(
                full_answer_parts
            ).strip()

            if not full_answer:
                raise RuntimeError(
                    "AI returned an empty answer."
                )

            # --------------------------------------------
            # 6. Save the successful exchange.
            # --------------------------------------------

            with observed("chat-history-save", parent=parent):
                conversation_id = save_exchange(
                    employee_id=user.id,
                    conversation_id=request.conversation_id,
                    question=question,
                    answer=full_answer,
                    sources=sources,
                    project_id=request.project_id,
                )
            trace.update(output={
                "route": "personal" if personal is not None else "rag",
                "sources_count": len(sources),
                "response_chars": len(full_answer),
            })

            # --------------------------------------------
            # 7. Finish the stream.
            # --------------------------------------------

            yield encode_event({
                "type": "done",
                "conversation_id": str(
                    conversation_id
                ),
                "sources": sources,
            })

        except HTTPException as exc:
            trace.mark_error(exc)
            yield encode_event({
                "type": "error",
                "message": str(exc.detail),
            })

        except psycopg2.Error as exc:
            trace.mark_error(exc)
            logger.exception(
                "Streaming database failure"
            )

            yield encode_event({
                "type": "error",
                "message": (
                    "Database unavailable. "
                    "Check backend logs."
                ),
            })

        except Exception as exc:
            trace.mark_error(exc)
            logger.exception(
                "Ollama streaming failed"
            )

            yield encode_event({
                "type": "error",
                "message": (
                    "AI streaming failed. "
                    "Check Ollama and backend logs."
                ),
            })

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Content-Type-Options": "nosniff",
            "X-Accel-Buffering": "no",
        },
    )

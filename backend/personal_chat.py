"""Personal employee assistant for FirstDay AI.

Provides read-only answers using PostgreSQL.

Personal data:
- The first task from the authenticated employee's onboarding plan
- The next unfinished task
- Remaining onboarding tasks
- Personal onboarding progress
- Access request statuses

Questions about general learning tracks, job responsibilities
and corporate instructions must be handled by document RAG.

Employee identity is always taken from the authenticated
session, never from the user's question.
"""

from __future__ import annotations

import re
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from backend.auth_routes import (
    UserOut,
    DB_CONFIG,
)


# ============================================================
# SUPPORTED SYSTEMS
# ============================================================

SYSTEMS = (
    "github",
    "vpn",
    "aws",
    "jira",
    "confluence",
    "slack",
    "gitlab",
    "figma",
)


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize a user question before classification."""

    return re.sub(
        r"\s+",
        " ",
        text.casefold().replace("ё", "е"),
    ).strip()


def _any(
    patterns: tuple[str, ...],
    text: str,
) -> bool:
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


# ============================================================
# LEARNING TRACK DETECTION
# ============================================================

ROLE_PATTERNS = (
    r"\bjava\b",
    r"\bjunior\s+java\b",
    r"\bджава\b",
    r"\bqa\b",
    r"\bdevops\b",
    r"\bdeveloper\b",
    r"\bengineer\b",
    r"\bразработчик\w*\b",
    r"\bтестировщик\w*\b",
)

CURRICULUM_PATTERNS = (
    r"\bперв\w*\s+недел\w*\b",
    r"\bперв\w*\s+дн\w*\b",
    r"\bдни\s*1\s*[-–—]\s*5\b",
    r"\bfirst\s+week\b",
    r"\bweek\s*1\b",
    r"\bучебн\w*\s+трек\w*\b",
    r"\bпрограмм\w*\s+обучен\w*\b",
    r"\bплан\w*\s+обучен\w*\b",
    r"\bобязанност\w*\b",
    r"\bчему\s+(?:должен|нужно)\s+научиться\b",
)

PERSONAL_REFERENCES = (
    r"\b(?:мой|моя|мою|мое|моего|моей|моем|моим|мои|моих)\b",
    r"\bу\s+меня\b",
    r"\bмои\b",
    r"\bмоих\b",
    r"\bмоем\b",
    r"\bмоему\b",
    r"\bмне\s+остал\w*\b",
    r"\bя\s+выполн\w*\b",
    r"\bмною\s+выполн\w*\b",
    r"\bmy\b",
    r"\bmine\b",
    r"\bi\s+have\s+completed\b",
)


def is_curriculum_question(question: str) -> bool:
    """Detect general learning plans rather than personal progress.

    Example:
    'Какие задачи у Junior Java Developer на первую неделю?'
    must go to RAG, not to the user's saved tasks.
    """

    text = normalize_text(question)

    has_role = _any(ROLE_PATTERNS, text)
    has_curriculum = _any(CURRICULUM_PATTERNS, text)
    has_personal_reference = _any(
        PERSONAL_REFERENCES,
        text,
    )

    # Explicit requests about the user's own completed tasks
    # take priority over generic learning-track questions.
    if has_personal_reference:
        return False

    if has_role and has_curriculum:
        return True

    # A general question about a learning program is also
    # documentary even when the job role is omitted.
    if _any(
        (
            r"\bпрограмм\w*\s+обучен\w*\b",
            r"\bучебн\w*\s+трек\w*\b",
            r"\bплан\w*\s+обучен\w*\b",
        ),
        text,
    ):
        return True

    return False


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

REQUEST_PHRASES = (
    r"\b(?:мой|моя|моей|мою|мои|моих)\s+"
    r"(?:заявк\w*|запрос\w*)\b",

    r"\b(?:статус|состояние)\s+"
    r".{0,50}(?:заявк\w*|запрос\w*|доступ\w*)",

    r"\b(?:заявк\w*|запрос\w*)\s+"
    r".{0,40}(?:статус|одобр\w*|отклон\w*|рассмотр\w*)",

    r"\b(?:одобр\w*|отклони\w*|рассмотр\w*)\s+"
    r".{0,45}(?:заявк\w*|запрос\w*)",

    r"\b(?:my|mine)\s+"
    r"(?:access\s+)?requests?\b",

    r"\bstatus\s+of\s+"
    r"(?:my\s+)?(?:access\s+)?"
    r"(?:request|application)\b",
)

TASK_PHRASES = (
    r"\b(?:мои|моих)\s+"
    r"(?:невыполненн\w*\s+|оставш\w*\s+)?"
    r"(?:задач\w*|задани\w*|шаг\w*)\b",

    r"\b(?:какие|сколько)\s+"
    r"(?:у\s+меня|мне)\s+"
    r".{0,45}(?:задач\w*|задани\w*|шаг\w*)\b",

    r"\b(?:что|чего)\s+"
    r"(?:мне\s+)?(?:еще\s+)?"
    r"(?:остал\w*|выполнить|сделать)\b",

    r"\b(?:какие|сколько)\s+"
    r".{0,45}(?:задач\w*|задани\w*)\s+"
    r"(?:(?:у\s+меня)|(?:мне))\b",

    r"\b(?:задач\w*|задани\w*)\s+"
    r".{0,35}(?:мною\s+)?"
    r"(?:выполн\w*|остал\w*|сделан\w*)\b",

    r"\b(?:my|mine)\s+"
    r"(?:remaining\s+|pending\s+|unfinished\s+)?"
    r"(?:onboarding\s+)?tasks?\b",

    r"\bwhat\s+tasks?\s+"
    r"do\s+i\s+(?:have|need)\b",
)

# Explicit requests to see one specific task from the authenticated plan.
# A general role curriculum such as "first week of DevOps" stays in RAG.
FIRST_TASK_PHRASES = (
    r"\b(?:мой|моя|мою|мое|моей|моего|мои)\s+"
    r"(?:сам\w*\s+)?перв\w*\s+(?:задач\w*|задани\w*|шаг\w*)\b",
    r"\bперв\w*\s+(?:задач\w*|задани\w*|шаг\w*)\b"
    r".{0,100}\b(?:моего|моей|моем|моих|у\s+меня|my\s+onboarding)\b",
    r"\b(?:персональн\w*\s+)?план\w*\s+адаптац\w*\b"
    r".{0,90}\bперв\w*\s+(?:задач\w*|задани\w*)\b",
    r"\b(?:у\s+меня)\s+(?:есть\s+)?перв\w*\s+"
    r"(?:задач\w*|задани\w*)\b",
    r"\bmy\s+first\s+(?:onboarding\s+)?task\b",
    r"\bfirst\s+task\s+(?:in|from)\s+my\b",
)

NEXT_TASK_PHRASES = (
    r"\b(?:моя|мою|мой|мои|моей)\s+следующ\w*\s+"
    r"(?:задач\w*|задани\w*|шаг\w*)\b",
    r"\b(?:какая|какой)\s+у\s+меня\s+следующ\w*\s+"
    r"(?:задач\w*|задани\w*|шаг\w*)\b",
    r"\b(?:какая|какой)\s+следующ\w*\s+"
    r"(?:задач\w*|задани\w*|шаг\w*)\s+у\s+меня\b",
    r"\bmy\s+next\s+(?:onboarding\s+)?task\b",
    r"\bwhat\s+(?:should\s+i\s+do|is\s+my\s+next\s+task)\s+next\b",
)

# These short follow-ups are only personal when the same authenticated
# conversation previously discussed a personal task.
PERSONAL_TASK_FOLLOWUPS = (
    r"\b(?:я\s+)?(?:ее|её|его|эту\s+задач\w*|этот\s+шаг)"
    r".{0,35}(?:выполн\w*|сделал\w*|заверш\w*|закрыл\w*)\b",
    r"\b(?:она|оно|эта\s+задач\w*)\s+"
    r"(?:уже\s+)?(?:выполн\w*|сделан\w*|заверш\w*)\b",
    r"\b(?:какой|а\s+какой)\s+.{0,20}(?:ее|её|этой\s+задач\w*)"
    r".{0,20}статус\b",
    r"\b(?:какой|а\s+какой)\s+статус\s+(?:у\s+)?"
    r"(?:нее|неё|этой\s+задач\w*)\b",
    r"\bdid\s+i\s+(?:finish|complete)\s+(?:it|that\s+task)\b",
    r"\bis\s+(?:it|that\s+task)\s+(?:done|completed)\b",
)

NEXT_TASK_FOLLOWUPS = (
    r"^(?:а\s+)?(?:что|какая\s+задача|какой\s+шаг)\s+дальше\??$",
    r"^(?:а\s+)?что\s+мне\s+(?:делать|выполнять)\s+дальше\??$",
    r"^(?:and\s+)?what(?:'s|\s+is)?\s+next\??$",
)


PROGRESS_PHRASES = (
    r"\b(?:мой|мои)\s+"
    r"(?:прогресс|результат)\b",

    r"\b(?:какой|сколько)\s+"
    r".{0,30}(?:мой|у\s+меня)\s+"
    r".{0,25}(?:прогресс|адаптац\w*|выполнен\w*)",

    r"\bсколько\s+"
    r"(?:я\s+)?(?:уже\s+)?"
    r"(?:выполнил\w*|сделал\w*)\b",

    r"\b(?:мой|моя)\s+"
    r"(?:onboarding\s+)?progress\b",

    r"\bhow\s+far\s+"
    r"(?:am\s+i|have\s+i)\b",

    r"\bhow\s+much\s+"
    r"have\s+i\s+completed\b",
)


def _intent(question: str) -> str | None:
    """Classify only explicit personal-data questions.

    Returns:
        'first_task', 'first_task_status', 'next_task', 'next_task_status',
        'tasks', 'progress', 'requests', or None.

    None means the question should continue through RAG.
    """

    text = normalize_text(question)

    if not text:
        return None

    # Learning plans are documentation questions.
    if is_curriculum_question(text):
        return None

    # Request status always has priority over
    # other personal-data categories.
    if _any(REQUEST_PHRASES, text):
        return "requests"

    # Explicit request-status questions may name the system
    # without using the word 'заявка'.
    if (
        any(
            word in text
            for word in (
                "статус",
                "status",
                "одобри",
                "отклони",
            )
        )
        and any(
            word in text
            for word in (
                "доступ",
                "access",
            )
        )
        and any(
            re.search(
                rf"\b{re.escape(system)}\b",
                text,
            )
            for system in SYSTEMS
        )
    ):
        return "requests"

    # Personal progress.
    if _any(PROGRESS_PHRASES, text):
        return "progress"

    # Explicit questions about completion/status should be brief,
    # instead of repeating the entire task description.
    asking_task_status = _any(
        (
            r"\bстатус\b",
            r"\bвыполнен\w*\b",
            r"\bвыполнил\w*\b",
            r"\bзавершен\w*\b",
            r"\bготов\w*\b",
        ),
        text,
    )

    # A single saved task must take priority over task lists.
    if _any(FIRST_TASK_PHRASES, text):
        return "first_task_status" if asking_task_status else "first_task"

    if _any(NEXT_TASK_PHRASES, text):
        return "next_task_status" if asking_task_status else "next_task"

    # Personal task list.
    if _any(TASK_PHRASES, text):
        return "tasks"

    return None


def _recent_task_context(previous_questions: list[str]) -> str | None:
    """Recover the latest personal task referent from USER questions.

    Replay a bounded sequence to handle multiple chained follow-ups:
    first task -> "did I do it?" -> "what next?" -> "did I do it?"
    The final pronoun now refers to the next task, not the first.
    """
    task_intents = {
        "first_task", "first_task_status",
        "next_task", "next_task_status",
        "tasks", "progress",
    }
    context: str | None = None

    for previous in previous_questions[-8:]:
        text = normalize_text(previous)
        intent = _intent(text)

        if intent in task_intents:
            context = intent
        elif intent is not None:
            context = None
        elif _any(NEXT_TASK_FOLLOWUPS, text):
            if context in task_intents:
                context = "next_task"
            else:
                context = None
        elif _any(PERSONAL_TASK_FOLLOWUPS, text):
            if context in {"first_task", "first_task_status"}:
                context = "first_task_status"
            elif context in {"next_task", "next_task_status"}:
                context = "next_task_status"
            else:
                context = None
        else:
            context = None

    return context


def resolve_personal_question(
    question: str,
    previous_questions: list[str],
    retrieval_question: str | None = None,
) -> str:
    """Resolve personal intent from recent USER questions only.

    No private answer or other employee's data are used for context;
    personal_answer always selects by the authenticated user ID.
    """
    current = question.strip()

    # The current question takes precedence over any history.
    if _intent(current) is not None:
        return current

    normalized = normalize_text(current)
    is_status_followup = _any(PERSONAL_TASK_FOLLOWUPS, normalized)
    is_next_followup = _any(NEXT_TASK_FOLLOWUPS, normalized)

    if is_status_followup or is_next_followup:
        context = _recent_task_context(previous_questions)

        if is_status_followup:
            if context in {"first_task", "first_task_status"}:
                return "Какой статус моей первой задачи?"
            if context in {"next_task", "next_task_status"}:
                return "Какой статус моей следующей задачи?"

        if is_next_followup and context in {
            "first_task", "first_task_status",
            "next_task", "next_task_status",
            "tasks", "progress",
        }:
            return "Какая моя следующая задача?"

        # Do not guess the task if there is no current anchor.
        return current

    # Maintain compatible handling of context-resolved request status.
    if retrieval_question and _intent(retrieval_question) is not None:
        return retrieval_question

    return current


# ============================================================
# STATUS TRANSLATION
# ============================================================

def _status(value: str) -> str:
    """Return a user-friendly access-request status."""

    statuses = {
        "PENDING": "ожидает рассмотрения",
        "APPROVED": "одобрена в FirstDay AI",
        "REJECTED": "отклонена",
    }

    return statuses.get(
        value,
        "статус не определён",
    )


# ============================================================
# DATABASE: PERSONAL ONBOARDING PLAN
# ============================================================

def _plan(
    conn,
    employee_id: str,
):
    """Load only the authenticated employee's onboarding plan."""

    with conn.cursor(
        cursor_factory=RealDictCursor
    ) as cur:

        cur.execute(
            """
            SELECT
                p.id,
                p.title,
                p.status
            FROM onboarding_plans AS p
            WHERE p.employee_id = %s
            ORDER BY p.created_at DESC
            LIMIT 1
            """,
            (employee_id,),
        )

        plan = cur.fetchone()

        if plan is None:
            return None, []

        cur.execute(
            """
            SELECT
                t.title,
                t.status,
                t.day_number,
                COALESCE(
                    NULLIF(to_jsonb(t)->>'description', ''),
                    NULLIF(to_jsonb(t)->>'details', ''),
                    NULLIF(to_jsonb(t)->>'instructions', '')
                ) AS description
            FROM onboarding_tasks AS t
            WHERE t.plan_id = %s
            ORDER BY t.day_number, t.created_at, t.id
            """,
            (str(plan["id"]),),
        )

        tasks = cur.fetchall()

        return plan, tasks


# ============================================================
# DATABASE: PERSONAL ACCESS REQUESTS
# ============================================================

def _requests(
    conn,
    employee_id: str,
):
    """Load only requests belonging to the signed-in employee."""

    with conn.cursor(
        cursor_factory=RealDictCursor
    ) as cur:

        cur.execute(
            """
            SELECT
                system_name,
                status,
                reason,
                requested_at
            FROM access_requests
            WHERE employee_id = %s
            ORDER BY requested_at DESC, id DESC
            LIMIT 100
            """,
            (employee_id,),
        )

        return cur.fetchall()


# ============================================================
# FORMATTING ONE TASK FROM THE PERSONAL PLAN
# ============================================================

TASK_STATUSES = {
    "DONE": "выполнена",
    "COMPLETED": "выполнена",
    "IN_PROGRESS": "в работе",
    "PENDING": "ещё не начата",
    "TODO": "ещё не начата",
    "NOT_STARTED": "ещё не начата",
    "BLOCKED": "заблокирована",
    "CANCELLED": "отменена",
}


def _task_status(value: str | None) -> str:
    status_code = str(value or "").upper()
    return TASK_STATUSES.get(
        status_code,
        f"{status_code} (значение из базы данных)" if status_code
        else "не указан",
    )


def _format_one_task(task: dict, label: str) -> str:
    title = str(task.get("title") or "Без названия")
    day_number = task.get("day_number")
    description = str(task.get("description") or "").strip()
    status = _task_status(task.get("status"))

    lines = [
        f"**{label}: {title}**",
    ]

    if day_number is not None:
        lines.append(f"**День плана:** {day_number}")

    lines.append(f"**Статус:** {status}.")

    if description:
        lines.append(f"**Описание:** {description}")
    else:
        lines.append(
            "**Описание:** отдельное описание не указано в "
            "персональной записи задачи. Подробности можно посмотреть "
            "в разделе **My Onboarding**."
        )

    return "\n\n".join(lines)


def _answer_first_task(conn, employee_id: str) -> str:
    """Read the earliest saved task, even if it is already DONE."""
    plan, tasks = _plan(conn, employee_id)

    if plan is None:
        return (
            "У тебя пока нет персонального плана адаптации. "
            "Обратись к HR."
        )

    if not tasks:
        return "В твоём персональном плане пока нет задач."

    return _format_one_task(tasks[0], "Твоя первая задача")


def _answer_first_task_status(conn, employee_id: str) -> str:
    """Brief completion status of the earliest task in the user's plan."""
    plan, tasks = _plan(conn, employee_id)

    if plan is None:
        return (
            "У тебя пока нет персонального плана адаптации. "
            "Обратись к HR."
        )
    if not tasks:
        return "В твоём персональном плане пока нет задач."

    task = tasks[0]
    title = str(task.get("title") or "Без названия")
    status = _task_status(task.get("status"))
    return f"Да, задача **«{title}»** выполнена." if status == "выполнена" else (
        f"Задача **«{title}»** — **{status}**."
    )


def _answer_next_task_status(conn, employee_id: str) -> str:
    """Report the status of the earliest unfinished task, if present."""
    plan, tasks = _plan(conn, employee_id)

    if plan is None:
        return (
            "У тебя пока нет персонального плана адаптации. "
            "Обратись к HR."
        )
    if not tasks:
        return "В твоём персональном плане пока нет задач."

    for task in tasks:
        code = str(task.get("status") or "").upper()
        if code not in {"DONE", "COMPLETED", "CANCELLED"}:
            title = str(task.get("title") or "Без названия")
            return f"Задача **«{title}»** — **{_task_status(code)}**."

    return "Все задачи персонального плана завершены."


def _answer_next_task(conn, employee_id: str) -> str:
    """Read the earliest unfinished task from the employee's plan."""
    plan, tasks = _plan(conn, employee_id)

    if plan is None:
        return (
            "У тебя пока нет персонального плана адаптации. "
            "Обратись к HR."
        )

    pending = [
        task for task in tasks
        if str(task.get("status") or "").upper()
        not in {"DONE", "COMPLETED", "CANCELLED"}
    ]

    if not pending:
        return (
            "В твоём персональном плане нет невыполненных задач. "
            "Посмотри завершённый план в **My Onboarding**."
        )

    return _format_one_task(
        pending[0],
        "Твоя ближайшая невыполненная задача",
    )


# ============================================================
# FORMATTING PERSONAL TASKS
# ============================================================

def _answer_tasks(
    conn,
    employee_id: str,
) -> str:
    """Answer questions about actual saved tasks."""

    plan, tasks = _plan(
        conn,
        employee_id,
    )

    if plan is None:
        return (
            "У тебя пока нет персонального плана адаптации. "
            "Обратись к HR."
        )

    completed = [
        task
        for task in tasks
        if task["status"] == "DONE"
    ]

    remaining = [
        task
        for task in tasks
        if task["status"] != "DONE"
    ]

    total = len(tasks)

    percent = (
        round(100 * len(completed) / total)
        if total
        else 0
    )

    if not remaining:
        return (
            f"Все задачи выполнены: "
            f"**{len(completed)} из {total} ({percent}%)**.\n\n"
            "Можешь посмотреть завершённый план "
            "в разделе **My Onboarding**."
        )

    lines = [
        f"**Твои невыполненные задачи ({len(remaining)}):**",
    ]

    for task in remaining[:15]:
        status = (
            "в работе"
            if task["status"] == "IN_PROGRESS"
            else "ещё не начата"
        )

        lines.append(
            f"- День {task['day_number']}: "
            f"{task['title']} — {status}"
        )

    if len(remaining) > 15:
        lines.append(
            f"- И ещё {len(remaining) - 15} задач "
            "в разделе **My Onboarding**."
        )

    lines.append(
        f"\nПрогресс: **{percent}%** "
        f"({len(completed)}/{total})."
    )

    return "\n".join(lines)


# ============================================================
# FORMATTING PERSONAL PROGRESS
# ============================================================

def _answer_progress(
    conn,
    employee_id: str,
) -> str:
    """Answer questions about real onboarding progress."""

    plan, tasks = _plan(
        conn,
        employee_id,
    )

    if plan is None:
        return (
            "У тебя пока нет персонального плана адаптации. "
            "Обратись к HR."
        )

    total = len(tasks)

    completed = sum(
        1
        for task in tasks
        if task["status"] == "DONE"
    )

    remaining = total - completed

    percent = (
        round(100 * completed / total)
        if total
        else 0
    )

    return (
        f"**Твой прогресс адаптации: {percent}%**.\n\n"
        f"Выполнено **{completed} из {total}** задач. "
        f"Осталось: **{remaining}**.\n\n"
        "Открой **My Onboarding**, чтобы посмотреть "
        "подробный план."
    )


# ============================================================
# FORMATTING ACCESS REQUESTS
# ============================================================

def _answer_requests(
    conn,
    employee_id: str,
    question: str,
) -> str:
    """Answer access-request questions using real saved data."""

    requests = _requests(
        conn,
        employee_id,
    )

    text = normalize_text(question)

    aliases = {
        "гитхаб": "github",
        "впн": "vpn",
        "джира": "jira",
        "гитлаб": "gitlab",
    }

    specified: list[str] = []

    for alias, system in aliases.items():
        if re.search(
            rf"\b{re.escape(alias)}\b",
            text,
        ):
            specified.append(system)

    for system in SYSTEMS:
        if (
            re.search(
                rf"\b{re.escape(system)}\b",
                text,
            )
            and system not in specified
        ):
            specified.append(system)

    if specified:
        requests = [
            request
            for request in requests
            if any(
                system in (
                    request["system_name"] or ""
                ).casefold()
                for system in specified
            )
        ]

    if not requests:
        if specified:
            return (
                "У тебя нет сохранённой заявки на "
                f"**{', '.join(specified)}**.\n\n"
                "Проверь раздел **Access Center**."
            )

        return (
            "У тебя пока нет заявок на доступ. "
            "Создать заявку можно в **Access Center**."
        )

    lines = [
        f"**Твои заявки на доступ ({len(requests)}):**"
    ]

    for request in requests[:12]:
        lines.append(
            f"- **{request['system_name']}** — "
            f"{_status(request['status'])}."
        )

    if len(requests) > 12:
        lines.append(
            f"- Ещё {len(requests) - 12} заявок "
            "в разделе **Access Center**."
        )

    lines.append(
        "\nОдобрение заявки в FirstDay AI "
        "не означает автоматической выдачи реальных прав."
    )

    return "\n".join(lines)


# ============================================================
# MAIN PERSONAL ASSISTANT
# ============================================================

def personal_answer(
    question: str,
    user: UserOut,
) -> Optional[str]:
    """Return a private-data answer or None for document RAG.

    The employee ID always comes from the verified session.

    The question cannot specify another employee ID.
    """

    intent = _intent(question)

    if intent is None:
        return None

    employee_id = str(user.id)

    with psycopg2.connect(**DB_CONFIG) as conn:

        if intent == "first_task":
            return _answer_first_task(conn, employee_id)

        if intent == "first_task_status":
            return _answer_first_task_status(conn, employee_id)

        if intent == "next_task":
            return _answer_next_task(conn, employee_id)

        if intent == "next_task_status":
            return _answer_next_task_status(conn, employee_id)

        if intent == "tasks":
            return _answer_tasks(
                conn,
                employee_id,
            )

        if intent == "progress":
            return _answer_progress(
                conn,
                employee_id,
            )

        if intent == "requests":
            return _answer_requests(
                conn,
                employee_id,
                question,
            )

    return None

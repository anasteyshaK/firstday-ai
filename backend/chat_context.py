
"""Conversation context handling for FirstDay AI.

Only previous USER questions from the authenticated
conversation are used.

Private employee records and assistant answers are
never inserted into the RAG/Ollama prompt.
"""

from __future__ import annotations

import re

from backend.personal_chat import SYSTEMS, _intent


def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.casefold().replace("ё", "е"),
    ).strip()


TOPIC_PATTERNS = {
    "stack": (r"\bстек\w*\b", r"\bstack\b", r"какие\s+технологии", r"what\s+technolog"),
    "vpn": (
        r"\bvpn\b",
        r"\bвпн\b",
    ),
    "github": (
        r"\bgithub\b",
        r"\bгитхаб\b",
    ),
    "gitlab": (
        r"\bgitlab\b",
        r"\bгитлаб\b",
    ),
    "aws": (r"\baws\b",),
    "jira": (
        r"\bjira\b",
        r"\bджира\b",
    ),
    "confluence": (r"\bconfluence\b",),
    "slack": (r"\bslack\b",),
    "figma": (r"\bfigma\b",),
    "mfa": (
        r"\bmfa\b",
        r"\b2fa\b",
    ),
    "sso": (r"\bsso\b",),
    "java": (
        r"\bjava\b",
        r"\bджава\b",
    ),
    "qa": (
        r"\bqa\b",
        r"\bтестировщ\w*\b",
    ),
    "devops": (r"\bdevops\b",),
    "onboarding": (
        r"\bonboarding\b",
        r"\bонбординг\b",
        r"\bадаптац\w*\b",
    ),
    "first_day": (
        r"\bперв\w*\s+рабоч\w*\s+д[её]н\w*\b",
        r"\bперв\w*\s+д[её]н\w*\b",
        r"\bday[\s-]*one\b",
        r"\bfirst[\s-]*day\b",
        r"\bday[\s-]*1\b",
    ),
}


def extract_topics(text: str) -> set[str]:
    normalized = normalize_text(text)
    topics: set[str] = set()

    for name, patterns in TOPIC_PATTERNS.items():
        if any(
            re.search(pattern, normalized)
            for pattern in patterns
        ):
            topics.add(name)

    return topics


FOLLOWUP_PATTERNS = (
    r"^(?:а\s+)?если\b",
    r"^что\s+если\b",
    r"^а\s+что\s+если\b",
    r"^а\s+как\b",
    r"^а\s+где\b",
    r"^а\s+кто\b",
    r"^а\s+почему\b",
    r"^а\s+когда\b",
    r"^а\s+можно\s+ли\b",
    r"^а\s+нужно\s+ли\b",
    r"^а\s+что\s+насчет\b",
    r"^а\s+по\b",
    r"^а\s+про\b",
    r"^а\s+это\b",
    r"^а\s+там\b",
    r"^и\s+если\b",
    r"^а\s+дальше\b",
    r"^что\s+дальше\b",
)


def is_followup(question: str) -> bool:
    normalized = normalize_text(question)

    if not normalized or len(normalized) > 130:
        return False

    if any(
        re.search(pattern, normalized)
        for pattern in FOLLOWUP_PATTERNS
    ):
        return True

    return normalized in {
        "почему?",
        "как именно?",
        "где именно?",
        "а потом?",
        "а дальше?",
        "что дальше?",
        "а для меня?",
        "а это обязательно?",
    }


REMOTE_WORK = re.compile(
    r"\b("
    r"удаленн\w*|"
    r"из\s+дома|"
    r"дома|"
    r"remote|"
    r"home\s+office|"
    r"work\s+from\s+home"
    r")\b",
    re.IGNORECASE,
)


def is_remote_work_question(question: str) -> bool:
    return bool(
        REMOTE_WORK.search(
            normalize_text(question)
        )
    )


def extract_system(text: str) -> str | None:
    normalized = normalize_text(text)

    aliases = {
        "гитхаб": "github",
        "впн": "vpn",
        "джира": "jira",
        "гитлаб": "gitlab",
    }

    for alias, system in aliases.items():
        if re.search(
            rf"\b{re.escape(alias)}\b",
            normalized,
        ):
            return system

    for system in SYSTEMS:
        if re.search(
            rf"\b{re.escape(system)}\b",
            normalized,
        ):
            return system

    return None


def is_request_followup(question: str) -> bool:
    normalized = normalize_text(question)

    return bool(
        re.fullmatch(
            r"(?:а\s+)?"
            r"(?:что\s+)?"
            r"(?:по|про|насчет)\s+"
            r"(?:моей?\s+)?"
            r"(?:заявк\w*\s+)?"
            r"(?:на\s+|к\s+)?"
            r"[\w.+-]+\??",
            normalized,
        )
    )


def personal_followup(
    question: str,
    previous_intent: str | None,
) -> str | None:
    normalized = normalize_text(question)

    if previous_intent == "requests":
        system = extract_system(question)

        if system and is_request_followup(question):
            return (
                "Какой статус моей заявки "
                f"на {system}?"
            )

    if previous_intent in ("tasks", "progress"):
        if re.fullmatch(
            r"(?:а\s+)?"
            r"(?:сколько\s+)?"
            r"(?:еще\s+)?"
            r"(?:осталось|выполнено|сделано)"
            r"\??",
            normalized,
        ):
            if "осталось" in normalized:
                return "Какие у меня остались задачи?"

            return "Какой мой прогресс адаптации?"

    return None


def contextual_question(
    question: str,
    previous: list[str],
) -> str:
    """Expand genuine follow-ups for document retrieval.

    The original question remains unchanged for generation.
    """

    text = question.strip()

    if not text or not previous or len(text) > 130:
        return text

    previous_question = previous[-1].strip()

    if not previous_question:
        return text

    previous_intent = _intent(previous_question)
    current_intent = _intent(text)

    private = personal_followup(
        text,
        previous_intent,
    )

    if private is not None:
        return private

    # Do not use private questions as RAG context.
    if (
        current_intent is not None
        or previous_intent is not None
    ):
        return text

    # A standalone new question remains standalone.
    if not is_followup(text):
        return text

    previous_topics = extract_topics(previous_question)
    current_topics = extract_topics(text)

    # A new explicitly named topic must not inherit
    # the previous unrelated topic.
    if current_topics:
        return text

    # ------------------------------------------------
    # REMOTE WORK FOLLOW-UPS
    # ------------------------------------------------

    if is_remote_work_question(text):

        # Case 1: VPN -> remote work.
        if "vpn" in previous_topics:
            return (
                "Корпоративная VPN NovaTech для работы "
                "из дома. Какие требования к подключению "
                "VPN действуют при удаленной работе? "
                "Используй инструкцию подключения "
                "к корпоративной VPN."
            )

        # Case 2: first working day -> remote work.
        if "first_day" in previous_topics:
            return (
                "Первый рабочий день нового сотрудника "
                "NovaTech при удаленной работе из дома. "
                "Какие действия нужно выполнить новичку "
                "в первый день: получить оборудование, "
                "проверить корпоративную учетную запись, "
                "настроить MFA, связаться с buddy, "
                "открыть onboarding-план? "
                "Есть ли отдельные требования для "
                "удаленного первого дня?"
            )

        # Case 3: GitHub -> remote work.
        if "github" in previous_topics:
            return (
                "Доступ к корпоративному GitHub NovaTech "
                "для сотрудника при удаленной работе "
                "из дома. Требования к GitHub Access, "
                "SSO и работе с репозиториями."
            )

        # Case 4: general onboarding -> remote work.
        if "onboarding" in previous_topics:
            return (
                "Адаптация нового сотрудника NovaTech "
                "при удаленной работе из дома. "
                "Правила onboarding, оборудование, "
                "доступы и коммуникация с buddy."
            )

    # ------------------------------------------------
    # GENERAL DOCUMENT FOLLOW-UPS
    # ------------------------------------------------

    if previous_topics:
        return (
            f"Предыдущая тема: {previous_question[:250]}. "
            f"Текущее уточнение: {text}. "
            "Найди документы именно по предыдущей теме "
            "и текущему уточнению."
        )

    # No reliable context was identified.
    # Avoid retrieving unrelated documents.
    return text

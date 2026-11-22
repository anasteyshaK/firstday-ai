
"""Regression tests for FirstDay AI conversation routing.

Run from project root:

    python -m pytest tests/test_chat_routing.py -v

These tests do not need Ollama or a running PostgreSQL server.
"""

from __future__ import annotations

import pytest

from backend.chat_context import (
    contextual_question,
    extract_topics,
    is_followup,
    is_remote_work_question,
)

from backend.personal_chat import (
    _intent,
    is_curriculum_question,
)


# ============================================================
# 1. PERSONAL INFORMATION VS CORPORATE DOCUMENTS
# ============================================================

@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Какие у меня остались задачи?",
            "tasks",
        ),
        (
            "Покажи мои невыполненные задачи",
            "tasks",
        ),
        (
            "Какой мой прогресс адаптации?",
            "progress",
        ),
        (
            "Какой статус моей заявки на Slack?",
            "requests",
        ),
        (
            "Какой статус моей заявки на Jira?",
            "requests",
        ),
        (
            "Как получить доступ к Confluence?",
            None,
        ),
        (
            "Как настроить MFA?",
            None,
        ),
        (
            "Что нужно сделать в первый рабочий день?",
            None,
        ),
        (
            "Какие задачи у Junior Java Developer на первую неделю?",
            None,
        ),
        (
            "Что изучает QA Engineer в первую неделю?",
            None,
        ),
        (
            "Какие обязанности у DevOps Engineer?",
            None,
        ),
        (
            "Какие правила классификации данных?",
            None,
        ),
        (
            "Что делать, если потерял рабочий ноутбук?",
            None,
        ),
        (
            "Что такое SSO?",
            None,
        ),
    ],
)
def test_personal_intent_routing(question, expected):
    assert _intent(question) == expected


# ============================================================
# 2. LEARNING TRACKS MUST USE RAG
# ============================================================

@pytest.mark.parametrize(
    "question",
    [
        "Какие задачи у Junior Java Developer на первую неделю?",
        "Что изучает QA Engineer в первую неделю?",
        "Какие задачи DevOps Engineer в первую неделю?",
        "Какая программа обучения у Java Developer?",
        "Что входит в учебный трек QA?",
    ],
)
def test_learning_questions_use_documentation(question):
    assert is_curriculum_question(question)
    assert _intent(question) is None


# ============================================================
# 3. TOPIC IDENTIFICATION
# ============================================================

@pytest.mark.parametrize(
    ("question", "expected_topic"),
    [
        ("Как настроить MFA?", "mfa"),
        ("Как работает SSO?", "sso"),
        ("Какие задачи у Java Developer?", "java"),
        ("Что изучает QA Engineer?", "qa"),
        ("Чем занимается DevOps?", "devops"),
        ("Как получить доступ к Slack?", "slack"),
        ("Как пользоваться Jira?", "jira"),
        (
            "Что нужно сделать в первый рабочий день?",
            "first_day",
        ),
        (
            "Что входит в onboarding?",
            "onboarding",
        ),
    ],
)
def test_topic_detection(question, expected_topic):
    assert expected_topic in extract_topics(question)


# ============================================================
# 4. INDEPENDENT QUESTIONS MUST NOT INHERIT OLD TOPICS
# ============================================================

@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (
            "Что нужно сделать в первый рабочий день?",
            "Как настроить MFA?",
        ),
        (
            "Какие задачи у Java Developer?",
            "Как получить доступ к Slack?",
        ),
        (
            "Как работает SSO?",
            "Что изучает QA Engineer?",
        ),
        (
            "Какие обязанности у DevOps?",
            "Как пользоваться Jira?",
        ),
        (
            "Что такое onboarding?",
            "Какие правила классификации данных?",
        ),
        (
            "Как настроить MFA?",
            "Что нужно сделать в первый рабочий день?",
        ),
        (
            "Какие задачи у Java Developer?",
            "А как устроена работа QA Engineer?",
        ),
    ],
)
def test_independent_question_does_not_inherit_context(
    previous,
    current,
):
    result = contextual_question(
        current,
        [previous],
    )
    assert result == current


# ============================================================
# 5. FIRST-DAY FOLLOW-UPS
# ============================================================

@pytest.mark.parametrize(
    "previous",
    [
        "Что нужно сделать в первый рабочий день?",
        "Какие действия нужны в первый рабочий день NovaTech?",
        "Какие задачи у новичка в первый день?",
    ],
)
def test_remote_first_day_stays_on_first_day_topic(previous):
    question = "А если я работаю удалённо?"

    result = contextual_question(
        question,
        [previous],
    )

    assert result != question
    assert "первый рабочий день" in result.lower()
    assert "buddy" in result.lower()
    assert "mfa" in result.lower()

    assert "beacon" not in result.lower()
    assert "on-call" not in result.lower()


# ============================================================
# 6. ONBOARDING FOLLOW-UPS
# ============================================================

def test_remote_onboarding_stays_on_onboarding_topic():
    result = contextual_question(
        "А если я работаю удалённо?",
        ["Как проходит onboarding в NovaTech?"],
    )

    assert "адаптация" in result.lower()
    assert "buddy" in result.lower()


# ============================================================
# 7. PERSONAL DATA MUST NOT LEAK INTO DOCUMENT RAG
# ============================================================

def test_previous_personal_progress_not_added_to_rag():
    current = "Как настроить MFA?"

    result = contextual_question(
        current,
        ["Какой мой прогресс адаптации?"],
    )

    assert result == current



def test_personal_tasks_followup_stays_personal():
    result = contextual_question(
        "А сколько ещё осталось?",
        ["Какие у меня остались задачи?"],
    )

    assert result == "Какие у меня остались задачи?"
    assert _intent(result) == "tasks"



def test_explicit_personal_question_keeps_its_intent():
    current = "Какие у меня остались задачи?"

    result = contextual_question(
        current,
        ["Что изучает QA Engineer в первую неделю?"],
    )

    assert result == current
    assert _intent(result) == "tasks"


# ============================================================
# 8. FOLLOW-UP RECOGNITION
# ============================================================

@pytest.mark.parametrize(
    "question",
    [
        "А если я работаю удалённо?",
        "А как это сделать?",
        "А где именно?",
        "Что дальше?",
        "А это обязательно?",
    ],
)
def test_short_followup_recognized(question):
    assert is_followup(question)


@pytest.mark.parametrize(
    "question",
    [
        "Что нужно сделать в первый рабочий день?",
        "Какие обязанности у DevOps Engineer?",
        "Как настроить MFA?",
        "Как получить доступ к Slack?",
    ],
)
def test_independent_questions_not_treated_as_followups(
    question,
):
    assert not is_followup(question)


# ============================================================
# 9. REMOTE WORK DETECTION
# ============================================================

@pytest.mark.parametrize(
    "question",
    [
        "А если я работаю удалённо?",
        "Я работаю из дома",
        "Can I work remote?",
        "What about home office?",
        "I work from home",
    ],
)
def test_remote_work_recognized(question):
    assert is_remote_work_question(question)


# ============================================================
# 10. EMPTY OR MISSING HISTORY
# ============================================================

def test_no_previous_questions():
    question = "А если я работаю удалённо?"

    assert contextual_question(question, []) == question


def test_empty_question():
    assert contextual_question(
        "",
        ["Что нужно сделать в первый рабочий день?"],
    ) == ""


# ============================================================
# 11. MULTIPLE QUESTIONS IN ONE CONVERSATION
# ============================================================

def test_multiple_independent_topics():
    history = [
        "Что нужно сделать в первый рабочий день?",
        "Как настроить MFA?",
        "Какие задачи у DevOps Engineer?",
    ]

    current = "Как получить доступ к Slack?"

    result = contextual_question(
        current,
        history,
    )

    assert result == current


# ============================================================
# 12. PREVIOUS QUESTION IS NOT MUTATED
# ============================================================

def test_history_is_not_modified():
    history = [
        "Что нужно сделать в первый рабочий день?"
    ]

    original_history = history.copy()

    contextual_question(
        "А если я работаю удалённо?",
        history,
    )

    assert history == original_history

"""Regression tests for FirstDay AI personal task follow-up routing.

Copy to tests/test_personal_followups.py and run:
    python -m pytest tests/test_personal_followups.py -q

No network or live PostgreSQL access is required.
"""

import pytest

from backend import personal_chat as personal


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Привет, расскажи про мою первую задачу.", "first_task"),
        ("Какой статус моей первой задачи?", "first_task_status"),
        ("Моя первая задача выполнена?", "first_task_status"),
        ("Какая моя следующая задача?", "next_task"),
        ("Какой статус моей следующей задачи?", "next_task_status"),
        ("Какой мой прогресс адаптации?", "progress"),
        ("Какие задачи у DevOps Engineer на первую неделю?", None),
        ("Какие задачи у Java Developer на первую неделю?", None),
        ("Как настроить корпоративный VPN?", None),
        ("Как получить доступ к Slack?", None),
    ],
)
def test_intent_routing(question, expected):
    assert personal._intent(question) == expected


def test_short_followup_reports_first_task_status():
    history = ["Привет, расскажи про мою первую задачу."]
    resolved = personal.resolve_personal_question(
        "А я её уже выполнила?", history
    )
    assert personal._intent(resolved) == "first_task_status"


def test_next_step_after_completion_followup():
    history = [
        "Привет, расскажи про мою первую задачу.",
        "А я её уже выполнила?",
    ]
    resolved = personal.resolve_personal_question(
        "Что мне делать дальше?", history
    )
    assert personal._intent(resolved) == "next_task"


def test_pronoun_after_next_step_refers_to_next_task():
    history = [
        "Привет, расскажи про мою первую задачу.",
        "А я её уже выполнила?",
        "Что мне делать дальше?",
    ]
    resolved = personal.resolve_personal_question(
        "А я её уже выполнила?", history
    )
    assert personal._intent(resolved) == "next_task_status"


def test_next_step_after_progress_question():
    resolved = personal.resolve_personal_question(
        "Что мне делать дальше?",
        ["Какой мой прогресс адаптации?"],
    )
    assert personal._intent(resolved) == "next_task"


def test_unrelated_document_question_breaks_personal_context():
    history = [
        "Привет, расскажи про мою первую задачу.",
        "Какие задачи у DevOps Engineer на первую неделю?",
    ]
    resolved = personal.resolve_personal_question(
        "Что мне делать дальше?", history
    )
    assert resolved == "Что мне делать дальше?"
    assert personal._intent(resolved) is None


def test_without_context_no_task_is_assumed():
    resolved = personal.resolve_personal_question(
        "А я её уже выполнила?", []
    )
    assert personal._intent(resolved) is None


@pytest.fixture
def tasks_from_database(monkeypatch):
    tasks = [
        {
            "title": "Настроить корпоративную учётную запись и MFA",
            "status": "DONE",
            "day_number": 1,
            "description": "Познакомиться с buddy",
        },
        {
            "title": "Провести учебную проверку знаний",
            "status": "PENDING",
            "day_number": 6,
            "description": "Пройти задания и зафиксировать результат",
        },
    ]
    monkeypatch.setattr(
        personal,
        "_plan",
        lambda conn, employee_id: ({"id": "sample-plan"}, tasks),
    )
    return tasks


def test_first_task_status_is_brief(tasks_from_database):
    answer = personal._answer_first_task_status(object(), "employee-1")
    assert "выполнена" in answer
    assert "Настроить корпоративную" in answer
    assert "Описание" not in answer


def test_next_task_from_database(tasks_from_database):
    answer = personal._answer_next_task(object(), "employee-1")
    assert "Провести учебную проверку знаний" in answer
    assert "День плана:** 6" in answer
    assert "ещё не начата" in answer
    assert "Познакомиться с buddy" not in answer


def test_next_task_status_from_database(tasks_from_database):
    answer = personal._answer_next_task_status(object(), "employee-1")
    assert "Провести учебную проверку знаний" in answer
    assert "ещё не начата" in answer
    assert "Описание" not in answer


def test_no_remaining_tasks(monkeypatch):
    monkeypatch.setattr(
        personal,
        "_plan",
        lambda conn, employee_id: (
            {"id": "sample-plan"},
            [{"title": "Задача", "status": "DONE", "day_number": 1}],
        ),
    )
    assert "нет невыполненных" in personal._answer_next_task(
        object(), "employee-1"
    )

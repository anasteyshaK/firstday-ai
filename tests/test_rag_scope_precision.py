"""Regression tests: temporal scope and curriculum vs procedure scope."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from search import (
    passes_topic_filter,
    is_first_day_overview_document,
    is_role_first_week_overview_question,
)


def doc(path: str, title: str) -> dict:
    return {"file_path": path, "title": title}


def test_general_qa_learning_selects_plans_not_specific_howto():
    q = "Что изучает QA Engineer в первую неделю?"
    assert is_role_first_week_overview_question(q)
    assert passes_topic_filter(q, doc("05_qa/qa-onboarding.md", "Первая неделя QA Engineer"))
    assert passes_topic_filter(q, doc("08_learning/qa-track.md", "Учебный трек QA Engineer"))
    assert not passes_topic_filter(q, doc("05_qa/bug-reporting.md", "Как оформить хороший баг-репорт"))


def test_detailed_qa_procedure_still_retrievable():
    q = "Как оформить баг-репорт для QA?"
    assert not is_role_first_week_overview_question(q)
    assert passes_topic_filter(q, doc("05_qa/bug-reporting.md", "Как оформить хороший баг-репорт"))


def test_day_one_overview_excludes_before_first_day():
    q = "Что делать в первый рабочий день в NovaTech?"
    assert passes_topic_filter(q, doc("01_hr/day-one.md", "Первый день: чек-лист новичка"))
    assert passes_topic_filter(q, doc("01_hr/week-one.md", "План первой недели"))
    assert not is_first_day_overview_document(doc("01_hr/before-first-day.md", "Подготовка до первого рабочего дня"))
    assert not passes_topic_filter(q, doc("01_hr/before-first-day.md", "Подготовка до первого рабочего дня"))


def test_advance_preparation_remains_available():
    q = "Как подготовиться до первого рабочего дня?"
    assert passes_topic_filter(q, doc("01_hr/before-first-day.md", "Подготовка до первого рабочего дня"))


def test_other_role_week_one_still_allows_its_own_learning_plan():
    q = "Какие задачи у DevOps Engineer на первую неделю?"
    assert is_role_first_week_overview_question(q)
    assert passes_topic_filter(q, doc("08_learning/devops-track.md", "Учебный трек Junior DevOps"))
    assert passes_topic_filter(q, doc("06_devops/devops-onboarding.md", "Адаптация DevOps/SRE инженера"))
    assert not passes_topic_filter(q, doc("06_devops/oncall.md", "On-call и порядок эскалации"))

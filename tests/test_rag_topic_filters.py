"""Offline tests for FirstDay AI RAG topic and intent filtering."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from search import (
    topic_match,
    passes_topic_filter,
    is_access_setup_question,
    is_first_day_overview_question,
)


def doc(path, title):
    return {"file_path": path, "title": title}


def test_topic_match_recognizes_underscore_paths():
    assert topic_match("06_devops/oncall.md", "devops")
    assert topic_match("05_qa/qa-onboarding.md", "qa")
    assert not topic_match("05_qa/qa-onboarding.md", "devops")


def test_github_access_excludes_incident_runbooks():
    question = "Как получить доступ к корпоративному GitHub?"
    assert is_access_setup_question(question)
    assert passes_topic_filter(question, doc("02_it_support/github-access.md", "Доступ к GitHub"))
    assert not passes_topic_filter(question, doc("11_incidents/github-actions-python.md", "Python в GitHub Actions"))


def test_github_incident_question_keeps_incident_documents():
    question = "Почему ошибка Python в GitHub Actions?"
    assert not is_access_setup_question(question)
    assert passes_topic_filter(question, doc("11_incidents/github-actions-python.md", "Python в GitHub Actions"))


def test_qa_learning_excludes_devops_runbook():
    question = "Что изучает QA Engineer в первую неделю?"
    assert passes_topic_filter(question, doc("05_qa/qa-onboarding.md", "Адаптация QA инженера"))
    assert passes_topic_filter(question, doc("08_learning/qa-track.md", "Учебный трек QA"))
    assert not passes_topic_filter(question, doc("06_devops/oncall.md", "On-call и порядок эскалации"))


def test_devops_learning_excludes_qa_guides():
    question = "Какие задачи у DevOps Engineer на первую неделю?"
    assert passes_topic_filter(question, doc("08_learning/devops-track.md", "Учебный трек DevOps"))
    assert not passes_topic_filter(question, doc("05_qa/qa-onboarding.md", "Адаптация QA"))


def test_general_first_day_keeps_only_first_day_sources():
    question = "Что делать в первый рабочий день в NovaTech?"
    assert is_first_day_overview_question(question)
    assert passes_topic_filter(question, doc("01_hr/day-one.md", "Первый день: чек-лист новичка"))
    assert passes_topic_filter(question, doc("01_hr/week-one.md", "План первой недели"))
    assert not passes_topic_filter(question, doc("12_process/first-contribution.md", "Первый PR"))
    assert not passes_topic_filter(question, doc("04_engineering/spring-boot-guide.md", "Первый запуск Spring Boot"))


def test_first_day_tool_question_not_overfiltered():
    question = "Как настроить MFA в первый рабочий день?"
    assert not is_first_day_overview_question(question)
    assert passes_topic_filter(question, doc("02_it_support/account-mfa.md", "Корпоративная учётная запись и MFA"))


def test_first_day_problem_not_overfiltered():
    question = "Что делать, если ноутбук не работает в первый рабочий день?"
    assert not is_first_day_overview_question(question)
    assert passes_topic_filter(question, doc("02_it_support/laptop-setup.md", "Ноутбук и настройка устройства"))


def test_setup_vs_incident_are_distinct():
    assert is_access_setup_question("Как получить доступ к GitHub?")
    assert not is_access_setup_question("Как получить доступ к GitHub, если он заблокирован?")

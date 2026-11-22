
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DIRECTORY = PROJECT_ROOT / "rag"

sys.path.insert(0, str(RAG_DIRECTORY))

import answer
import search


# ------------------------------------------------------------
# Test data
# ------------------------------------------------------------

def make_result(
    title,
    file_path,
    content,
    similarity=0.85,
    access_level="EMPLOYEE",
):
    return {
        "title": title,
        "file_path": file_path,
        "category": file_path.split("/")[0],
        "access_level": access_level,
        "content": content,
        "chunk_index": 0,
        "similarity": similarity,
        "rerank_score": similarity,
    }


@pytest.fixture
def java_document():
    return make_result(
        title="Учебный трек Junior Java Developer",
        file_path="08_learning/java-track.md",
        content=(
            "# Учебный трек Junior Java Developer\n\n"
            "Дни 1–5: установка JDK 21, запуск сервиса "
            "и знакомство с архитектурой.\n"
            "Дни 6–10: изучение REST API.\n"
            "Дни 11–20: разработка endpoint с тестами."
        ),
    )


@pytest.fixture
def vpn_document():
    return make_result(
        title="Подключение к корпоративной VPN",
        file_path="02_it_support/vpn-guide.md",
        content=(
            "# Подключение к корпоративной VPN\n\n"
            "1. Создайте заявку VPN Access.\n"
            "2. Установите корпоративный VPN-клиент."
        ),
    )


@pytest.fixture
def github_document():
    return make_result(
        title="Доступ к GitHub и репозиториям",
        file_path="02_it_support/github-access.md",
        content=(
            "# Доступ к GitHub\n\n"
            "Создайте заявку GitHub Access.\n"
            "Получите подтверждение владельца репозитория."
        ),
    )


# ------------------------------------------------------------
# Helper: fake PostgreSQL
# ------------------------------------------------------------

def mock_database(monkeypatch, rows):
    """
    Подменяет PostgreSQL тестовыми данными.

    Настоящая база данных для этих тестов
    не требуется.
    """

    connection = MagicMock()
    cursor = MagicMock()

    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor

    cursor.fetchall.return_value = rows

    monkeypatch.setattr(
        answer.psycopg2,
        "connect",
        lambda **kwargs: connection,
    )

    return connection, cursor


# ------------------------------------------------------------
# 1. Question validation
# ------------------------------------------------------------

def test_empty_question_is_rejected():
    with pytest.raises(
        ValueError,
        match="Вопрос не может быть пустым"
    ):
        answer.ask_firstday("")


def test_invalid_access_level_is_rejected():
    with pytest.raises(
        ValueError,
        match="Неизвестный уровень доступа"
    ):
        answer.ask_firstday(
            "Как настроить VPN?",
            access_level="SUPER_ADMIN",
        )


# ------------------------------------------------------------
# 2. Question classification
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "Какие задачи у Junior Java Developer на первой неделе?",
        "Что делает Java Developer в первую неделю?",
        "Java Developer first week tasks",
        "Java Developer неделя 1",
    ],
)
def test_java_first_week_detection(question):
    assert answer.is_java_first_week_question(
        question
    ) is True


def test_other_roles_not_classified_as_java():
    assert answer.is_java_first_week_question(
        "Какие задачи у Junior DevOps на первой неделе?"
    ) is False


def test_access_status_detection():
    question = (
        "Я уже получил доступ к GitHub? "
        "Проверь статус моей заявки."
    )

    assert answer.is_access_status_question(
        question
    ) is True


def test_github_instruction_not_status_question():
    assert answer.is_access_status_question(
        "Как получить доступ к GitHub?"
    ) is False


# ------------------------------------------------------------
# 3. Document cleaning
# ------------------------------------------------------------

def test_remove_service_sections():
    document = (
        "# Настройка VPN\n\n"
        "1. Установите VPN-клиент.\n"
        "2. Импортируйте профиль.\n\n"
        "## Контрольный результат\n"
        "После выполнения инструкции сотрудник должен..."
    )

    cleaned = answer.clean_chunk(document)

    assert "Установите VPN-клиент" in cleaned
    assert "Импортируйте профиль" in cleaned
    assert "Контрольный результат" not in cleaned


def test_only_service_content_is_removed():
    document = (
        "После выполнения инструкции сотрудник должен "
        "уметь объяснить процесс."
    )

    assert answer.clean_chunk(document) == ""


# ------------------------------------------------------------
# 4. Complete document retrieval
# ------------------------------------------------------------

def test_document_chunks_are_combined(
    monkeypatch,
    vpn_document,
):
    """
    Проверяет исправленную ошибку с VPN:
    система должна получить оба фрагмента.
    """

    path = vpn_document["file_path"]

    mock_database(
        monkeypatch,
        rows=[
            (
                path,
                "EMPLOYEE",
                0,
                "# VPN\n1. Создайте заявку VPN Access.",
            ),
            (
                path,
                "EMPLOYEE",
                1,
                "2. Импортируйте профиль.\n"
                "3. Проверьте доступ к Nimbus.",
            ),
        ],
    )

    selected = answer.select_context_sources(
        [vpn_document]
    )

    assert len(selected) == 1

    content = selected[0]["content"]

    assert "Создайте заявку" in content
    assert "Импортируйте профиль" in content
    assert "Проверьте доступ" in content


def test_duplicate_document_is_selected_once(
    monkeypatch,
    vpn_document,
):
    path = vpn_document["file_path"]

    mock_database(
        monkeypatch,
        rows=[
            (
                path,
                "EMPLOYEE",
                0,
                "# VPN\n"
                "Настройка корпоративного VPN-клиента.",
            ),
        ],
    )

    selected = answer.select_context_sources(
        [vpn_document, vpn_document]
    )

    assert len(selected) == 1
    assert selected[0]["file_path"] == path


def test_document_access_level_must_match(
    monkeypatch,
    vpn_document,
):
    """
    Проверяем дополнительную защиту
    при загрузке полного документа.
    """

    mock_database(
        monkeypatch,
        rows=[
            (
                vpn_document["file_path"],
                "ADMIN",
                0,
                "Конфиденциальная инструкция для администратора.",
            ),
        ],
    )

    selected = answer.select_context_sources(
        [vpn_document]
    )

    assert selected == []


# ------------------------------------------------------------
# 5. Java Developer first-week answer
# ------------------------------------------------------------

def test_java_first_week_excludes_later_tasks(
    monkeypatch,
):
    """
    Проверяем, что дни 6–30 не попадают
    в список задач первой недели.
    """

    mock_database(
        monkeypatch,
        rows=[
            (
                "# Junior Java Developer\n\n"
                "Дни 1–5: установка JDK 21, "
                "запуск сервиса и знакомство "
                "с архитектурой.\n"
                "Дни 6–10: изучение REST API.\n"
                "Дни 11–20: разработка endpoint.\n"
                "Дни 21–30: итоговая демонстрация.",
            ),
        ],
    )

    result = answer.ask_firstday(
        "Какие задачи у Junior Java Developer "
        "на первой неделе?"
    )

    text = result["answer"]

    assert "JDK 21" in text
    assert "запуск сервиса" in text

    assert "REST API" not in text
    assert "endpoint" not in text
    assert "итоговая демонстрация" not in text

    assert len(result["sources"]) == 1
    assert (
        result["sources"][0]["file_path"]
        == "08_learning/java-track.md"
    )


def test_java_answer_does_not_claim_completed(
    monkeypatch,
):
    mock_database(
        monkeypatch,
        rows=[
            (
                "Дни 1–5: установка JDK 21 "
                "и запуск сервиса.",
            ),
        ],
    )

    result = answer.ask_firstday(
        "Java Developer first week tasks"
    )

    text = result["answer"].lower()

    assert "запланированы" in text
    assert "не подтверждение" in text


# ------------------------------------------------------------
# 6. Access request status
# ------------------------------------------------------------

def test_access_status_is_not_invented(
    monkeypatch,
):
    """
    Бот не должен обращаться к LLM,
    чтобы угадать статус реальной заявки.
    """

    def forbidden_search(*args, **kwargs):
        raise AssertionError(
            "Статус заявки нельзя получать через RAG"
        )

    monkeypatch.setattr(
        answer,
        "search_documents",
        forbidden_search,
    )

    result = answer.ask_firstday(
        "Я уже получил доступ к GitHub? "
        "Проверь статус моей заявки."
    )

    text = result["answer"].lower()

    assert "не могу проверить" in text
    assert "не подключён" in text
    assert result["sources"] == []


# ------------------------------------------------------------
# 7. RAG pipeline and source references
# ------------------------------------------------------------

def test_rag_returns_correct_source(
    monkeypatch,
    github_document,
):
    """
    Проверяет правильную передачу источников
    от поиска к генерации и обратно.
    """

    monkeypatch.setattr(
        answer,
        "search_documents",
        lambda **kwargs: [github_document],
    )

    monkeypatch.setattr(
        answer,
        "select_context_sources",
        lambda results, max_documents=3: results,
    )

    captured = {}

    def fake_generate(question, context):
        captured["question"] = question
        captured["context"] = context

        return (
            "Создайте заявку GitHub Access [1]."
        )

    monkeypatch.setattr(
        answer,
        "generate_answer",
        fake_generate,
    )

    result = answer.ask_firstday(
        "Как получить доступ к GitHub?"
    )

    assert "GitHub Access" in result["answer"]
    assert "[1]" in result["answer"]

    assert len(result["sources"]) == 1
    assert (
        result["sources"][0]["file_path"]
        == "02_it_support/github-access.md"
    )

    assert "Доступ к GitHub" in captured["context"]
    assert captured["question"] == (
        "Как получить доступ к GitHub?"
    )


# ------------------------------------------------------------
# 8. Search access filtering
# ------------------------------------------------------------

def test_search_filters_documents_by_access_level(
    monkeypatch,
):
    """
    Проверяет, какие уровни доступа передаются
    в SQL-запрос векторного поиска.

    Это unit-тест фильтрации, а не проверка
    аутентификации реального пользователя.
    """

    monkeypatch.setattr(
        search,
        "create_embedding",
        lambda question: "[0,0,0]",
    )

    connection = MagicMock()
    cursor = MagicMock()

    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    cursor.fetchall.return_value = []

    monkeypatch.setattr(
        search.psycopg2,
        "connect",
        lambda **kwargs: connection,
    )

    search.search_documents(
        question="Как получить GitHub Access?",
        access_level="EMPLOYEE",
    )

    sql, params = cursor.execute.call_args.args

    assert "access_level" in sql
    assert params[1] == ["EMPLOYEE"]


def test_invalid_search_access_level():
    with pytest.raises(ValueError):
        search.search_documents(
            "Как настроить VPN?",
            access_level="UNKNOWN",
        )

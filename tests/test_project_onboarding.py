"""Project boundaries and chat contracts. Offline, with no employee records or network."""
from datetime import date
from uuid import uuid4
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import project_routes as projects
from backend import project_chat
from backend import chat_history
from backend import main as api
from backend.auth_routes import UserOut, current_user


@pytest.fixture
def user():
    return UserOut(id=uuid4(), full_name="Synthetic tester", email="tester@example.invalid",
                   role="EMPLOYEE", job_title="Senior Engineer", department="Testing", start_date=date.today())


@pytest.fixture
def client(user):
    api.app.dependency_overrides[current_user] = lambda: user
    with TestClient(api.app) as result:
        yield result
    api.app.dependency_overrides.clear()


def connection_mock(monkeypatch, module):
    conn = MagicMock()
    cursor = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr(module.psycopg2, "connect", lambda **kwargs: conn)
    return cursor


def test_project_routes_require_authentication():
    with TestClient(api.app) as client:
        assert client.get('/api/v1/projects').status_code == 401
        assert client.post(f'/api/v1/projects/{uuid4()}/join', json={}).status_code == 401


def test_nonmember_cannot_read_project():
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    with pytest.raises(HTTPException) as caught:
        projects.require_membership(cursor, uuid4(), uuid4())
    assert caught.value.status_code == 404


def test_membership_uses_authenticated_identity():
    cursor = MagicMock()
    cursor.fetchone.return_value = {'id': 'p', 'slug': 'sample', 'name': 'Sample', 'project_role': 'QA'}
    project_id, employee_id = uuid4(), uuid4()
    projects.require_membership(cursor, project_id, employee_id)
    sql, params = cursor.execute.call_args.args
    assert 'm.employee_id=%s' in sql
    assert params == (str(project_id), str(employee_id))


@pytest.mark.parametrize('role', ['ENGINEERING', 'QA', 'DEVOPS', 'GENERAL'])
def test_project_routes_are_not_limited_to_juniors(role):
    tasks = projects.task_template(role)
    assert len(tasks) == 6
    assert {task[0] for task in tasks} == {'CONTEXT', 'ENVIRONMENT', 'DELIVERY'}
    assert 'junior' not in str(tasks).lower()


def test_blocker_requires_explanation():
    with pytest.raises(ValidationError):
        projects.TaskUpdate(status='BLOCKED', blocker_note='   ')
    assert projects.TaskUpdate(status='BLOCKED', blocker_note=' No sandbox ').blocker_note == 'No sandbox'
    assert projects.TaskUpdate(status='DONE', blocker_note='old note').blocker_note == ''


def test_cannot_self_join_private_project(client, monkeypatch):
    cursor = connection_mock(monkeypatch, projects)
    cursor.fetchone.return_value = None
    response = client.post(f'/api/v1/projects/{uuid4()}/join', json={'project_role': 'QA'})
    assert response.status_code == 404
    assert len(cursor.execute.call_args_list) == 1
    assert 'm.is_active=TRUE' in cursor.execute.call_args.args[0]


def test_task_update_scopes_project_and_employee(client, monkeypatch, user):
    cursor = connection_mock(monkeypatch, projects)
    monkeypatch.setattr(projects, 'require_membership', lambda *args: {})
    cursor.fetchone.return_value = None
    project_id, task_id = uuid4(), uuid4()
    response = client.patch(f'/api/v1/projects/{project_id}/onboarding/tasks/{task_id}', json={'status': 'DONE'})
    assert response.status_code == 404
    sql, params = cursor.execute.call_args.args
    assert 'project_id=%s AND employee_id=%s' in sql
    assert params[-3:] == (str(task_id), str(project_id), str(user.id))


def test_project_docs_authorize_before_read(monkeypatch):
    cursor = connection_mock(monkeypatch, projects)
    cursor.fetchone.return_value = None
    with pytest.raises(HTTPException):
        projects.read_project_documents(uuid4(), uuid4())
    assert len(cursor.execute.call_args_list) == 1


def test_project_document_view_rejects_nonmember(client, monkeypatch):
    def denied(*args, **kwargs):
        raise HTTPException(404, 'unavailable')
    monkeypatch.setattr('backend.document_routes.read_project_documents', denied)
    response = client.get('/api/v1/documents/content', params={'path': 'projects/sample/overview.md', 'project_id': str(uuid4())})
    assert response.status_code == 404


def test_lexical_retrieval_rejects_undocumented_tool():
    documents = [{'title': 'Доступы и окружение', 'content': 'Получите доступ к GitHub и VPN.', 'file_path': 'projects/a/environment.md'}]
    assert project_chat.rank_project_documents('Как получить доступ к Slack?', documents) == []
    assert project_chat.rank_project_documents('Как получить доступ к GitHub?', documents)


def test_project_rag_never_uses_corporate_search(monkeypatch):
    documents = [{'title': 'Архитектура проекта', 'content': 'Компоненты: gateway и audit.', 'file_path': 'projects/a/architecture.md'}]
    monkeypatch.setattr(project_chat, 'read_project_documents', lambda *args: documents)
    captured = {}
    def generate(question, context, *args):
        captured['context'] = context
        return 'Компоненты: gateway и audit.'
    monkeypatch.setattr(project_chat, 'generate_answer', generate)
    result = project_chat.ask_project('Какая архитектура проекта?', uuid4(), uuid4())
    assert result['sources'][0]['file_path'] == 'projects/a/architecture.md'
    assert 'gateway' in captured['context']


def test_project_abstention_does_not_call_model(monkeypatch):
    monkeypatch.setattr(project_chat, 'read_project_documents', lambda *args: [])
    def forbidden(*args, **kwargs):
        raise AssertionError('No evidence: generation must not run')
    monkeypatch.setattr(project_chat, 'generate_answer', forbidden)
    result = project_chat.ask_project('Unknown instruction', uuid4(), uuid4())
    assert result['sources'] == []


def test_project_stream_preserves_real_chunks(monkeypatch):
    docs = [{'title': 'Архитектура', 'content': 'gateway', 'file_path': 'projects/a/architecture.md', 'similarity': None}]
    monkeypatch.setattr(project_chat, 'prepare_project_rag', lambda *args: docs)
    monkeypatch.setattr(project_chat, 'generate_answer_stream', lambda *args: iter(['gateway', ' и audit']))
    events = list(project_chat.stream_project('Архитектура?', uuid4(), uuid4()))
    assert [event['type'] for event in events] == ['sources', 'delta', 'delta']
    assert events[1]['text'] == 'gateway'


def test_history_rejects_cross_project_context(monkeypatch):
    connection_mock(monkeypatch, chat_history)
    monkeypatch.setattr(chat_history, '_find_owned', lambda *args: {'project_id': uuid4()})
    with pytest.raises(HTTPException) as caught:
        chat_history.recent_questions(uuid4(), uuid4(), uuid4())
    assert caught.value.status_code == 409


def test_history_handles_string_uuid_from_database():
    project_id = uuid4()
    assert chat_history.same_project(str(project_id), project_id)
    assert chat_history.same_project(None, None)
    assert not chat_history.same_project(str(project_id), None)


def test_history_requires_current_membership(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = {'id': uuid4(), 'title': 'Sample', 'project_id': uuid4()}
    def denied(*args):
        raise HTTPException(404, 'unavailable')
    monkeypatch.setattr(chat_history, 'require_membership', denied)
    with pytest.raises(HTTPException):
        chat_history._find_owned(cursor, uuid4(), uuid4())


def test_chat_validates_membership_before_personal_answers(client, monkeypatch):
    def denied(*args):
        raise HTTPException(404, 'unavailable')
    monkeypatch.setattr(api, 'check_project_access', denied)
    response = client.post('/api/v1/chat', json={'question': 'Мой прогресс', 'project_id': str(uuid4())})
    assert response.status_code == 404


def test_project_chat_stream_keeps_ndjson_contract(client, monkeypatch):
    monkeypatch.setattr(api, 'check_project_access', lambda *args: None)
    monkeypatch.setattr(api, 'project_personal_answer', lambda *args: None)
    monkeypatch.setattr(api, 'stream_project', lambda **kwargs: iter([
        {'type': 'sources', 'sources': []}, {'type': 'delta', 'text': 'A'}, {'type': 'delta', 'text': 'B'},
    ]))
    monkeypatch.setattr(api, 'save_exchange', lambda **kwargs: uuid4())
    response = client.post('/api/v1/chat/stream', json={'question': 'Архитектура?', 'project_id': str(uuid4())})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('application/x-ndjson')
    import json
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event['type'] for event in events] == ['status', 'delta', 'delta', 'done']


def test_project_personal_progress_never_enters_generation(monkeypatch, user):
    cursor = connection_mock(monkeypatch, project_chat)
    monkeypatch.setattr(project_chat, 'project_plan', lambda *args: projects.ProjectPlan(
        project_id=uuid4(), project_name='Sample', project_role='ENGINEERING', tasks=[],
        total_tasks=6, completed_tasks=3, blocked_tasks=1, progress_percent=50,
    ))
    answer = project_chat.project_personal_answer('Какой мой прогресс адаптации?', user, uuid4())
    assert '50%' in answer
    assert '3 из 6' in answer
    assert not cursor.execute.called


@pytest.mark.parametrize('question', ['Что делать, если GitHub заблокирован?', 'Как восстановить доступ к MFA?'])
def test_technical_lockout_is_not_a_personal_project_blocker(monkeypatch, user, question):
    def forbidden(**kwargs):
        raise AssertionError('Technical instructions must continue to document retrieval')
    monkeypatch.setattr(project_chat.psycopg2, 'connect', forbidden)
    assert project_chat.project_personal_answer(question, user, uuid4()) is None


def test_personal_blocker_is_answered_locally(monkeypatch, user):
    connection_mock(monkeypatch, project_chat)
    monkeypatch.setattr(project_chat, 'project_plan', lambda *args: projects.ProjectPlan(
        project_id=uuid4(), project_name='Sample', project_role='ENGINEERING',
        tasks=[projects.ProjectTask(id=uuid4(), position=1, stage='ENVIRONMENT', title='Sandbox',
                                   description='Check sandbox', document_path=None,
                                   status='BLOCKED', blocker_note='Synthetic access blocker')],
        total_tasks=1, completed_tasks=0, blocked_tasks=1, progress_percent=0,
    ))
    answer = project_chat.project_personal_answer('Что мешает моему подключению?', user, uuid4())
    assert 'Synthetic access blocker' in answer

"""Authorization regression tests with synthetic data and no database/network."""
from datetime import date
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import admin_routes as admin
from backend import project_routes as projects
from backend.auth_routes import RegisterBody, UserOut, current_user
from backend.main import app


@pytest.fixture
def user():
    return UserOut(id=uuid4(), full_name='Synthetic user', email='synthetic@example.invalid',
                   role='EMPLOYEE', job_title='Engineer', department='Testing', start_date=date.today())


@pytest.fixture
def client(user):
    app.dependency_overrides[current_user] = lambda: user
    with TestClient(app) as value:
        yield value
    app.dependency_overrides.clear()


@pytest.mark.parametrize('method,path,body', [
    ('get', '/api/v1/admin/projects', None),
    ('get', f'/api/v1/admin/employees/{uuid4()}/projects', None),
    ('put', f'/api/v1/admin/employees/{uuid4()}/projects/{uuid4()}', {}),
    ('patch', f'/api/v1/admin/employees/{uuid4()}/profile', {'employment_type': 'INTERN', 'onboarding_mode': 'PROJECT'}),
])
def test_assignment_routes_require_admin(client, method, path, body):
    kwargs = {'json': body} if body is not None else {}
    assert getattr(client, method)(path, **kwargs).status_code == 403


def test_hr_cannot_assign(user):
    with pytest.raises(HTTPException) as caught:
        admin.require_admin(user.model_copy(update={'role': 'HR'}))
    assert caught.value.status_code == 403


@pytest.mark.parametrize('field,value', [('employment_type', 'ADMIN'), ('onboarding_mode', 'ALL')])
def test_registration_profile_is_constrained(field, value):
    body = dict(full_name='Synthetic tester', email='test@example.invalid', password='synthetic-password-123', job_track='QA')
    body[field] = value
    with pytest.raises(ValidationError):
        RegisterBody(**body)


def test_intern_and_project_are_independent():
    body = RegisterBody(full_name='Synthetic tester', email='test@example.invalid', password='synthetic-password-123',
                        job_track='QA', employment_type='INTERN', onboarding_mode='PROJECT')
    assert body.employment_type == 'INTERN' and body.onboarding_mode == 'PROJECT'


def test_projects_catalog_is_membership_only(monkeypatch, user):
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    monkeypatch.setattr(projects.psycopg2, 'connect', lambda **kwargs: connection)
    assert projects.list_projects(user) == []
    sql, params = cursor.execute.call_args.args
    assert 'WHERE m.is_active=TRUE' in sql
    assert 'WHERE p.is_demo OR' not in sql
    assert params == (str(user.id),)


def test_demo_join_cannot_create_membership(monkeypatch, user):
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = None
    monkeypatch.setattr(projects.psycopg2, 'connect', lambda **kwargs: connection)
    with pytest.raises(HTTPException) as caught:
        projects.join_demo_project(uuid4(), projects.JoinBody(), user)
    assert caught.value.status_code == 404
    assert len(cursor.execute.call_args_list) == 1
    assert 'INSERT' not in cursor.execute.call_args.args[0]


def test_assignment_preserves_existing_role(monkeypatch, user):
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [{'id': uuid4(), 'slug': 'synthetic', 'name': 'Synthetic'}, {'id': uuid4()},
                                   {'project_role': 'QA', 'is_active': False}]
    monkeypatch.setattr(admin.psycopg2, 'connect', lambda **kwargs: connection)
    with pytest.raises(HTTPException) as caught:
        admin.assign_project(uuid4(), uuid4(), admin.AssignmentBody(project_role='DEVOPS'), user)
    assert caught.value.status_code == 409
    assert all('INSERT' not in call.args[0] and 'UPDATE project_memberships' not in call.args[0]
               for call in cursor.execute.call_args_list)


def test_revoke_retains_tasks_and_history(monkeypatch, user):
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [{'id': uuid4(), 'slug': 'synthetic', 'name': 'Synthetic'}, {'id': uuid4()},
                                   {'project_role': 'QA', 'is_active': True}]
    monkeypatch.setattr(admin.psycopg2, 'connect', lambda **kwargs: connection)
    result = admin.assign_project(uuid4(), uuid4(), admin.AssignmentBody(is_active=False), user)
    assert result.is_active is False and result.project_role == 'QA'
    assert 'SET is_active=FALSE' in cursor.execute.call_args.args[0]
    assert all('DELETE' not in call.args[0] for call in cursor.execute.call_args_list)

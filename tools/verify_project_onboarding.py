"""Local API smoke checks; all synthetic records are rolled back.

python -m tools.verify_project_onboarding [--generate]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


class TransactionConnection:
    def __init__(self, connection):
        self.connection = connection
        self.counter = 0
        self.savepoints = []

    def __enter__(self):
        self.counter += 1
        name = f"firstday_smoke_{self.counter}"
        self.savepoints.append(name)
        with self.connection.cursor() as cursor:
            cursor.execute(f"SAVEPOINT {name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        name = self.savepoints.pop()
        with self.connection.cursor() as cursor:
            if exc_type:
                cursor.execute(f"ROLLBACK TO SAVEPOINT {name}")
            cursor.execute(f"RELEASE SAVEPOINT {name}")
        return False

    def cursor(self, *args, **kwargs):
        return self.connection.cursor(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    os.environ['LANGFUSE_ENABLED'] = 'false'
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env', override=False)
    except ImportError:
        pass
    from fastapi.testclient import TestClient
    import psycopg2
    from backend import main as api
    from backend.auth_routes import DB_CONFIG, UserOut, current_user
    conn = None
    total = 0

    def check(condition, label):
        nonlocal total
        if not condition:
            raise AssertionError(label)
        total += 1
        print(f'PASS {label}', flush=True)

    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=5)
        user_id, other_id, admin_id, private_id = uuid4(), uuid4(), uuid4(), uuid4()
        with conn.cursor() as cursor:
            for identity in (user_id, other_id, admin_id):
                cursor.execute(
                    """INSERT INTO employees (id,full_name,email,role,job_title,department,start_date)
                       VALUES (%s,'Synthetic API check',%s,'EMPLOYEE','Senior Engineer','Synthetic testing',CURRENT_DATE)""",
                    (str(identity), f"smoke-{identity}@example.invalid"),
                )
            cursor.execute("INSERT INTO projects (id,slug,name,description,is_demo) VALUES (%s,%s,'Synthetic private project','Boundary check',FALSE)",
                           (str(private_id), f"smoke-{private_id}"))
            cursor.execute("SELECT slug, id FROM projects WHERE slug IN ('orbitflow','beacon')")
            demo_ids = dict(cursor.fetchall())
            orbit, beacon = str(demo_ids['orbitflow']), str(demo_ids['beacon'])
        user = UserOut(id=user_id, full_name='Synthetic API check', email='smoke@example.invalid', role='EMPLOYEE',
                       job_title='Senior Engineer', department='Synthetic testing', start_date=date.today())
        api.app.dependency_overrides[current_user] = lambda: user
        with patch('psycopg2.connect', return_value=TransactionConnection(conn)), TestClient(api.app) as client:
            admin_user = user.model_copy(update={'id': admin_id, 'role': 'ADMIN'})
            def assignment(project_id, role='ENGINEERING', active=True, employee_id=user_id):
                nonlocal user
                previous, user = user, admin_user
                try:
                    return client.put(f'/api/v1/admin/employees/{employee_id}/projects/{project_id}',
                                      json={'project_role': role, 'is_active': active})
                finally:
                    user = previous
            response = client.get('/api/v1/projects')
            check(response.status_code == 200 and response.json() == [], 'all unassigned projects hidden including demos')
            check(client.get('/api/v1/admin/projects').status_code == 403, 'employee cannot enumerate admin catalog')
            check(client.put(f'/api/v1/admin/employees/{user_id}/projects/{orbit}', json={}).status_code == 403,
                  'employee cannot grant own access')
            check(client.get(f'/api/v1/projects/{orbit}/onboarding').status_code == 404, 'nonmember plan denied')
            check(client.get(f'/api/v1/projects/{orbit}/documents').status_code == 404, 'nonmember documents denied')
            check(client.post(f'/api/v1/projects/{private_id}/join', json={}).status_code == 404, 'private enrollment denied')
            check(client.post(f'/api/v1/projects/{orbit}/join', json={}).status_code == 404, 'demo enrollment denied without admin assignment')
            check(assignment(orbit).status_code == 200, 'admin assigns project')
            check([item['id'] for item in client.get('/api/v1/projects').json()] == [orbit], 'only assigned project visible')
            joined = client.post(f'/api/v1/projects/{orbit}/join', json={'project_role': 'ENGINEERING'})
            check(joined.status_code == 200 and joined.json()['total_tasks'] == 6, 'personal project route created')
            task_id = joined.json()['tasks'][0]['id']
            repeated = client.post(f'/api/v1/projects/{orbit}/join', json={'project_role': 'QA'})
            check(repeated.json()['tasks'][0]['id'] == task_id and repeated.json()['project_role'] == 'ENGINEERING', 'idempotent enrollment')
            task_url = f'/api/v1/projects/{orbit}/onboarding/tasks/{task_id}'
            check(client.patch(task_url, json={'status': 'BLOCKED'}).status_code == 422, 'empty blocker rejected')
            blocked = client.patch(task_url, json={'status': 'BLOCKED', 'blocker_note': 'Synthetic sandbox blocker'})
            check(blocked.status_code == 200 and blocked.json()['blocked_tasks'] == 1, 'blocker persisted')
            personal_blocker = client.post('/api/v1/chat', json={'project_id': orbit, 'question': 'Что мешает моему подключению?'})
            check(personal_blocker.status_code == 200 and 'Synthetic sandbox blocker' in personal_blocker.json()['answer'], 'personal blocker answered locally')
            done = client.patch(task_url, json={'status': 'DONE'})
            check(done.json()['completed_tasks'] == 1 and done.json()['tasks'][0]['blocker_note'] == '', 'progress and blocker cleared')
            docs = client.get(f'/api/v1/projects/{orbit}/documents').json()['items']
            check(len(docs) == 4 and all(doc['file_path'].startswith('projects/orbitflow/') for doc in docs), 'project documents isolated')
            view = client.get('/api/v1/documents/content', params={'project_id': orbit, 'path': docs[0]['file_path']})
            check(view.status_code == 200 and bool(view.json()['content']), 'project source opens')
            personal = client.post('/api/v1/chat', json={'project_id': orbit, 'question': 'Какой мой прогресс адаптации?'})
            check(personal.status_code == 200 and '1 из 6' in personal.json()['answer'] and not personal.json()['sources'], 'personal answer from database')
            conversation = personal.json()['conversation_id']
            followup = client.post('/api/v1/chat', json={'project_id': orbit, 'conversation_id': conversation, 'question': 'Какие у меня остались задачи?'})
            check(followup.status_code == 200, 'project conversation continues')
            check(len(client.get('/api/v1/chat/conversations', params={'project_id': orbit}).json()) == 2, 'project history persisted')
            check(client.get('/api/v1/chat/conversations').json() == [], 'corporate history isolated')
            check(assignment(beacon, 'DEVOPS').status_code == 200, 'admin assigns second project with separate role')
            for scope in ({'project_id': beacon}, {}):
                mismatch = client.post('/api/v1/chat', json={**scope, 'conversation_id': conversation, 'question': 'Какие у меня остались задачи?'})
                check(mismatch.status_code == 409, 'conversation scope mismatch rejected')
            response = client.post('/api/v1/chat/stream', json={'project_id': orbit, 'question': 'Как получить доступ к Slack?'})
            events = [json.loads(line) for line in response.text.splitlines()]
            check([event['type'] for event in events] == ['status', 'delta', 'done'] and events[-1]['sources'] == [], 'streaming abstention contract')
            with patch.object(api, 'stream_project', return_value=iter([
                {'type': 'sources', 'sources': []}, {'type': 'delta', 'text': 'Synthetic '}, {'type': 'delta', 'text': 'stream'},
            ])):
                response = client.post('/api/v1/chat/stream', json={'project_id': orbit, 'question': 'Архитектура проекта?'})
                events = [json.loads(line) for line in response.text.splitlines()]
                check([event['type'] for event in events] == ['status', 'delta', 'delta', 'done'], 'chunk streaming saved')
            if args.generate:
                response = client.post('/api/v1/chat/stream', json={'project_id': orbit, 'question': 'Как поднять проект локально?'})
                events = [json.loads(line) for line in response.text.splitlines()]
                check(events[-1]['type'] == 'done' and bool(events[-1]['sources']) and sum(event['type'] == 'delta' for event in events) > 1,
                      'real Ollama project streaming')
            check(assignment(orbit, active=False).status_code == 200, 'admin revokes membership without deleting progress')
            check(all(item['id'] != orbit for item in client.get('/api/v1/projects').json()), 'revoked project hidden')
            check(client.get(f'/api/v1/projects/{orbit}/documents').status_code == 404, 'revoked documents denied')
            check(client.get(f'/api/v1/projects/{orbit}/onboarding').status_code == 404, 'revoked plan denied')
            check(client.get(f'/api/v1/chat/conversations/{conversation}').status_code == 404, 'revoked history denied')
            check(client.get('/api/v1/chat/conversations', params={'project_id': orbit}).status_code == 404, 'revoked history list denied')
            check(client.get('/api/v1/documents/content', params={'project_id': orbit, 'path': docs[0]['file_path']}).status_code == 404,
                  'revoked source viewer denied')
            check(client.patch(task_url, json={'status': 'TODO'}).status_code == 404, 'revoked task write denied')
            check(client.post('/api/v1/chat', json={'project_id': orbit, 'question': 'Мой прогресс'}).status_code == 404,
                  'revoked chat denied')
            revoked_stream = client.post('/api/v1/chat/stream', json={'project_id': orbit, 'question': 'Мой прогресс'})
            check(revoked_stream.status_code == 404 or all(json.loads(line)['type'] != 'done' for line in revoked_stream.text.splitlines()),
                  'revoked streaming denied')
            check(assignment(orbit, 'QA').status_code == 409, 'role change cannot silently reset old route')
            check(assignment(orbit).status_code == 200, 'admin reactivates original route')
            restored = client.get(f'/api/v1/projects/{orbit}/onboarding').json()
            check(restored['tasks'][0]['id'] == task_id and restored['completed_tasks'] == 1, 'reactivation preserves task ids and progress')
            check(client.get(f'/api/v1/chat/conversations/{conversation}').status_code == 200, 'reactivation restores retained history')
            check(assignment(str(private_id), 'QA').status_code == 200, 'admin can assign non-demo projects')
            user = admin_user
            assigned_list = client.get(f'/api/v1/admin/employees/{user_id}/projects')
            check(assigned_list.status_code == 200 and len(assigned_list.json()) == 3, 'admin sees target employee assignments')
            check(client.get(f'/api/v1/projects/{orbit}/documents').status_code == 404, 'admin catalog role does not bypass content membership')
            changed_profile = client.patch(f'/api/v1/admin/employees/{other_id}/profile',
                                           json={'employment_type': 'INTERN', 'onboarding_mode': 'PROJECT'})
            check(changed_profile.status_code == 200 and changed_profile.json()['employment_type'] == 'INTERN'
                  and changed_profile.json()['role'] == 'EMPLOYEE', 'admin can set scenario without changing security role')
            user = user.model_copy(update={'id': other_id})
            user = user.model_copy(update={'role': 'EMPLOYEE'})
            check(client.get('/api/v1/projects').json() == [], 'scenario update grants no projects')
            check(client.get(f'/api/v1/chat/conversations/{conversation}').status_code == 404, 'other user history denied')
            check(assignment(orbit, employee_id=other_id).status_code == 200, 'admin assigns another employee')
            check(client.patch(task_url, json={'status': 'DONE'}).status_code == 404, 'other member task denied')
            # Exercise real registration/auth after the profession-label change.
            credentials = {'email': f'registration-smoke-{uuid4()}@example.invalid', 'password': uuid4().hex + uuid4().hex}
            registered = client.post('/api/v1/auth/register', json={**credentials, 'full_name': 'Synthetic registration check', 'job_track': 'JAVA',
                                                                 'employment_type': 'INTERN', 'onboarding_mode': 'PROJECT', 'role': 'ADMIN'})
            check(registered.status_code == 201 and registered.json()['user']['job_title'] == 'Java Developer', 'registration does not assume junior seniority')
            signed_in = client.post('/api/v1/auth/login', json=credentials)
            check(signed_in.status_code == 200, 'real password login')
            api.app.dependency_overrides.pop(current_user)
            headers = {'Authorization': f"Bearer {signed_in.json()['access_token']}"}
            me = client.get('/api/v1/auth/me', headers=headers)
            check(me.status_code == 200 and me.json()['id'] == registered.json()['user']['id'], 'real bearer session identity')
            check(me.json()['employment_type'] == 'INTERN' and me.json()['onboarding_mode'] == 'PROJECT' and me.json()['role'] == 'EMPLOYEE',
                  'registration profile persists without security role escalation')
            check(client.get('/api/v1/projects', headers=headers).json() == [], 'new intern has no automatic project permissions')
            check(client.put(f'/api/v1/admin/employees/{registered.json()["user"]["id"]}/projects/{orbit}', headers=headers, json={}).status_code == 403,
                  'real session cannot self-assign project')
            check(client.post('/api/v1/auth/logout', headers=headers).status_code == 204, 'real session logout')
            check(client.get('/api/v1/auth/me', headers=headers).status_code == 401, 'revoked session rejected')
        print(f'RESULT: {total} checks passed; synthetic records will roll back.')
        return 0
    except Exception as exc:
        label = str(exc) if isinstance(exc, AssertionError) else type(exc).__name__
        print(f'FAIL: {label}')
        return 1
    finally:
        api.app.dependency_overrides.clear()
        if conn is not None:
            conn.rollback()
            conn.close()


if __name__ == '__main__':
    raise SystemExit(main())

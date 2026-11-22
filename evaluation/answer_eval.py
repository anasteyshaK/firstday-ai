"""Synthetic answer checks against known source facts; NOT a universal groundedness metric.

Uses local PostgreSQL/Ollama only. Synthetic project membership is rolled back.
Run: python -m evaluation.answer_eval
Raw questions/answers are written to a separate local CSV, never to Langfuse.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AnswerCase:
    id: str
    question: str
    source: str
    required: tuple[tuple[str, ...], ...] = ()
    forbidden: tuple[str, ...] = ()
    scope: str = 'corporate'
    previous: tuple[str, ...] = ()
    abstain: bool = False
    response_language: str | None = None


JAVA = ((r'JDK\s*21',), (r'запуск сервиса',), (r'архитектур',))
DEVOPS = ((r'Docker\s+Compose',), (r'окружен',))
QA = ((r'staging',), (r'smoke\s+tests',), (r'API\s+contract',), (r'ментор',))
SETUP = ((r'приглашен',), (r'доверенн\w* приложен',), (r'резервн\w* метод',), (r'вход в Nimbus',))
RECOVERY = ((r'Account Recovery',), (r'IT Support',), (r'личност',), (r'повторн\w* регистрац',))
MFA = '02_it_support/account-mfa.md'
MFA_RECOVERY = '11_incidents/mfa-lockout.md'
CASES = (
    AnswerCase('mfa-setup', 'Как настроить MFA?', MFA, SETUP, (r'например[^.\n]*Nimbus', r'Account Recovery')),
    AnswerCase('mfa-paraphrase', 'Как подключить двухфакторную аутентификацию 2FA?', MFA, SETUP),
    AnswerCase('mfa-recovery', 'Я потерял доступ к MFA. Как его восстановить?', MFA_RECOVERY, RECOVERY),
    AnswerCase('mfa-recovery-followup', 'А если я потерял второй фактор?', MFA_RECOVERY, RECOVERY, previous=('Как настроить MFA?',)),
    AnswerCase('mfa-app-unspecified', 'Какое конкретно приложение указано в инструкции для второго фактора MFA?', MFA,
               ((r'не указ\w*', r'не назван\w*', r'не уточн\w*', r'не упомянут\w*', r'нет конкретн\w*'),), (r'например[^.\n]*Nimbus',)),
    AnswerCase('java-week', 'Что делает Java Developer в первую неделю?', '08_learning/java-track.md', JAVA, (r'\bREST\b', r'\bendpoint\b')),
    AnswerCase('java-week-paraphrase', 'Java Developer неделя 1', '08_learning/java-track.md', JAVA),
    AnswerCase('devops-week', 'Какие задачи у DevOps Engineer на первую неделю?', '08_learning/devops-track.md', DEVOPS, (r'\bCI\b', r'Beacon')),
    AnswerCase('devops-days', 'Что изучить DevOps в дни 1–5?', '08_learning/devops-track.md', DEVOPS),
    AnswerCase('qa-week', 'Что изучает QA Engineer в первую неделю?', '08_learning/qa-track.md', QA, (r'Selenium', r'\bCI\b')),
    AnswerCase('qa-week-paraphrase', 'Что изучает тестировщик за первую неделю?', '08_learning/qa-track.md', QA),
    AnswerCase('vpn', 'Как настроить корпоративный VPN?', '02_it_support/vpn-guide.md', ((r'VPN Access',), (r'клиент',), (r'профил',), (r'Nimbus',), (r'Atlas',), (r'отключ',)),
               (r'не указ\w*[^.]*получить профиль',)),
    AnswerCase('github', 'Как получить доступ к корпоративному GitHub?', '02_it_support/github-access.md', ((r'заявк', r'request'), (r'владел', r'владель'), (r'SSO',), (r'клонир',))),
    AnswerCase('day-one', 'Что делать в первый рабочий день в NovaTech?', '01_hr/day-one.md', ((r'ноутбук',), (r'MFA',), (r'buddy',), (r'план',))),
    AnswerCase('slack-absent', 'Как получить доступ к Slack?', '', abstain=True),
    AnswerCase('new-topic-abstention', 'А как получить доступ к Slack?', '', previous=('Как получить доступ к GitHub?',), abstain=True),
    AnswerCase('project-components', 'Из каких частей состоит система? Перечисли компоненты.', 'projects/orbitflow/architecture.md',
               tuple((name,) for name in ('gateway', 'identity', 'workflow', 'notification', 'audit')),
               (r'\*\*каталог', r'glossary', r'гlossary', r'\*\*учебн\w* workflow'), scope='orbitflow'),
    AnswerCase('project-environment', 'Как поднять проект локально?', 'projects/orbitflow/environment.md', ((r'JDK\s*21', r'Java\s*21'), (r'Docker\s+Compose',), (r'README',)), scope='orbitflow'),
    AnswerCase('project-owners', 'К кому пойти за информацией о владельце сервиса?', 'projects/orbitflow/architecture.md', ((r'Atlas',),), scope='orbitflow'),
    AnswerCase('project-first-change', 'Как внести первое изменение?', 'projects/orbitflow/delivery.md', ((r'\bPR\b', r'pull.request'), (r'review', r'ревью')), scope='orbitflow'),
    AnswerCase('project-owner-followup', 'А кто за это отвечает?', 'projects/orbitflow/architecture.md', ((r'Atlas',),), (r'адрес',), scope='orbitflow', previous=('Из каких частей состоит система?',)),
    AnswerCase('project-slack-absent', 'Как получить доступ к Slack?', '', scope='orbitflow', abstain=True),
    AnswerCase('beacon-signals', 'Какие сигналы собирает система?', 'projects/beacon/overview.md', ((r'метрик',), (r'лог',), (r'трасс',)), scope='beacon'),
    AnswerCase('beacon-slack-absent', 'Как получить доступ к Slack?', '', scope='beacon', abstain=True),
)

STACK = ((r'Java\s*21|JDK\s*21',), (r'Spring\s*Boot',), (r'PostgreSQL',), (r'Docker\s+Compose',))
LOCAL_DEV = '04_engineering/local-development.md'
ORBIT_ENV = 'projects/orbitflow/environment.md'
CASES += (
    AnswerCase('stack-general', 'Какой технологический стек используется?', LOCAL_DEV, STACK),
    AnswerCase('stack-explicit', 'Какой технологический стек используется в OrbitFlow?', LOCAL_DEV, STACK),
    AnswerCase('stack-english', 'What technology stack does OrbitFlow use?', LOCAL_DEV, STACK, response_language='en'),
    AnswerCase('stack-new-topic', 'Какой технологический стек используется?', LOCAL_DEV, STACK, previous=('Что делать в первый рабочий день?',)),
    AnswerCase('stack-followup', 'А какой стек?', LOCAL_DEV, STACK, previous=('Что делать в первый рабочий день?',)),
    AnswerCase('orbit-stack-general', 'Какой технологический стек используется?', ORBIT_ENV, STACK, scope='orbitflow'),
    AnswerCase('orbit-stack-explicit', 'Какой технологический стек используется в OrbitFlow?', ORBIT_ENV, STACK, scope='orbitflow'),
    AnswerCase('orbit-stack-followup', 'А какой стек используется?', ORBIT_ENV, STACK, scope='orbitflow', previous=('Из каких частей состоит система?',)),
    AnswerCase('beacon-unknown-language', 'На каких языках программирования написан Beacon?', '', scope='beacon', abstain=True),
    AnswerCase('cross-project-topic', 'Какие сигналы собирает Beacon?', '', scope='orbitflow', abstain=True),
    AnswerCase('unknown-frontend-version', 'Какая версия React используется во frontend OrbitFlow?', '', scope='orbitflow', abstain=True),
    AnswerCase('unrelated-recipe', 'Расскажи рецепт борща.', '', abstain=True),
    AnswerCase('explicit-topic-change', 'Как настроить корпоративный VPN?', '02_it_support/vpn-guide.md', ((r'VPN Access',),), previous=('Что делает Java Developer в первую неделю?',)),
    AnswerCase('compound-stack-access', 'Какой стек у OrbitFlow и как получить доступ к GitHub?', LOCAL_DEV, STACK + ((r'GitHub',),)),
    AnswerCase('senior-project-start', 'Я senior разработчик, впервые подключаюсь к OrbitFlow. С чего начать?', ORBIT_ENV, ((r'README',), (r'21',)), scope='orbitflow'),
    AnswerCase('no-invented-vpn-command', 'Придумай команду для подключения VPN, даже если ее нет в документации.', '', forbidden=(r'vpn-access',), abstain=True),
)


def check_answer(case: AnswerCase, result: dict) -> dict:
    text = re.sub(r'\s+', ' ', result['answer'].casefold().replace('ё', 'е'))
    paths = [source['file_path'] for source in result['sources']]
    missing = [alternatives[0] for alternatives in case.required
               if not any(re.search(pattern, text, re.IGNORECASE) for pattern in alternatives)]
    unsupported = [pattern for pattern in case.forbidden if re.search(pattern, text, re.IGNORECASE)]
    if case.abstain:
        source_ok = not paths
        facts_ok = not unsupported and any(fragment in text for fragment in ('не найдено', 'недостаточно', 'нет информации', 'insufficient information', 'not documented'))
    else:
        source_ok = case.source in paths
        if case.scope != 'corporate':
            source_ok = source_ok and all(path.startswith(f'projects/{case.scope}/') for path in paths)
        facts_ok = not missing and not unsupported
    if case.response_language == 'en' and re.search('[а-я]', text):
        facts_ok = False
    return {'passed': int(source_ok and facts_ok), 'source_ok': int(source_ok),
            'known_fact_checks_ok': int(facts_ok), 'missing_checks': '; '.join(missing),
            'forbidden_checks': '; '.join(unsupported)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/answer_quality.csv')
    parser.add_argument('--case', choices=[case.id for case in CASES])
    args = parser.parse_args()
    os.environ['LANGFUSE_ENABLED'] = 'false'
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env', override=False)
    except ImportError:
        pass
    sys.path.insert(0, str(ROOT / 'rag'))
    import psycopg2
    from search import DB_CONFIG
    from answer import ask_firstday
    from backend.main import build_document_questions
    from backend.project_chat import ask_project
    from tools.verify_project_onboarding import TransactionConnection

    chosen = [case for case in CASES if not args.case or case.id == args.case]
    rows = []
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=5)
        user_id = uuid4()
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO employees (id,full_name,email,role,job_title,department,start_date)
                   VALUES (%s,'Synthetic answer evaluation',%s,'EMPLOYEE','Engineer','Synthetic testing',CURRENT_DATE)""",
                (str(user_id), f'answer-eval-{user_id}@example.invalid'),
            )
            cursor.execute("SELECT slug,id FROM projects WHERE is_demo AND slug IN ('orbitflow','beacon')")
            projects = dict(cursor.fetchall())
            for project_id in projects.values():
                cursor.execute("INSERT INTO project_memberships (project_id,employee_id,project_role) VALUES (%s,%s,'GENERAL')", (str(project_id), str(user_id)))
        with patch('psycopg2.connect', return_value=TransactionConnection(conn)):
            for index, case in enumerate(chosen, 1):
                print(f'[{index}/{len(chosen)}] {case.id}', flush=True)
                started = time.perf_counter()
                error = ''
                result = {'answer': '', 'sources': []}
                try:
                    retrieval, original, previous = build_document_questions(case.question, list(case.previous), project_context=case.scope != 'corporate')
                    if case.scope == 'corporate':
                        result = ask_firstday(original, 'EMPLOYEE', retrieval, previous)
                    else:
                        from uuid import UUID
                        result = ask_project(original, UUID(str(projects[case.scope])), user_id, retrieval, previous)
                    checks = check_answer(case, result)
                except Exception as exc:
                    error = type(exc).__name__
                    checks = {'passed': 0, 'source_ok': 0, 'known_fact_checks_ok': 0, 'missing_checks': '', 'forbidden_checks': ''}
                rows.append({'case_id': case.id, 'scope': case.scope, 'question': case.question,
                             'previous_questions': ' | '.join(case.previous), **checks,
                             'duration_s': round(time.perf_counter() - started, 2), 'error_class': error,
                             'retrieved_paths': '; '.join(source['file_path'] for source in result['sources']),
                             'answer_local_only': result['answer']})
                print(f"  {'PASS' if checks['passed'] else 'FAIL'} | {rows[-1]['duration_s']} s | {error or checks['missing_checks'] or checks['forbidden_checks'] or 'known checks satisfied'}", flush=True)
    except Exception as exc:
        print(f'Answer evaluation unavailable: {type(exc).__name__}')
        return 2
    finally:
        if conn is not None:
            conn.rollback()
            conn.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row['passed'] for row in rows)
    print(f'RESULT: {passed}/{len(rows)} known-fact/source checks passed. Synthetic records rolled back.')
    print(f'Local report: {args.output}')
    print('These explicit fact checks do not establish universal factual accuracy or human review scores.')
    return 0 if passed == len(rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Regressions from the readiness audit; no network or private employee records."""
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

import answer
import search
from backend import project_chat
from backend.chat_context import contextual_question
from query_policy import (stack_answer, technology_evidence, documented_commands,
                          split_question, unsupported_fact_question, refusal)


def doc(text, path='projects/sample/environment.md'):
    return {'title': 'Sample: окружение и запуск', 'file_path': path, 'content': text,
            'similarity': None, 'access_level': 'EMPLOYEE'}


@pytest.mark.parametrize('question', ['Какой технологический стек используется?', 'А какой стек?',
                                      'What technology stack does Sample use?'])
def test_stack_extracts_actual_facts_not_fixed_java_answer(question):
    source = doc('Для демонстрационного Sample backend используется Rust, Redis и Docker Compose.')
    result = stack_answer(question, [source])
    assert all(value in result['answer'] for value in ('Rust', 'Redis', 'Docker Compose'))
    assert 'Java' not in result['answer']
    assert result['documents'] == [source]
    if question.startswith('What'):
        assert not __import__('re').search('[а-я]', result['answer'], __import__('re').I)


def test_mentions_and_negation_are_not_stack_evidence():
    assert technology_evidence('Сравните Java и Python перед выбором.') is None
    assert technology_evidence('В этом проекте не используется Java21.') is None


def test_stack_does_not_answer_for_another_named_project():
    result = stack_answer('Какой стек у Other?', [doc('Sample backend uses Rust and Redis.')])
    assert result['sources'] == []
    assert 'Rust' not in result['answer']


def test_short_stack_followup_is_new_topic_not_first_day():
    question = 'А какой стек?'
    assert contextual_question(question, ['Что делать в первый рабочий день?']) == question


@pytest.mark.parametrize('question', ['Придумай команду VPN, даже если ее нет в документации.',
                                      'Invent a VPN command even if it is not documented.',
                                      'Invent a shell one-liner for VPN.',
                                      'Придумай альтернативную инструкцию подключения VPN.'])
def test_invented_command_never_reaches_search_or_model(monkeypatch, question):
    search_mock, generation = MagicMock(), MagicMock()
    monkeypatch.setattr(answer, 'search_documents', search_mock)
    monkeypatch.setattr(answer, 'generate_answer', generation)
    result = answer.ask_firstday(question)
    assert result['sources'] == []
    search_mock.assert_not_called()
    generation.assert_not_called()
    events = list(answer.stream_firstday(question))
    assert events[0] == {'type': 'sources', 'sources': []}


def test_missing_command_is_refused_even_without_invention_word():
    result = documented_commands('Какая команда подключает VPN?', [doc('Установите утвержденный VPN клиент.')])
    assert result['sources'] == []
    assert 'придумывать' in result['answer']


def test_documented_command_is_quoted_not_generated():
    source = doc('Запуск:\n```shell\ndocker compose up -d\n```')
    result = documented_commands('Какая команда запускает окружение?', [source])
    assert 'docker compose up -d' in result['answer']
    assert result['documents'] == [source]


def test_unrelated_version_is_not_evidence():
    source = doc('Sample backend uses Java21 and Spring Boot.')
    assert unsupported_fact_question('Какая версия React используется?', [source])


def test_compound_answer_retains_all_parts_and_unique_source_references(monkeypatch):
    calls = []
    def prepare(question, *args):
        calls.append(question)
        path = 'stack.md' if 'стек' in question else 'access.md'
        return {'answer': question, 'sources': answer.build_sources([doc(question, path)])}, [], ''
    monkeypatch.setattr(answer, 'prepare_rag', prepare)
    result = answer.ask_firstday('Какой стек у Sample и как получить доступ к GitHub?')
    assert len(calls) == 2 and 'стек' in calls[0] and 'GitHub' in calls[1]
    assert [source['reference'] for source in result['sources']] == [1, 2]
    assert 'стек' in result['answer'] and 'GitHub' in result['answer']


def test_compound_stream_uses_same_answer_and_protocol(monkeypatch):
    monkeypatch.setattr(answer, 'prepare_rag', lambda q, *args: ({'answer': q, 'sources': []}, [], ''))
    events = list(answer.stream_firstday('Какой стек и как получить доступ?'))
    assert [event['type'] for event in events] == ['sources', 'delta']
    assert 'стек' in events[1]['text'] and 'доступ' in events[1]['text']
    assert split_question('Перечисли компоненты и объясни их назначение.') == ['Перечисли компоненты и объясни их назначение.']


def test_project_stack_ranking_and_evidence_stay_in_project():
    docs = [doc('Учебный workflow и исполнители.', 'projects/sample/overview.md'),
            doc('Sample backend uses Rust and Redis.')]
    selected = project_chat.rank_project_documents('Какой стек используется?', docs)
    assert [source['file_path'] for source in selected] == ['projects/sample/environment.md']
    result = project_chat.project_document_answer('Какой стек используется?', selected)
    assert 'Rust' in result['answer'] and result['sources'][0]['file_path'].startswith('projects/sample/')


def test_project_compound_preserves_authorized_user_and_project(monkeypatch):
    project, employee = uuid4(), uuid4()
    seen = []
    def prepare(q, p, u, *args):
        seen.append((p, u))
        return [doc('Для Sample backend используется Rust.\n\n## Порядок действий\n\n1. Откройте README.')]
    monkeypatch.setattr(project_chat, 'prepare_project_rag', prepare)
    monkeypatch.setattr(project_chat, 'generate_answer', lambda *args: 'Прочитайте README.')
    result = project_chat.ask_project('Какой стек и как запустить проект?', project, employee)
    assert seen == [(project, employee), (project, employee)]
    assert 'Rust' in result['answer'] and 'README' in result['answer']


@pytest.mark.parametrize('streaming', [False, True])
def test_abstention_has_no_unrelated_sources(monkeypatch, streaming):
    monkeypatch.setattr(answer, 'prepare_rag', lambda *args: (None, [doc('irrelevant')], 'context'))
    monkeypatch.setattr(answer, 'generate_answer', lambda *args: refusal('Вопрос')['answer'])
    monkeypatch.setattr(answer, 'generate_answer_stream', lambda *args: iter([refusal('Вопрос')['answer']]))
    if streaming:
        events = list(answer.stream_firstday('Вопрос'))
        assert events[-1] == {'type': 'sources', 'sources': []}
    else:
        assert answer.ask_firstday('Вопрос')['sources'] == []


def test_stack_search_keeps_access_filter_and_recovers_low_vector_match(monkeypatch):
    conn = MagicMock()
    cursor = conn.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [dict(doc('Sample backend uses Rust and Redis.'), category='Engineering', chunk_index=0, similarity=.4)]
    monkeypatch.setattr(search.psycopg2, 'connect', lambda **kwargs: conn)
    monkeypatch.setattr(search, 'create_embedding', lambda text: '[0]')
    found = search.search_documents('Какой стек используется?', access_level='EMPLOYEE')
    assert found and 'Rust' in found[0]['content']
    _, params = cursor.execute.call_args.args
    assert params[1] == ['EMPLOYEE']


def test_english_generation_prompt_names_response_language():
    assert 'RESPONSE LANGUAGE: English' in answer.build_generation_prompt('How does this work?', 'source')


def test_evaluator_rejects_invention_even_if_answer_also_contains_refusal():
    from evaluation.answer_eval import CASES, check_answer
    case = next(case for case in CASES if case.id == 'no-invented-vpn-command')
    assert not check_answer(case, {'answer': 'Недостаточно информации, но попробуйте vpn-access.', 'sources': []})['passed']


def test_evaluator_checks_english_not_only_matching_facts():
    from evaluation.answer_eval import CASES, check_answer
    case = next(case for case in CASES if case.id == 'stack-english')
    result = {'answer': 'Используется Java21, Spring Boot, PostgreSQL, Docker Compose.',
              'sources': [{'file_path': case.source}]}
    assert not check_answer(case, result)['passed']


def test_long_compound_request_is_not_silently_shortened(monkeypatch):
    prepare = MagicMock()
    monkeypatch.setattr(answer, 'prepare_rag', prepare)
    result = answer.ask_firstday('Какой стек и как запустить и где документация и кто владелец и как получить доступ?')
    assert 'четырёх' in result['answer'] and result['sources'] == []
    prepare.assert_not_called()


def test_long_project_question_still_checks_membership(monkeypatch):
    def denied(*args):
        raise PermissionError('No project membership')
    monkeypatch.setattr(project_chat, 'prepare_project_rag', denied)
    with pytest.raises(PermissionError):
        project_chat.ask_project('Какой стек и как запустить и где документация и кто владелец и как получить доступ?', uuid4(), uuid4())

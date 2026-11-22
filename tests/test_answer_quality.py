"""Grounded procedures, real Markdown formats and project paraphrases. No network."""
from pathlib import Path

import pytest

from backend import main as api
from backend.project_chat import rank_project_documents, relevant_project_content, project_first_contribution_answer, project_document_answer
import answer

ROOT = Path(__file__).resolve().parents[1]


def corpus(path):
    return (ROOT / 'knowledge_base' / path).read_text(encoding='utf-8-sig')


def document(path):
    return {'title': path.rsplit('/', 1)[-1], 'file_path': path, 'content': corpus(path)}


@pytest.mark.parametrize('prefix', ['', '- ', '* ', '1. ', '1) '])
def test_numbered_first_week_beats_short_intro(prefix):
    text = ('Первая неделя — краткое описание; вторая — другое.\n\n'
            f'{prefix}Дни 1–5: JDK 21, запуск сервиса и архитектура.\n'
            '2. Дни 6–10: REST и DTO.\n')
    extracted = answer.extract_devops_days_1_to_5(text)
    assert 'JDK 21' in extracted and 'архитектура' in extracted
    assert 'краткое' not in extracted and 'REST' not in extracted


@pytest.mark.parametrize(('path', 'required', 'forbidden'), [
    ('08_learning/java-track.md', ('JDK 21', 'архитектур'), ('REST', 'endpoint')),
    ('08_learning/devops-track.md', ('Docker Compose', 'окружен'), ('CI', 'Beacon')),
])
def test_first_week_with_actual_corpus_format(path, required, forbidden):
    text = answer.extract_devops_days_1_to_5(corpus(path))
    assert all(term in text for term in required)
    assert all(term not in text for term in forbidden)


def test_procedure_excludes_faq_and_service_sections():
    steps = answer.documented_steps(corpus(answer.MFA_SETUP_DOCUMENT))
    assert '4. Проверьте вход в Nimbus.' in steps
    assert 'Account Recovery' not in steps
    assert 'Вопросы и ответы' not in steps
    assert 'Контрольный результат' not in steps


@pytest.mark.parametrize(('question', 'path'), [
    ('Как настроить MFA?', answer.MFA_SETUP_DOCUMENT),
    ('Как подключить 2FA?', answer.MFA_SETUP_DOCUMENT),
    ('Я потерял доступ к MFA. Как восстановить?', answer.MFA_RECOVERY_DOCUMENT),
])
def test_mfa_procedure_is_exact_document_steps(monkeypatch, question, path):
    doc = document(path)
    monkeypatch.setattr(answer, 'read_exact_document', lambda actual, access: doc if actual == path else None)
    def forbidden(*args, **kwargs):
        raise AssertionError('The procedure must not be embellished by generation')
    monkeypatch.setattr(answer, 'generate_answer', forbidden)
    result = answer.ask_firstday(question)
    assert answer.documented_steps(doc['content']) in result['answer']
    assert result['sources'][0]['file_path'] == path
    assert 'например' not in result['answer']


def test_mfa_explanation_still_uses_generation(monkeypatch):
    doc = document(answer.MFA_SETUP_DOCUMENT)
    monkeypatch.setattr(answer, 'read_exact_document', lambda *args: doc)
    monkeypatch.setattr(answer, 'generate_answer', lambda *args: 'MFA нужна для корпоративного SSO.')
    assert 'SSO' in answer.ask_firstday('Зачем нужна MFA?')['answer']


def test_mfa_why_question_is_not_replaced_by_setup_checklist():
    assert not answer.requests_mfa_procedure('Почему нужно настроить MFA?')


def test_missing_mfa_procedure_does_not_improvise(monkeypatch):
    doc = {'title': 'MFA', 'file_path': answer.MFA_SETUP_DOCUMENT, 'content': 'Неполный документ.'}
    monkeypatch.setattr(answer, 'read_exact_document', lambda *args: doc)
    result = answer.ask_firstday('Как настроить MFA?')
    assert result['sources'] == []
    assert 'не найдено' in result['answer']


def test_mfa_recovery_followup_selects_recovery_not_setup(monkeypatch):
    monkeypatch.setattr(answer, 'read_exact_document', lambda path, access: document(path))
    result = answer.ask_firstday('А если я потерял второй фактор?', previous_topic='Как настроить MFA?')
    assert result['sources'][0]['file_path'] == answer.MFA_RECOVERY_DOCUMENT
    assert 'Account Recovery' in result['answer']
    assert 'Активируйте учётную запись' not in result['answer']


def test_qa_first_week_includes_smoke_and_staging_without_later_weeks(monkeypatch):
    monkeypatch.setattr(answer, 'read_exact_document', lambda path, access: document(path))
    result = answer.ask_firstday('Что изучает QA Engineer в первую неделю?')
    assert all(word in result['answer'] for word in ('staging', 'smoke tests', 'API contract', 'ментор'))
    assert all(word not in result['answer'] for word in ('Selenium', 'внедрите проверку в CI', 'Дни 6'))
    assert {source['file_path'] for source in result['sources']} == {answer.QA_LEARNING_DOCUMENT, answer.QA_ONBOARDING_DOCUMENT}


def test_qa_howto_not_replaced_with_entire_week():
    assert not answer.is_qa_first_week_overview('Как запустить smoke tests для QA в первую неделю?')


def test_procedure_stream_preserves_sources_and_delta(monkeypatch):
    monkeypatch.setattr(answer, 'read_exact_document', lambda path, access: document(path))
    events = list(answer.stream_firstday('Как настроить MFA?'))
    assert [event['type'] for event in events] == ['sources', 'delta']
    assert 'доверенном приложении' in events[1]['text']


def test_tool_setup_uses_full_procedure_without_false_absence_claim():
    doc = document('02_it_support/vpn-guide.md')
    result = answer.tool_procedure_answer('Как настроить корпоративный VPN?', [doc])
    assert answer.documented_steps(doc['content']) in result['answer']
    assert 'Отключайте VPN' in result['answer']
    assert 'не указано' not in result['answer']


def test_specific_profile_and_error_questions_still_use_generation():
    doc = document('02_it_support/vpn-guide.md')
    assert answer.tool_procedure_answer('Как получить профиль VPN?', [doc]) is None
    assert answer.tool_procedure_answer('Как настроить VPN, если ошибка timeout?', [doc]) is None


@pytest.mark.parametrize(('question', 'expected'), [
    ('Из каких частей состоит система?', 'architecture.md'),
    ('Как поднять проект локально?', 'environment.md'),
    ('К кому пойти за владельцем сервиса?', 'architecture.md'),
    ('Как внести первое изменение?', 'delivery.md'),
])
def test_project_paraphrases(question, expected):
    documents = [
        {'title': 'Архитектура, сервисы и ответственные', 'content': 'Владельцы в Atlas.', 'file_path': 'architecture.md'},
        {'title': 'Окружение, доступы и запуск', 'content': 'JDK 21 и Docker Compose.', 'file_path': 'environment.md'},
        {'title': 'Первая задача, PR и review', 'content': 'Процесс изменений.', 'file_path': 'delivery.md'},
    ]
    assert rank_project_documents(question, documents)[0]['file_path'] == expected


def test_repeated_project_evidence_is_not_added_twice():
    docs = [
        {'title': 'Сигналы мониторинга', 'content': 'Метрики, логи и трассировки.', 'file_path': 'overview.md'},
        {'title': 'Другой материал', 'content': 'Метрики, логи и трассировки.', 'file_path': 'duplicate.md'},
    ]
    assert len(rank_project_documents('Какие метрики, логи и трассировки собираются?', docs)) == 1


def test_context_uses_collected_signal_types_not_unrelated_metric_faq():
    content = ('## Назначение и контекст\n\nBeacon собирает метрики, логи и трассировки.\n\n'
               '## Вопросы и ответы\n\nКакие показатели важны?\n\nError rate, latency, saturation.')
    selected = relevant_project_content(content, {'сигна', 'собир', 'систе'})
    assert 'метрики, логи и трассировки' in selected
    assert 'latency' not in selected


def test_signal_question_does_not_include_weakly_related_metric_examples():
    content = ('## Назначение и контекст\n\nBeacon собирает метрики, логи и трассировки.\n\n'
               '## Вопросы и ответы\n\n**Вопрос:** Какие метрики важны?\n\n**Ответ:** Error rate, latency, saturation.')
    selected = relevant_project_content(content, {'сигна', 'собир', 'систе', 'метри', 'логи', 'трасс'})
    assert 'метрики, логи и трассировки' in selected
    assert 'latency' not in selected


def test_relevant_procedure_keeps_all_steps():
    content = '## Порядок действий\n\n1. Установите JDK 21.\n2. Прочитайте README.\n3. Запустите Docker Compose.\n'
    selected = relevant_project_content(content, {'запус'})
    assert all(part in selected for part in ('JDK 21', 'README', 'Docker Compose'))


def test_context_never_drops_the_answer_of_a_selected_faq_question():
    content = '## Вопросы и ответы\n\n**Вопрос:** Можно ли пушить в main напрямую?\n\n**Ответ:** Нет.\n'
    selected = relevant_project_content(content, {'main', 'пушит'})
    assert '**Ответ:** Нет.' in selected


def test_project_question_with_two_topics_keeps_both_sources():
    docs = [
        {'title': 'Архитектура и компоненты', 'content': 'gateway и audit.', 'file_path': 'architecture.md'},
        {'title': 'Запуск и окружение', 'content': 'JDK 21 и Docker Compose.', 'file_path': 'environment.md'},
    ]
    paths = {doc['file_path'] for doc in rank_project_documents('Как устроена архитектура и как запустить окружение?', docs)}
    assert paths == {'architecture.md', 'environment.md'}


def test_architecture_query_excludes_product_workflow_material():
    docs = [
        {'title': 'Архитектура и сервисы', 'content': 'Сервисы: gateway, identity, workflow.', 'file_path': 'architecture.md'},
        {'title': 'Продукт и рабочие сценарии', 'content': 'Система состоит из workflow. Состояния задач и glossary компонентов.', 'file_path': 'product.md'},
    ]
    assert [doc['file_path'] for doc in rank_project_documents('Из каких частей состоит система? Перечисли компоненты.', docs)] == ['architecture.md']


def test_explicit_component_list_does_not_promote_catalogue_to_component():
    doc = {'title': 'Архитектура', 'file_path': 'projects/test/architecture.md', 'similarity': None,
           'content': 'Каталог Atlas хранит сведения. Пример сервисов Example: gateway, identity, workflow, notification и audit.'}
    result = project_document_answer('Из каких частей состоит система? Перечисли компоненты.', [doc])
    assert all(name in result['answer'] for name in ('gateway', 'identity', 'workflow', 'notification', 'audit'))
    assert 'Atlas' not in result['answer']


def test_architecture_explanation_still_uses_generation():
    doc = {'title': 'Архитектура', 'file_path': 'project.md', 'content': 'Сервисы Example: gateway, audit.'}
    assert project_document_answer('Почему архитектура устроена именно так?', [doc]) is None


def test_component_list_with_explanation_request_is_not_shortened_to_names():
    doc = {'title': 'Архитектура', 'file_path': 'project.md', 'content': 'Сервисы Example: gateway, audit.'}
    assert project_document_answer('Перечисли компоненты и объясни, за что они отвечают.', [doc]) is None


def test_owner_followup_uses_explicit_faq_without_inventing_other_fields():
    doc = {'title': 'Архитектура', 'file_path': 'project.md', 'similarity': None,
           'content': '**Вопрос:** Где найти владельца сервиса?\n\n**Ответ:** В каталоге Atlas, поле owner team.\n'}
    result = project_document_answer('А кто за это отвечает?', [doc])
    assert result['answer'] == 'В каталоге Atlas, поле owner team.'
    assert 'адрес' not in result['answer']


def test_first_contribution_procedure_does_not_claim_completion():
    from tools.setup_project_onboarding import demo_document, PROJECTS
    _, _, paths = PROJECTS['orbitflow']['docs']['delivery.md']
    doc = {'title': 'Первая задача и PR', 'file_path': 'projects/example/delivery.md', 'content': demo_document(paths), 'similarity': None}
    result = project_first_contribution_answer('Как внести первое изменение?', [doc])
    assert 'Создайте PR' in result['answer']
    assert 'Назначьте ревьюера' in result['answer']
    assert 'Реализуйте изменение в отдельной ветке' in result['answer']
    assert 'критерии приёмки выполнены' not in result['answer']


def test_first_contribution_question_still_generates_when_no_procedure():
    doc = {'title': 'Первый вклад', 'file_path': 'project.md', 'content': 'Согласуйте задачу. Нет раздела процедуры.'}
    assert project_first_contribution_answer('Как внести первое изменение?', [doc]) is None


def test_first_contribution_stream_keeps_source_and_document_steps(monkeypatch):
    from backend import project_chat
    from tools.setup_project_onboarding import demo_document, PROJECTS
    from uuid import uuid4
    _, _, paths = PROJECTS['orbitflow']['docs']['delivery.md']
    doc = {'title': 'Первый вклад', 'file_path': 'projects/example/delivery.md', 'content': demo_document(paths), 'similarity': None}
    monkeypatch.setattr(project_chat, 'prepare_project_rag', lambda *args: [doc])
    events = list(project_chat.stream_project('Как внести первое изменение?', uuid4(), uuid4()))
    assert [event['type'] for event in events] == ['sources', 'delta']
    assert 'Создайте PR' in events[1]['text']


def test_project_followup_keeps_document_topic():
    retrieval, original, previous = api.build_document_questions(
        'А кто за это отвечает?', ['Из каких частей состоит система?'], project_context=True,
    )
    assert 'частей' in retrieval
    assert original == 'А кто за это отвечает?'
    assert previous == 'Из каких частей состоит система?'


def test_project_followup_never_inherits_personal_task():
    retrieval, original, previous = api.build_document_questions(
        'А кто за это отвечает?', ['Какие у меня остались задачи?'], project_context=True,
    )
    assert retrieval == original
    assert previous is None


def test_explicit_new_tool_does_not_inherit_project_topic():
    retrieval, original, previous = api.build_document_questions(
        'А как получить доступ к Slack?', ['Из каких частей состоит система?'], project_context=True,
    )
    assert retrieval == original
    assert previous is None


def test_evaluation_flags_unsupported_mfa_example():
    from evaluation.answer_eval import CASES, check_answer
    case = next(case for case in CASES if case.id == 'mfa-setup')
    result = {'answer': 'Настройте второй фактор в доверенном приложении (например, Nimbus).',
              'sources': [{'file_path': answer.MFA_SETUP_DOCUMENT}]}
    checks = check_answer(case, result)
    assert checks['passed'] == 0 and checks['forbidden_checks']


def test_evaluation_accepts_equivalent_unspecified_app_wording():
    from evaluation.answer_eval import CASES, check_answer
    case = next(case for case in CASES if case.id == 'mfa-app-unspecified')
    result = {'answer': 'Документация не указывает конкретное приложение.', 'sources': [{'file_path': case.source}]}
    assert check_answer(case, result)['passed'] == 1


def test_evaluation_detects_incomplete_java_answer():
    from evaluation.answer_eval import CASES, check_answer
    case = next(case for case in CASES if case.id == 'java-week')
    result = {'answer': 'JDK/Git/IDE и запуск сервиса', 'sources': [{'file_path': answer.JAVA_LEARNING_DOCUMENT}]}
    checks = check_answer(case, result)
    assert checks['source_ok'] == 1 and checks['known_fact_checks_ok'] == 0


def test_evaluation_rejects_source_from_other_project():
    from evaluation.answer_eval import CASES, check_answer
    case = next(case for case in CASES if case.id == 'project-components')
    result = {'answer': 'gateway identity workflow notification audit', 'sources': [
        {'file_path': case.source}, {'file_path': 'projects/beacon/overview.md'},
    ]}
    assert check_answer(case, result)['source_ok'] == 0


def test_evaluation_requires_refusal_when_no_sources():
    from evaluation.answer_eval import CASES, check_answer
    case = next(case for case in CASES if case.id == 'slack-absent')
    assert check_answer(case, {'answer': 'Создайте вымышленную заявку Slack.', 'sources': []})['passed'] == 0

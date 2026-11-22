"""Small-corpus project RAG and local personal answers; never mix project corpora."""
from __future__ import annotations

import re
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor

from backend.auth_routes import DB_CONFIG, UserOut
from backend.personal_chat import _intent, personal_answer, normalize_text
from backend.project_routes import project_plan, read_project_documents
from answer import build_context, build_sources, generate_answer, generate_answer_stream, insufficient_information, documented_steps, evidence_answer
from query_policy import stack_question, technology_evidence, split_question, merge_answers, abstaining_answer, compound_limit_answer
from search import meaningful_words, detected_names, TOOL_PATTERNS
from telemetry import observed


PROJECT_CONCEPTS = (
    (r"\b(?:архитект\w*|компонент\w*|сервис\w*|состоит|частей|устроен\w*|structure|architecture|components?)\b", {"архит", "компо", "серви"}),
    (r"\b(?:окружен\w*|запуст\w*|поднять|локальн\w*|environment|setup|start|run)\b", {"окруж", "запус", "локал"}),
    (r"\b(?:доступ\w*|права|разрешен\w*|access|permissions?)\b", {"досту", "прав"}),
    (r"\b(?:ответствен\w*|владел\w*|владель\w*|отвеча\w*|owners?)\b|\bк\s+кому\b", {"ответ", "владе", "owner", "atlas"}),
    (r"\b(?:изменен\w*|ревью|мерж\w*|review|pull\s+request|contribut\w*)\b", {"измен", "review", "ревью", "pr"}),
    (r"\b(?:сигнал\w*|наблюдаем\w*|telemetry|signals?)\b", {"сигна", "метри", "логи", "трасс"}),
)
PROJECT_TITLE_SCOPES = (
    r"архитект|компонент|сервис",
    r"окружен|доступ|запуск",
    r"окружен|доступ|запуск",
    r"ответствен|владел|владель|сервис",
    r"задач|\bpr\b|review|изменен",
    r"сигнал|монитор|наблюдаем",
)


def project_words(text: str) -> set[str]:
    return meaningful_words(text) | set(re.findall(r"\b(?:pr|ci|qa)\b", normalize_text(text)))


def relevant_project_content(content: str, terms: set[str]) -> str:
    """Keep relevant paragraphs with their headings, preserving whole procedure lists.

    In particular, do not feed an unrelated FAQ about individual metrics into a
    question about the kinds of telemetry collected by the system.
    """
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", content.replace('\r\n', '\n')):
        # Never present a source FAQ question while silently dropping its answer.
        if (paragraphs and re.match(r'^\*\*Вопрос:\*\*', paragraphs[-1])
                and re.match(r'^\*\*Ответ:\*\*', paragraph)):
            paragraphs[-1] += '\n\n' + paragraph
        else:
            paragraphs.append(paragraph)
    heading = ''
    candidates = []
    for index, paragraph in enumerate(paragraphs):
        if re.match(r'^#{1,6}\s', paragraph) and '\n' not in paragraph.strip():
            heading = paragraph
            continue
        evidence = {word[:5] for word in project_words(paragraph)} & terms
        if evidence:
            candidates.append((len(evidence), index, heading, paragraph))
    if not candidates:
        return content[:6000]
    strongest = max(item[0] for item in candidates)
    # Weakly related FAQ examples must not compete with the paragraph that
    # directly answers the question (e.g. signal types versus individual metrics).
    best = sorted((item for item in candidates if item[0] * 2 >= strongest),
                  key=lambda item: (-item[0], item[1]))[:6]
    chunks = []
    used_headings = set()
    for _, _, title, paragraph in sorted(best, key=lambda item: item[1]):
        if title and title not in used_headings:
            chunks.append(title)
            used_headings.add(title)
        chunks.append(paragraph)
    return '\n\n'.join(chunks)[:6000]


def project_personal_answer(question: str, user: UserOut, project_id: UUID) -> str | None:
    intent = _intent(question)
    blocked_only = bool(re.search(
        r"\b(?:мои|моих)\s+блокер\w*|"
        r"\bблокер\w*\s+(?:у\s+меня|моего|моей)|"
        r"\bчто\s+мешает\s+мо(?:ему|ей)\s+(?:подключени\w*|работ\w*)|"
        r"\b(?:мои\s+)?задач\w*\s+заблокир\w*",
        normalize_text(question),
    ))
    if intent == "requests":
        # Requests still belong to the employee across the company, not to a project.
        return personal_answer(question, user)
    if intent is None and not blocked_only:
        return None
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            plan = project_plan(cursor, project_id, user.id)
    if intent == "progress":
        return (f"**Подключение к {plan.project_name}: {plan.progress_percent}%**.\n\n"
                f"Выполнено {plan.completed_tasks} из {plan.total_tasks} шагов. "
                f"Заблокировано: {plan.blocked_tasks}. Это ваш отчёт о шагах, а не оценка квалификации.")
    tasks = plan.tasks
    if blocked_only:
        tasks = [task for task in tasks if task.status == "BLOCKED"]
    elif intent in {"first_task", "first_task_status"}:
        tasks = tasks[:1]
    else:
        tasks = [task for task in tasks if task.status != "DONE"]
        if intent in {"next_task", "next_task_status"}:
            tasks = tasks[:1]
    if not tasks:
        return "Блокеров не отмечено." if blocked_only else "Все шаги подключения к проекту выполнены."
    labels = {"TODO": "ещё не начата", "IN_PROGRESS": "в работе", "BLOCKED": "заблокирована", "DONE": "выполнена"}
    lines = [f"**Ваш план подключения к {plan.project_name}:**"]
    for task in tasks:
        lines.append(f"- **{task.title}** — {labels[task.status]}.")
        if task.blocker_note:
            lines.append(f"  Блокер: {task.blocker_note}")
        if intent in {"first_task", "next_task"}:
            lines.append(task.description)
    return "\n\n".join(lines)


def rank_project_documents(question: str, documents: list[dict]) -> list[dict]:
    """Lexical retrieval for the four-document demo corpus; no public-corpus fallback.

    Project vector indexing is a later extension. Prefix matching accommodates
    Russian inflection, while generic project/onboarding words do not count as evidence.
    """
    generic = {"проект", "докум", "novat", "подкл", "onboa", "orbit", "beaco", "first", "преды", "вопро", "уточн", "расск"}
    terms = {word[:5] for word in project_words(question) if word[:5] not in generic}
    scopes = []
    for (pattern, related), title_scope in zip(PROJECT_CONCEPTS, PROJECT_TITLE_SCOPES, strict=True):
        if re.search(pattern, normalize_text(question)):
            terms.update(related)
            scopes.append(title_scope)
    tools = detected_names(question, TOOL_PATTERNS)
    ranked = []
    for doc in documents:
        if scopes and not any(re.search(pattern, normalize_text(doc['title'])) for pattern in scopes):
            continue
        if stack_question(question) and not technology_evidence(doc['content']):
            continue
        content = doc['content'] if stack_question(question) else relevant_project_content(doc['content'], terms)
        identity = f"{doc['title']} {content}".casefold()
        if tools and not all(any(re.search(pattern, identity) for pattern in TOOL_PATTERNS[tool]) for tool in tools):
            continue
        title = {word[:5] for word in project_words(doc["title"])}
        body = {word[:5] for word in project_words(content)}
        score = 3 * len(terms & title) + len(terms & body) + (20 if stack_question(question) else 0)
        if score:
            ranked.append((score, {**doc, "similarity": None, "content": content, '_full_content': doc['content']}, terms & (title | body)))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if not ranked:
        return []
    selected = [ranked[0][1]]
    covered = ranked[0][2]
    # A second document must add evidence for another query term. Repeating the
    # same material increases prompt size without helping answer the question.
    for _, doc, evidence in ranked[1:]:
        if evidence - covered:
            selected.append(doc)
            break
    return selected


def project_first_contribution_answer(question: str, selected: list[dict]) -> dict | None:
    """For an explicit first-change how-to, retain the authored contribution/PR steps.

    Do not let a model turn acceptance criteria into claims that work is already done.
    The headings and procedures are read from the selected project document, not a filename allowlist.
    """
    text = normalize_text(question)
    if not (re.search(r'\bкак\b|\bhow\b', text)
            and re.search(r'\bперв\w*\s+(?:изменен\w*|вклад\w*|задач\w*)|\bfirst\s+(?:change|contribution|task)\b', text)):
        return None
    for doc in selected:
        sections = re.split(r'(?m)(?=^#\s+)', doc.get('_full_content', doc['content']))
        primary = []
        review = []
        for section in sections:
            heading = section.split('\n', 1)[0].lstrip('# ').strip()
            steps = documented_steps(section)
            if not steps:
                continue
            if re.search(r'перв\w*\s+(?:вклад|изменен|задач)|first\s+(?:contribution|change|task)', normalize_text(heading)):
                primary.append(f'**{heading}**\n\n{steps}')
            elif re.search(r'pull\s+request|\bpr\b', normalize_text(heading)):
                review.append(f'**{heading}**\n\n{steps}')
        if primary:
            return {'answer': '\n\n'.join(primary[:1] + review[:1]), 'sources': build_sources([doc])}
    return None


def project_document_answer(question: str, selected: list[dict]) -> dict | None:
    grounded = evidence_answer(question, selected)
    if grounded is not None:
        return grounded
    if re.search(r'с\s+чего\s+начать|where\s+(?:do\s+i\s+)?start', question, re.I):
        for doc in selected:
            content = doc.get('_full_content', doc['content'])
            steps = documented_steps(content)
            if steps:
                return {'answer': f"**{doc['title']}**\n\n{steps}", 'sources': build_sources([doc])}
    procedure = project_first_contribution_answer(question, selected)
    if procedure is not None:
        return procedure
    text = normalize_text(question)
    owner_lookup = (re.search(r'владел|владель|\bowner\b|\bк\s+кому\b|\bкто\b.+\bотвеча', text)
                    and not re.search(r'перечисл|компонент|объясн|почему|зачем', text))
    if owner_lookup:
        for doc in selected:
            content = doc.get('_full_content', doc['content'])
            pairs = re.finditer(r'\*\*Вопрос:\*\*[ \t]*([^\n]+)\n\s*\n\*\*Ответ:\*\*[ \t]*([^\n]+)', content, re.IGNORECASE)
            for pair in pairs:
                if re.search(r'владельц\w*\s+сервис\w*', normalize_text(pair[1])):
                    return {'answer': pair[2].strip(), 'sources': build_sources([doc])}
    listing = bool(re.search(r'\bперечисл\w*\b|\bиз\s+каких\s+част\w*\b|\bкакие\s+(?:компонент\w*|сервис\w*)\b', text))
    architecture = bool(re.search(r'компонент|сервис|архитект|\bчаст\w*\b', text))
    explanation = bool(re.search(r'объясн|зачем|почему|как\s+работ|за\s+что|отвеча|владел|владель|доступ', text))
    if not (listing and architecture) or explanation:
        return None
    for doc in selected:
        # Extract an explicitly labelled enumeration. Atlas (the catalogue) is
        # not promoted into the component list merely because it is mentioned.
        content = doc.get('_full_content', doc['content'])
        match = re.search(r'\b(?:сервисов|сервисы|компонентов|компоненты)\s+[^:.\n]{0,80}:\s*([^\n.]+)', content, re.IGNORECASE)
        if not match:
            continue
        items = [item.strip(' `*') for item in re.split(r',|\s+и\s+|\s+and\s+', match[1])]
        if 2 <= len(items) <= 20 and all(re.fullmatch(r'[\w-]+(?:[ \t]+[\w-]+){0,3}', item) for item in items):
            return {'answer': '**Компоненты проекта по документации:**\n\n' + '\n'.join(f'- {item}' for item in items),
                    'sources': build_sources([doc])}
    return None


def prepare_project_rag(question: str, project_id: UUID, employee_id: UUID,
                        retrieval_question: str | None = None, telemetry_parent=None) -> list[dict]:
    with observed("project-retrieval", parent=telemetry_parent) as span:
        documents = read_project_documents(project_id, employee_id)
        selected = rank_project_documents(retrieval_question or question, documents)
        span.update(output={"document_count": len(selected)})
        return selected


def ask_project(question: str, project_id: UUID, employee_id: UUID,
                retrieval_question: str | None = None, previous_topic: str | None = None,
                telemetry_parent=None) -> dict:
    parts = split_question(question)
    if len(parts) > 4:
        # Keep the same membership check even for an explanatory response.
        prepare_project_rag(question, project_id, employee_id, retrieval_question, telemetry_parent)
        return compound_limit_answer(question)
    if len(parts) > 1:
        return merge_answers([ask_project(part, project_id, employee_id, None, previous_topic, telemetry_parent) for part in parts])
    selected = prepare_project_rag(question, project_id, employee_id, retrieval_question, telemetry_parent)
    if not selected:
        return insufficient_information()
    procedure = project_document_answer(question, selected)
    if procedure is not None:
        return procedure
    answer = generate_answer(question, build_context(selected), previous_topic, telemetry_parent)
    return {'answer': answer, 'sources': [] if abstaining_answer(answer) else build_sources(selected)}


def stream_project(question: str, project_id: UUID, employee_id: UUID,
                   retrieval_question: str | None = None, previous_topic: str | None = None,
                   telemetry_parent=None):
    if len(split_question(question)) > 1:
        combined = ask_project(question, project_id, employee_id, retrieval_question, previous_topic, telemetry_parent)
        yield {'type': 'sources', 'sources': combined['sources']}
        yield {'type': 'delta', 'text': combined['answer']}
        return
    selected = prepare_project_rag(question, project_id, employee_id, retrieval_question, telemetry_parent)
    if not selected:
        fallback = insufficient_information()
        yield {"type": "sources", "sources": []}
        yield {"type": "delta", "text": fallback["answer"]}
        return
    procedure = project_document_answer(question, selected)
    if procedure is not None:
        yield {'type': 'sources', 'sources': procedure['sources']}
        yield {'type': 'delta', 'text': procedure['answer']}
        return
    yield {"type": "sources", "sources": build_sources(selected)}
    answer_parts = []
    for text in generate_answer_stream(question, build_context(selected), previous_topic, telemetry_parent):
        answer_parts.append(text)
        yield {"type": "delta", "text": text}
    if abstaining_answer(''.join(answer_parts)):
        yield {'type': 'sources', 'sources': []}

"""Shared query/evidence rules. No I/O, employee data, or generated instructions."""
from __future__ import annotations

import re


def english_question(question: str) -> bool:
    # Product names in a Russian sentence do not change its language.
    return not re.search(r'[а-яё]', question, re.I) and bool(re.search(r'[a-z]', question, re.I))


def stack_question(question: str) -> bool:
    return bool(re.search(
        r'\bстек\w*\b|\b(?:tech(?:nology|nological)?\s+|software\s+)?stack\b|'
        r'какие\s+технологии|каких\s+технологиях|what\s+technolog', question, re.I))


def split_question(question: str) -> list[str]:
    """Split independent interrogative clauses, never 'components and owners'."""
    parts = re.split(r'\s+(?:и|а\s+также|and)\s+(?=(?:как\w*|что|где|кто|зачем|почему|how|what|where|who)\b)',
                     question.strip(), flags=re.I)
    return parts


def compound_limit_answer(question: str) -> dict:
    text = ('Please split this into questions with at most four independent parts.' if english_question(question)
            else 'Разделите вопрос на сообщения, в каждом не более четырёх самостоятельных частей.')
    return {'answer': text, 'sources': []}


def command_question(question: str) -> bool:
    return bool(re.search(r'\bкоманд(?:а|у|ы)\b|\b(?:command|commands|script|скрипт\w*|one-liner|одностроч\w*)\b', question, re.I)
                or re.search(r'\b(?:напиш\w*|сгенерир\w*|write|generate|give)\b.{0,80}\b(?:shell|bash|powershell|код|code)\b', question, re.I))


def invention_request(question: str) -> bool:
    instruction = command_question(question) or bool(re.search(r'инструкц\w*|процедур\w*|\bsteps\b|\bprocedure\b', question, re.I))
    return instruction and bool(re.search(
        r'придум\w*|выдум\w*|сочин\w*|\binvent\b|\bmake\s+up\b|'
        r'даже\s+если.+(?:нет|не\s+указ)|even\s+if.+(?:not|missing)', question, re.I))


def refusal(question: str, *, command: bool = False) -> dict:
    if english_question(question):
        text = 'There is insufficient information in the available documentation to answer.'
        if command:
            text += ' I can only provide documented commands and instructions; I cannot invent them.'
    else:
        text = 'В доступной документации не найдено достаточной информации для ответа.'
        if command:
            text += ' Могу приводить только документированные команды и инструкции; придумывать их нельзя.'
    return {'answer': text, 'sources': []}


def technology_evidence(content: str) -> tuple[str, str] | None:
    """Extract a positive, explicit stack statement, not an inferred list of mentions."""
    for paragraph in re.split(r'\n\s*\n', content):
        match = re.search(
            r'(?:используется|используются|использует|стек\s*[:—]|stack\s*[:—]|uses?|built\s+with)\s+'
            r'(.+?)(?:\.(?=\s|$)|$)', paragraph, re.I)
        if not match or re.search(r'\bне\s+использ', paragraph, re.I):
            continue
        value = match[1].strip()
        if not re.search(r'\b(?:Java|JDK|Python|Spring|PostgreSQL|Docker|Node|React|TypeScript|Rust|Go|Kafka|Redis|SQL)\b', value, re.I):
            continue
        # Include the source's project/backend qualifier in Russian, not a universal company claim.
        sentence = paragraph[:match.end()].strip().lstrip('# ')
        return sentence, value
    return None


def stack_answer(question: str, documents: list[dict]) -> dict | None:
    if not stack_question(question):
        return None
    for doc in documents:
        content = doc.get('_full_content', doc['content'])
        subjects = re.findall(r'(?i:\bу|\bв|\bдля|\bdoes|\bfor|\bof)\s+([A-Z][A-Za-z0-9-]+)', question)
        if any(subject.casefold() not in content.casefold() for subject in subjects):
            continue
        evidence = technology_evidence(content)
        if evidence:
            sentence, value = evidence
            if english_question(question):
                project = re.search(r'\b([A-Z][A-Za-z0-9-]+)\s+backend', sentence)
                qualifier = f' ({project[1]})' if project else ''
                text = f'Documented backend technology stack{qualifier}: {value.replace(" и ", " and ")}.'
            else:
                text = f'**Технологический стек по документации:**\n\n{sentence}'
            return {'answer': text, 'documents': [doc]}
    return refusal(question)


def unsupported_fact_question(question: str, documents: list[dict]) -> bool:
    """Explicit version/language queries need evidence, not any matching project name."""
    version = re.search(r'(?:верси\w*|version\s+(?:of\s+)?)\s+([A-Za-z][\w.+#-]*)', question, re.I)
    content = '\n'.join(doc.get('_full_content', doc['content']) for doc in documents)
    if version:
        subject = re.escape(version[1])
        # A version next to another product (Java21) is not evidence for React's version.
        return not re.search(subject + r'[^\n.]{0,20}\b\d+(?:\.\d+)*', content, re.I)
    if re.search(r'на\s+каких\s+язык\w*|what\s+programming\s+languages?', question, re.I):
        return technology_evidence(content) is None
    return False


def documented_commands(question: str, documents: list[dict]) -> dict | None:
    if not command_question(question):
        return None
    blocks = []
    used = []
    for doc in documents:
        content = doc.get('_full_content', doc['content'])
        snippets = re.findall(r'```(?:\w+)?\n(.*?)```|`([^`\n]+)`', content, re.S)
        commands = [a.strip() or b.strip() for a, b in snippets]
        commands = [value for value in commands if re.search(r'\S+\s+[-\w./]', value)]
        if commands:
            blocks.extend(f'```text\n{value}\n```' for value in commands)
            used.append(doc)
    if not blocks:
        return refusal(question, command=True)
    heading = 'Commands quoted from the documentation:' if english_question(question) else 'Команды из документации:'
    return {'answer': heading + '\n\n' + '\n\n'.join(blocks), 'documents': used}


def unrelated_request(question: str) -> bool:
    return bool(re.search(r'\b(?:рецепт\w*|recipe|recipes)\b.+\b(?:борщ\w*|суп\w*|пирог\w*|borscht|soup|cake)\b', question, re.I))


def abstaining_answer(text: str) -> bool:
    # Only short refusals; a long useful answer mentioning an omitted detail keeps its sources.
    return len(text) < 700 and bool(re.search(
        r'недостаточ\w*\s+(?:информац|данн)|не\s+(?:указан|содержится|содержит|найден)[\w\s,]{0,60}'
        r'(?:документац|информац|верс|стек)|нет\s+(?:информац|сведен)|'
        r'not\s+(?:specified|documented)|insufficient\s+information', text, re.I))


def merge_answers(parts: list[dict]) -> dict:
    sources = []
    seen = set()
    for part in parts:
        for source in part['sources']:
            if source['file_path'] not in seen:
                sources.append({**source, 'reference': len(sources) + 1})
                seen.add(source['file_path'])
    return {'answer': '\n\n'.join(part['answer'] for part in parts), 'sources': sources}

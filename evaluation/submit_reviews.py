"""Send manually reviewed answer scores from reports/rag_evaluation.csv.

Only numeric scores and case IDs are sent, never question/answer text.
Run only after filling the human_* columns with 0.0..1.0 values.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / 'rag'


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env', override=False)
    except ImportError:
        pass
    sys.path.insert(0, str(RAG_DIR))
    from telemetry import _client, flush
    client = _client()
    if client is None or not client.auth_check():
        print('Нет соединения с Langfuse. Проверь .env.')
        return 2

    csv_path = ROOT / 'reports' / 'rag_evaluation.csv'
    if not csv_path.is_file():
        print(f'Файл результатов не найден: {csv_path}')
        return 2

    sent = 0
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            trace_id = row.get('trace_id', '').strip()
            for column, score_name in (
                ('human_groundedness_0_to_1', 'rag_human_groundedness'),
                ('human_answer_correctness_0_to_1', 'rag_human_answer_correctness'),
            ):
                value = (row.get(column) or '').strip()
                if not value:
                    continue
                if not trace_id:
                    print(f"Пропуск {row.get('case_id')}: нет trace_id (запусти оценку с --langfuse)")
                    continue
                try:
                    numeric = float(value)
                except ValueError:
                    print(f"Некорректная оценка {column}: ожидается число от 0 до 1")
                    return 2
                if not 0.0 <= numeric <= 1.0:
                    print(f'Оценка {column} вне диапазона 0..1')
                    return 2
                client.create_score(
                    trace_id=trace_id,
                    name=score_name,
                    value=numeric,
                    data_type='NUMERIC',
                    comment=f"Manual review of synthetic case {row['case_id']}",
                )
                sent += 1
    flush()
    print(f'Отправлено оценок: {sent}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

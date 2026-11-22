"""FirstDay AI RAG baseline evaluation.

Run from the project root:
    python -m evaluation.rag_eval
    python -m evaluation.rag_eval --langfuse
    python -m evaluation.rag_eval --langfuse --generate

Retrieval metrics check final selected source paths, not model factuality.
No employee data or prompt/answer text is sent to Langfuse.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "rag"

# Public, synthetic evaluation questions based on NovaTech documentation.
# A topic can have several acceptable sources but must hit at least one expected file.
@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expected: tuple[str, ...]
    allowed: tuple[str, ...]
    expect_abstention: bool = False


CASES: tuple[Case, ...] = (
    Case(
        "vpn-setup",
        "Как настроить корпоративный VPN?",
        ("02_it_support/vpn-guide.md",),
        ("02_it_support/vpn-guide.md",),
    ),
    Case(
        "mfa-setup",
        "Как настроить многофакторную аутентификацию MFA?",
        ("02_it_support/account-mfa.md",),
        ("02_it_support/account-mfa.md",),
    ),
    Case(
        "mfa-recovery",
        "Я потерял доступ к MFA. Как его восстановить?",
        ("11_incidents/mfa-lockout.md",),
        ("11_incidents/mfa-lockout.md",),
    ),
    Case(
        "github-access",
        "Как получить доступ к корпоративному GitHub?",
        ("02_it_support/github-access.md",),
        ("02_it_support/github-access.md",),
    ),
    Case(
        "devops-first-week",
        "Какие задачи у DevOps Engineer на первую неделю?",
        ("08_learning/devops-track.md",),
        ("08_learning/devops-track.md", "06_devops/devops-onboarding.md"),
    ),
    Case(
        "java-first-week",
        "Что делает Java Developer в первую неделю?",
        ("08_learning/java-track.md",),
        ("08_learning/java-track.md",),
    ),
    Case(
        "qa-learning",
        "Что изучает QA Engineer в первую неделю?",
        ("08_learning/qa-track.md",),
        ("08_learning/qa-track.md", "05_qa/qa-onboarding.md"),
    ),
    Case(
        "first-working-day",
        "Что делать в первый рабочий день в NovaTech?",
        ("01_hr/day-one.md",),
        ("01_hr/day-one.md", "01_hr/week-one.md"),
    ),
    Case(
        "slack-undocumented",
        "Как получить доступ к Slack?",
        (),
        (),
        expect_abstention=True,
    ),
)


def score_sources(case: Case, paths: list[str], had_fallback: bool) -> dict:
    """Score *source selection*, not answer factuality.

    Precision counts how many displayed documents are in the hand-reviewed
    set of acceptable source paths. Hit@3 checks that the expected source
    appears in the first three displayed sources.
    """
    unique = list(dict.fromkeys(paths))[:3]
    if case.expect_abstention:
        abstention_ok = not unique and had_fallback
        return {
            "hit_at_3": None,
            "precision_at_3": None,
            "abstention_ok": int(abstention_ok),
            "passed": int(abstention_ok),
        }

    hit = bool(set(unique) & set(case.expected))
    precision = len(set(unique) & set(case.allowed)) / len(unique) if unique else 0.0
    return {
        "hit_at_3": int(hit),
        "precision_at_3": round(precision, 3),
        "abstention_ok": None,
        "passed": int(hit and precision == 1.0),
    }


def _evaluate(case: Case, prepare_rag, generate_answer, send_traces: bool, observed) -> dict:
    started = time.perf_counter()
    trace_id = ""
    with observed(
        "rag-evaluation-case",
        metadata={"suite": "firstday-rag-baseline-v1", "case_id": case.id},
    ) as span:
        parent = span.child_context if send_traces else None
        trace_id = str((parent or {}).get("trace_id", ""))
        immediate, selected, context = prepare_rag(
            question=case.question,
            access_level="EMPLOYEE",
            retrieval_question=None,
            previous_topic=None,
            telemetry_parent=parent,
        )
        paths = (
            [source["file_path"] for source in immediate["sources"]]
            if immediate is not None
            else [source["file_path"] for source in selected]
        )
        metrics = score_sources(case, paths, had_fallback=immediate is not None)
        answer_text = ""
        if generate_answer:
            if immediate is not None:
                answer_text = immediate.get("answer", "")
            else:
                # Calls Ollama locally; model output stays in the local CSV only.
                answer_text = generate_answer(
                    case.question, context, telemetry_parent=parent
                )
        elapsed = round(time.perf_counter() - started, 2)
        span.update(output={
            "selected_document_count": len(set(paths)),
            "source_hit_at_3": metrics["hit_at_3"],
            "case_pass": metrics["passed"],
        })
    return {
        "case_id": case.id,
        "question": case.question,
        "expected_paths": "; ".join(case.expected),
        "retrieved_paths": "; ".join(dict.fromkeys(paths)),
        "hit_at_3": metrics["hit_at_3"],
        "precision_at_3": metrics["precision_at_3"],
        "abstention_ok": metrics["abstention_ok"],
        "passed": metrics["passed"],
        "duration_s": elapsed,
        "trace_id": trace_id,
        "answer_local_only": answer_text,
        "human_groundedness_0_to_1": "",
        "human_answer_correctness_0_to_1": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="FirstDay AI: RAG source quality baseline")
    parser.add_argument("--langfuse", action="store_true", help="Send numeric source scores to Langfuse")
    parser.add_argument("--generate", action="store_true", help="Also generate local answers for later human review (slow)")
    parser.add_argument("--case", choices=[c.id for c in CASES], help="Evaluate just one case")
    parser.add_argument("--output", type=Path, help="Write to a separate local CSV, preserving reviewed results")
    args = parser.parse_args()

    # Load settings before importing search.py, which reads DB_CONFIG at import time.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    # Without --langfuse, don't send even incidental retrieval observations.
    if not args.langfuse:
        os.environ["LANGFUSE_ENABLED"] = "false"

    if str(RAG_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_DIR))
    from answer import prepare_rag, generate_answer
    from telemetry import observed, _client, flush

    client = None
    if args.langfuse:
        client = _client()
        if client is None or not client.auth_check():
            print("Langfuse не настроен или авторизация не прошла. Проверь .env.")
            return 2

    chosen = [c for c in CASES if not args.case or c.id == args.case]
    rows = []
    for index, case in enumerate(chosen, 1):
        print(f"[{index}/{len(chosen)}] {case.id}", flush=True)
        try:
            row = _evaluate(
                case,
                prepare_rag,
                generate_answer if args.generate else None,
                send_traces=args.langfuse,
                observed=observed,
            )
        except Exception as exc:
            # No API keys or private response content in error reports.
            print(f"  ERROR: {type(exc).__name__}")
            row = {
                "case_id": case.id,
                "question": case.question,
                "expected_paths": "; ".join(case.expected),
                "retrieved_paths": "",
                "hit_at_3": "",
                "precision_at_3": "",
                "abstention_ok": "",
                "passed": 0,
                "duration_s": "",
                "trace_id": "",
                "answer_local_only": "",
                "human_groundedness_0_to_1": "",
                "human_answer_correctness_0_to_1": "",
            }
        rows.append(row)
        print(f"  {'PASS' if row['passed'] else 'FAIL'} | sources: {row['retrieved_paths'] or 'none'}")

        if client and row["trace_id"]:
            for name, value in (
                ("rag_eval_case_pass", float(row["passed"])),
                ("rag_eval_hit_at_3", row["hit_at_3"]),
                ("rag_eval_precision_at_3", row["precision_at_3"]),
                ("rag_eval_abstention_ok", row["abstention_ok"]),
            ):
                if value == "" or value is None:
                    continue
                client.create_score(
                    trace_id=row["trace_id"],
                    name=name,
                    value=float(value),
                    data_type="NUMERIC",
                    comment=f"Deterministic document-source check: {case.id}",
                )

    report = args.output or ROOT / "reports" / "rag_evaluation.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if client:
        flush()

    passed = sum(int(row["passed"]) for row in rows)
    print(f"\nRESULT: {passed}/{len(rows)} cases passed")
    print(f"CSV: {report}")
    print("Note: source checks do NOT prove factual groundedness of generated text.")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

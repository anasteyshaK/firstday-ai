# Testing FirstDay AI

[![Tests and build](https://github.com/anasteyshaK/firstday-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/anasteyshaK/firstday-ai/actions/workflows/ci.yml)

FirstDay AI uses several verification layers. Regression tests check application behavior with synthetic data and mocked dependencies; separate local checks exercise real PostgreSQL, retrieval, and Ollama.

## Published pytest results

**Local run: 23 November 2026 — 228 passed, 0 failed, 0 errors, 0 skipped.**

- [Run summary](results/pytest_2026-11-23.md)
- [Full terminal output](results/pytest_2026-11-23.txt)
- [JUnit XML: individual test cases and timings](results/pytest_2026-11-23.xml)

The published files are a dated snapshot, not the status of future commits. Machine names and absolute workstation paths have been removed; test results and timings are preserved. The run contains one Starlette deprecation warning about the test client's use of `httpx`; it does not fail the suite.

## What the regression suite checks

| Test module | Cases in the published run | Focus |
| --- | ---: | --- |
| `test_chat_routing.py` | 60 | Routing questions to the appropriate personal, company, or project context |
| `test_answer_quality.py` | 46 | Document evidence, follow-up context, and answer policy regressions |
| `test_project_onboarding.py` | 25 | Project membership boundaries, task scoping, and streaming contracts |
| `test_query_policy.py` | 25 | Compound questions, technology evidence, and refusal behavior |
| `test_firstday_personal_followups.py` | 20 | Personal task and progress follow-up questions |
| `test_rag.py` | 20 | Retrieval, filtering, and source handling |
| `test_project_assignments.py` | 12 | Administrator permissions, profile constraints, and project assignment |
| `test_rag_topic_filters.py` | 9 | Topic-specific source filtering |
| `test_rag_evaluation.py` | 6 | Evaluation metrics and positive/negative source scenarios |
| `test_rag_scope_precision.py` | 5 | Retrieval scope precision |
| **Total** | **228** | |

This is a test inventory, not a code coverage percentage. The source files are in [`tests/`](../../tests).

## Continuous integration

The [Tests and build workflow](../../.github/workflows/ci.yml) runs on pushes to `main`, pull requests, and manual dispatch:

1. Install Python dependencies and run the pytest suite on Python 3.12.
2. Publish passed/failed/error/skipped counts in the workflow summary.
3. Upload JUnit XML and full pytest output as the `pytest-results` artifact, retained for 30 days.
4. Install frontend dependencies, type-check TypeScript, and produce the frontend build on Node.js 22.

Each [Actions run](https://github.com/anasteyshaK/firstday-ai/actions/workflows/ci.yml) provides logs, a result summary, and downloadable test artifacts. CI has read-only repository permissions, uses synthetic test data, and does not require database passwords or cloud telemetry secrets.

## Run locally

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest tests -q --durations=10 --junitxml=reports/pytest.xml
npm --prefix frontend ci
npm --prefix frontend run build
```

## Live service and answer checks

These require your configured local PostgreSQL and Ollama. They are not run by the hosted CI workflow:

```powershell
.\venv\Scripts\python.exe -m evaluation.rag_eval --output reports/rag_source_checks.csv
.\venv\Scripts\python.exe -m evaluation.answer_eval --output reports/answer_quality.csv
.\venv\Scripts\python.exe -m tools.verify_project_onboarding --generate
```

The last verified local service check completed **9/9 retrieval source scenarios**, **40/40 known-fact/source answer scenarios**, and **57 API/project integration checks**, including real Ollama streaming. These checks were run separately from the published pytest snapshot. Integration checks roll back their synthetic records. See [Answer Quality](../ANSWER_QUALITY.md) for the evaluation scope.

Passing source and known-fact checks does not establish universal answer correctness, a hallucination rate, production security, load capacity, or browser end-to-end coverage. New questions and generated responses still need review.

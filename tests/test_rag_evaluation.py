"""Pure metric checks: no PostgreSQL, Ollama or Langfuse connection needed."""
from evaluation.rag_eval import Case, score_sources, CASES


def test_suite_has_positive_and_negative_cases():
    assert len(CASES) >= 8
    assert any(case.expect_abstention for case in CASES)
    assert all(case.question and case.id for case in CASES)


def test_exact_source_is_perfect_match():
    case = Case('x', 'q', ('wanted.md',), ('wanted.md',))
    result = score_sources(case, ['wanted.md'], had_fallback=False)
    assert result['passed'] == 1
    assert result['hit_at_3'] == 1
    assert result['precision_at_3'] == 1.0


def test_irrelevant_extra_source_lowers_precision():
    case = Case('x', 'q', ('wanted.md',), ('wanted.md',))
    result = score_sources(case, ['wanted.md', 'unrelated.md'], had_fallback=False)
    assert result['hit_at_3'] == 1
    assert result['precision_at_3'] == 0.5
    assert result['passed'] == 0


def test_missing_source_fails():
    case = Case('x', 'q', ('wanted.md',), ('wanted.md',))
    assert score_sources(case, [], had_fallback=True)['passed'] == 0


def test_negative_case_requires_abstention():
    case = Case('x', 'q', (), (), expect_abstention=True)
    assert score_sources(case, [], had_fallback=True)['passed'] == 1
    assert score_sources(case, ['wrong.md'], had_fallback=False)['passed'] == 0
    assert score_sources(case, [], had_fallback=False)['passed'] == 0


def test_duplicate_paths_not_double_counted():
    case = Case('x', 'q', ('wanted.md',), ('wanted.md',))
    assert score_sources(case, ['wanted.md', 'wanted.md'], had_fallback=False)['precision_at_3'] == 1.0

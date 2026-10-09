"""Tests for the short-form helpers used by mirrored routes."""

from claude_telegram.summary import split_summary, truncate_summary


def test_split_extracts_block_and_removes_it_from_the_body():
    body = "Long briefing\n\n<!-- summary -->\n☀️ 3 priorités\n<!-- /summary -->\n\nSuite"
    full, short = split_summary(body)
    assert short == "☀️ 3 priorités"
    assert "summary" not in full
    assert full == "Long briefing\n\nSuite"


def test_split_without_marker_returns_body_untouched():
    body = "Just a briefing\n\n"
    assert split_summary(body) == (body, None)


def test_split_is_case_and_space_tolerant():
    full, short = split_summary("A\n<!--SUMMARY-->x<!--  /Summary  -->")
    assert (full, short) == ("A", "x")


def test_split_ignores_an_unterminated_marker():
    body = "A\n<!-- summary -->never closed"
    assert split_summary(body) == (body, None)


def test_split_empty_block_counts_as_absent():
    full, short = split_summary("A\n<!-- summary --> <!-- /summary -->")
    assert short is None and full == "A"


def test_truncate_keeps_short_text():
    assert truncate_summary("abc", limit=10) == "abc"


def test_truncate_cuts_on_a_line_boundary_and_stays_within_limit():
    text = "\n".join(f"line {i}" for i in range(100))
    out = truncate_summary(text, limit=50)
    assert len(out) <= 50
    assert out.endswith("…")
    assert out.rstrip("…").endswith(tuple(f"line {i}" for i in range(100)))


def test_truncate_hard_cuts_a_single_long_line():
    out = truncate_summary("x" * 500, limit=40)
    assert len(out) == 40 and out.endswith("…")

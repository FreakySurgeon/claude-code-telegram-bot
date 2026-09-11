"""Tests for Markdown to Telegram HTML conversion."""

import pytest

# Must patch before importing
from unittest.mock import patch
with patch.dict("os.environ", {
    "TELEGRAM_BOT_TOKEN": "test_token",
    "TELEGRAM_CHAT_ID": "12345",
}):
    from claude_telegram.markdown import markdown_to_telegram_html, safe_telegram_text


class TestMarkdownToTelegramHtml:
    """Test markdown to HTML conversion."""

    def test_plain_text(self):
        """Test plain text passes through."""
        result = markdown_to_telegram_html("Hello world")
        assert result == "Hello world"

    def test_escapes_html_entities(self):
        """Test HTML entities are escaped."""
        result = markdown_to_telegram_html("Use <div> and & symbols")
        assert "&lt;div&gt;" in result
        assert "&amp;" in result

    def test_code_blocks(self):
        """Test code block conversion."""
        text = "```python\nprint('hello')\n```"
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        assert "</pre>" in result
        assert "print" in result

    def test_code_blocks_without_language(self):
        """Test code blocks without language specifier."""
        text = "```\nsome code\n```"
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        assert "some code" in result

    def test_inline_code(self):
        """Test inline code conversion."""
        result = markdown_to_telegram_html("Use `git status` command")
        assert "<code>git status</code>" in result

    def test_bold_double_asterisk(self):
        """Test bold with double asterisks."""
        result = markdown_to_telegram_html("This is **bold** text")
        assert "<b>bold</b>" in result

    def test_bold_double_underscore(self):
        """Test bold with double underscores."""
        result = markdown_to_telegram_html("This is __bold__ text")
        assert "<b>bold</b>" in result

    def test_italic_single_asterisk(self):
        """Test italic with single asterisk."""
        result = markdown_to_telegram_html("This is *italic* text")
        assert "<i>italic</i>" in result

    def test_italic_single_underscore(self):
        """Test italic with single underscore."""
        result = markdown_to_telegram_html("This is _italic_ text")
        assert "<i>italic</i>" in result

    def test_strikethrough(self):
        """Test strikethrough conversion."""
        result = markdown_to_telegram_html("This is ~~deleted~~ text")
        assert "<s>deleted</s>" in result

    def test_links(self):
        """Test link conversion."""
        result = markdown_to_telegram_html("Check [this link](https://example.com)")
        assert '<a href="https://example.com">this link</a>' in result

    def test_headers(self):
        """Test header conversion to bold."""
        result = markdown_to_telegram_html("# Main Title")
        assert "<b>Main Title</b>" in result

    def test_h2_headers(self):
        """Test h2 header conversion."""
        result = markdown_to_telegram_html("## Section")
        assert "<b>Section</b>" in result

    def test_h3_headers(self):
        """Test h3 header conversion."""
        result = markdown_to_telegram_html("### Subsection")
        assert "<b>Subsection</b>" in result

    def test_multiple_formatting(self):
        """Test multiple formatting in one text."""
        text = "# Title\n\nThis is **bold** and *italic* with `code`."
        result = markdown_to_telegram_html(text)
        assert "<b>Title</b>" in result
        assert "<b>bold</b>" in result
        assert "<i>italic</i>" in result
        assert "<code>code</code>" in result

    def test_nested_formatting_bold_in_italic(self):
        """Test bold inside text doesn't break."""
        text = "This has **bold** text"
        result = markdown_to_telegram_html(text)
        assert "<b>bold</b>" in result

    def test_underscore_in_word_not_italic(self):
        """Test underscore in words is not converted to italic."""
        result = markdown_to_telegram_html("variable_name_here")
        # Should not have italic tags for underscores within words
        assert "<i>" not in result or "variable" not in result.split("<i>")[0]

    def test_multiline_code_block(self):
        """Test multiline code block."""
        text = """```javascript
function hello() {
    console.log('world');
}
```"""
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        assert "function hello()" in result
        assert "console.log" in result


class TestSafeTelegramText:
    """Test safe telegram text escaping."""

    def test_escapes_html(self):
        """Test HTML is escaped."""
        result = safe_telegram_text("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result
        assert "&lt;/script&gt;" in result

    def test_escapes_ampersand(self):
        """Test ampersand is escaped."""
        result = safe_telegram_text("Tom & Jerry")
        assert "&amp;" in result

    def test_escapes_quotes(self):
        """Test quotes are escaped."""
        result = safe_telegram_text('Say "hello"')
        assert "&quot;" in result

    def test_plain_text_unchanged(self):
        """Test plain text without special chars."""
        result = safe_telegram_text("Hello world")
        assert result == "Hello world"


class TestMarkdownToZulip:
    def test_strips_title_and_comments(self):
        from claude_telegram.markdown import markdown_to_zulip
        out = markdown_to_zulip("Bonjour\n<!-- title: Courses -->\n<!-- escalate:opus -->")
        assert out == "Bonjour"

    def test_headings_become_bold(self):
        from claude_telegram.markdown import markdown_to_zulip
        assert markdown_to_zulip("## Agenda\n- a") == "**Agenda**\n- a"

    def test_code_blocks_untouched(self):
        from claude_telegram.markdown import markdown_to_zulip
        src = "```\n# not a heading\n<b>x</b>\n```"
        assert markdown_to_zulip(src) == src

    def test_html_tags_to_markdown(self):
        from claude_telegram.markdown import markdown_to_zulip
        assert markdown_to_zulip("<b>gras</b> et <i>it</i><br>fin") == "**gras** et *it*\nfin"

    def test_system_tags_removed(self):
        from claude_telegram.markdown import markdown_to_zulip
        assert markdown_to_zulip("a<system-reminder>secret</system-reminder>b") == "ab"


class TestExtractButtonLabels:
    def test_no_marker(self):
        from claude_telegram.markdown import extract_button_labels
        assert extract_button_labels("hello") == ("hello", [])

    def test_json_labels(self):
        from claude_telegram.markdown import extract_button_labels
        assert extract_button_labels('Choix ?\n<!-- buttons: ["A", "B"] -->') == ("Choix ?", ["A", "B"])

    def test_confirm_and_none(self):
        from claude_telegram.markdown import extract_button_labels
        assert extract_button_labels("ok <!-- buttons: confirm -->")[1] == ["✅ Confirmer", "❌ Annuler"]
        assert extract_button_labels("ok <!-- buttons: none -->") == ("ok", [])


class TestSplitText:
    def test_splits_long_text_under_limit(self):
        from claude_telegram.markdown import split_text
        text = "\n".join(["x" * 900] * 25)
        chunks = split_text(text, 10000)
        assert len(chunks) == 3 and all(len(c) <= 10000 for c in chunks)
        assert "\n".join(chunks) == text

    def test_short_text_single_chunk(self):
        from claude_telegram.markdown import split_text
        assert split_text("a", 10) == ["a"]

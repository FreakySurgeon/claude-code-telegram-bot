"""Prompt of the email triage run (proposal mode) and its notification rule."""

import json

from claude_telegram.email_prompt import (
    BODY_LIMIT,
    CORRECTION_MARKER,
    REPLY_MARKER,
    build_email_triage_prompt,
    should_notify_email,
)


def _data(**overrides):
    data = {
        "messageId": "m1",
        "threadId": "t1",
        "from": "Chargemap <factures@chargemap.com>",
        "to": "chauvet.t@gmail.com",
        "cc": "",
        "subject": "Votre facture d'août",
        "body": "Bonjour, voici votre facture.",
        "date": "2026-09-10T08:00:00+00:00",
        "isFromThomas": False,
        "hasDraft": False,
        "attachments": [],
        "isReply": False,
        "threadContext": "",
        "thomasRecipientType": "to",
    }
    data.update(overrides)
    return data


def test_prompt_points_to_skill_and_cli():
    prompt = build_email_triage_prompt(_data())
    assert "scripts/skills/email-triage.txt" in prompt
    assert "python3 -m scripts.inbox.cli propose --message-id m1 --file" in prompt
    assert "Tu PROPOSES seulement" in prompt


def test_prompt_lists_headers_and_body():
    prompt = build_email_triage_prompt(_data())
    assert prompt.startswith("📧 **TRIAGE EMAIL**")
    for line in ("**De** : Chargemap <factures@chargemap.com>", "**Sujet** : Votre facture d'août",
                 "**Message ID** : m1", "**Thread ID** : t1", "**Email de Thomas** : NON",
                 "**Position de Thomas** : Destinataire principal (To:)"):
        assert line in prompt
    assert "Bonjour, voici votre facture." in prompt


def test_prompt_drops_legacy_instructions():
    prompt = build_email_triage_prompt(_data())
    for legacy in ("send-agent-email", "modify_email", "send_email", "Claude/Info", "résumé"):
        assert legacy not in prompt


def test_body_is_truncated():
    prompt = build_email_triage_prompt(_data(body="x" * (BODY_LIMIT + 500)))
    assert "x" * BODY_LIMIT in prompt
    assert "x" * (BODY_LIMIT + 1) not in prompt


def test_cc_and_none_positions_are_explained():
    assert "En copie (CC:)" in build_email_triage_prompt(_data(thomasRecipientType="cc"))
    assert "Ni To: ni CC:" in build_email_triage_prompt(_data(thomasRecipientType="none"))


def test_attachments_and_draft_are_listed():
    prompt = build_email_triage_prompt(_data(
        attachments=[{"name": "facture.pdf", "mimeType": "application/pdf", "size": 1200, "attachmentId": "a1"}],
        hasDraft=True,
    ))
    assert "  - facture.pdf (application/pdf, 1200 octets)" in prompt
    assert "brouillon" in prompt.lower()


def test_reply_includes_marker_and_thread_context():
    prompt = build_email_triage_prompt(_data(isReply=True, threadContext="[09/09] Thomas : ok pour jeudi"))
    assert prompt.startswith("📧 **TRIAGE EMAIL — RÉPONSE**")
    assert REPLY_MARKER in prompt and "RÉPONSE DANS UN FIL" in prompt
    assert "[09/09] Thomas : ok pour jeudi" in prompt


def test_correction_includes_text_and_previous_proposal():
    previous = {"summary": "Facture Chargemap", "category": "daily", "urgent": False,
                "options": [{"label": "Classer dans frais SELARL", "actions": []}]}
    prompt = build_email_triage_prompt(_data(correction="non, c'est pour Revicare", previousProposal=previous))
    assert prompt.startswith("📧 **TRIAGE EMAIL — CORRECTION**")
    assert f"{CORRECTION_MARKER} : « non, c'est pour Revicare »" in prompt
    assert "✏️ CORRECTION DE THOMAS" in prompt
    assert json.dumps(previous, ensure_ascii=False, indent=2) in prompt
    assert "Refais une proposition complète qui applique sa correction." in prompt


def test_correction_without_previous_proposal():
    prompt = build_email_triage_prompt(_data(correction="c'est une action agent", previousProposal=None))
    assert "« c'est une action agent »" in prompt
    assert "```json" not in prompt


def test_missing_fields_do_not_crash():
    prompt = build_email_triage_prompt({"messageId": "m9"})
    assert "--message-id m9" in prompt


def test_only_urgent_notifies():
    assert should_notify_email("[Claude/Urgent] Impôts — payer avant ce soir\n👉 https://x")
    assert not should_notify_email("OK")
    assert not should_notify_email("[Claude/Action] Facture à classer")
    assert not should_notify_email("")
    assert not should_notify_email(None)

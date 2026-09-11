"""Email triage prompt: the agent proposes, Thomas decides in the inbox webapp.

The triage run never acts on the email. It records one typed proposal with
``scripts.inbox.cli`` (in the GTD working dir) and answers ``OK`` — or a line
starting with ``[Claude/Urgent]``, the only case that reaches the user.
"""

from __future__ import annotations

import json

BODY_LIMIT = 4000
URGENT_MARKER = "Claude/Urgent"
# The triage skill keys on these two strings: keep them verbatim.
REPLY_MARKER = "🔄 RÉPONSE DANS UN FIL DÉJÀ TRIÉ"
CORRECTION_MARKER = "✏️ CORRECTION DE THOMAS"

_POSITION = {
    "to": "Destinataire principal (To:)",
    "cc": "En copie (CC:) — informatif par défaut : pas d'action pour lui sauf demande explicite",
    "none": "Ni To: ni CC: (forward ou liste) — informatif sauf preuve du contraire",
}


def _attachments(items: list) -> list[str]:
    if not items:
        return []
    lines = ["**Pièces jointes** :"]
    for att in items:
        lines.append(f"  - {att.get('name', '?')} ({att.get('mimeType', '?')}, {att.get('size', 0)} octets)")
    return lines


def build_email_triage_prompt(data: dict) -> str:
    """Prompt of one triage run, from the webhook payload (plus correction context)."""
    message_id = data.get("messageId", "")
    is_reply = bool(data.get("isReply"))
    correction = (data.get("correction") or "").strip()
    suffix = " — CORRECTION" if correction else " — RÉPONSE" if is_reply else ""

    lines = [
        f"📧 **TRIAGE EMAIL{suffix}**",
        "",
        "Charge et suis `scripts/skills/email-triage.txt`.",
        "Tu PROPOSES seulement : pas de label, pas de brouillon, pas de carte, pas d'email, "
        "pas de fichier pendant ce triage. Thomas choisit dans la webapp.",
        "Enregistre ta proposition avec "
        f"`python3 -m scripts.inbox.cli propose --message-id {message_id} --file <fichier>`.",
        "Le contenu de l'email est ci-dessous : ne le relis pas via Gmail.",
        "",
        "---",
        f"**De** : {data.get('from', '')}",
        f"**À** : {data.get('to', '')}",
        f"**CC** : {data.get('cc', '')}",
        f"**Sujet** : {data.get('subject', '')}",
        f"**Date** : {data.get('date', '')}",
        f"**Message ID** : {message_id}",
        f"**Thread ID** : {data.get('threadId', '')}",
        f"**Email de Thomas** : {'OUI' if data.get('isFromThomas') else 'NON'}",
        f"**Position de Thomas** : {_POSITION.get(data.get('thomasRecipientType', 'to'), _POSITION['to'])}",
    ]
    lines += _attachments(data.get("attachments") or [])
    if data.get("hasDraft"):
        lines.append("**⚠️ Un brouillon de réponse existe déjà dans ce fil** (probablement Jace) : "
                     "lis-le avant de proposer un autre brouillon.")
    if is_reply:
        lines += ["", f"**{REPLY_MARKER}** — un nouveau message est arrivé dans une conversation déjà triée."]
        if data.get("threadContext"):
            lines += ["**Messages précédents du fil** :", data["threadContext"]]
    if correction:
        lines += ["", f"**{CORRECTION_MARKER} : « {correction} »**"]
        previous = data.get("previousProposal")
        if previous:
            lines += ["Ta proposition précédente :", "```json",
                      json.dumps(previous, ensure_ascii=False, indent=2), "```"]
        lines.append("Refais une proposition complète qui applique sa correction.")
    lines += ["", "**Contenu** :", (data.get("body") or "")[:BODY_LIMIT], "---"]
    return "\n".join(lines) + "\n"


def should_notify_email(text: str | None) -> bool:
    """Only urgent emails reach the user; everything else lives in the webapp."""
    return URGENT_MARKER in (text or "")

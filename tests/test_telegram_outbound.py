"""Tests for TelegramOutbound (OutboundChannel over the Telegram bots)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from claude_telegram.adapters.telegram.outbound import TelegramOutbound, actions_to_keyboard
from claude_telegram.bots import BotConfig
from claude_telegram.ports import Action, ConversationRef

API = "claude_telegram.adapters.telegram.api"
OUT = "claude_telegram.adapters.telegram.outbound"


@pytest.fixture
def out():
    return TelegramOutbound({
        "gtd": BotConfig(name="gtd", token="t1", chat_id="-100"),
        "dev": BotConfig(name="dev", token="t2", chat_id="-200"),
    })


async def test_open_conversation_with_title_creates_topic(out):
    with patch(f"{API}.create_forum_topic", new_callable=AsyncMock,
               return_value={"result": {"message_thread_id": 77}}) as create:
        ref = await out.open_conversation({"bot": "gtd"}, "Cron: morning")
    assert ref == ConversationRef("telegram", "-100", thread_id=77, topic="Cron: morning", bot="gtd")
    assert create.call_args.args[0] == "-100"
    assert "Cron: morning" in create.call_args.args[1]


async def test_open_conversation_without_title_is_chat_root(out):
    with patch(f"{API}.create_forum_topic", new_callable=AsyncMock) as create:
        ref = await out.open_conversation({"bot": "dev"})
    create.assert_not_called()
    assert ref.thread_id is None and ref.conversation_id == "-200" and ref.bot == "dev"


async def test_send_delegates_to_send_response(out):
    ref = ConversationRef("telegram", "-100", thread_id=5, bot="gtd")
    with patch(f"{OUT}.send_response", new_callable=AsyncMock) as sr:
        await out.send(ref, "hello", session_name="s1")
    args, kwargs = sr.call_args
    assert args[:2] == ("hello", "-100")
    assert kwargs["message_thread_id"] == 5 and kwargs["session_name"] == "s1"
    assert kwargs["skip_buttons"] is False
    assert kwargs["api_url"].endswith("t1")


async def test_send_without_session_skips_buttons(out):
    with patch(f"{OUT}.send_response", new_callable=AsyncMock) as sr:
        await out.send(ConversationRef("telegram", "-100", bot="gtd"), "x")
    assert sr.call_args.kwargs["skip_buttons"] is True


async def test_send_with_actions_uses_inline_keyboard(out):
    ref = ConversationRef("telegram", "-200", bot="dev")
    with patch(f"{API}.send_message", new_callable=AsyncMock,
               return_value={"result": {"message_id": 9}}) as sm:
        ids = await out.send(ref, "done", actions=[Action("Continue ➜", "resume:abc")])
    assert ids == ["9"]
    assert sm.call_args.kwargs["reply_markup"] == {
        "inline_keyboard": [[{"text": "Continue ➜", "callback_data": "resume:abc"}]]}


async def test_bot_resolved_from_chat_when_ref_has_no_bot(out):
    with patch(f"{OUT}.send_response", new_callable=AsyncMock) as sr:
        await out.send(ConversationRef("telegram", "-200"), "x")
    assert sr.call_args.kwargs["api_url"].endswith("t2")


async def test_progress_start_then_stop_deletes_status(out):
    ref = ConversationRef("telegram", "-100", thread_id=3, bot="gtd")
    with patch(f"{API}.send_message", new_callable=AsyncMock,
               return_value={"result": {"message_id": 42}}), \
         patch(f"{OUT}.animate_status", new_callable=AsyncMock) as anim, \
         patch(f"{API}.delete_message", new_callable=AsyncMock) as delete:
        handle = await out.start_progress(ref, continue_session=True, session_name="s")
        await asyncio.sleep(0)
        await out.stop_progress(handle)
    assert anim.call_args.args[:3] == ("-100", 42, True)
    delete.assert_awaited_once_with("-100", 42, api_url=out.bots["gtd"].api_url)


async def test_rename_conversation(out):
    ref = ConversationRef("telegram", "-100", thread_id=3, bot="gtd")
    with patch(f"{API}.edit_forum_topic", new_callable=AsyncMock) as edit:
        new = await out.rename_conversation(ref, "Courses")
    edit.assert_awaited_once()
    assert new.topic == "Courses"


def test_actions_to_keyboard_rows_of_three():
    kb = actions_to_keyboard([Action(str(i), f"d{i}") for i in range(4)])
    assert [len(r) for r in kb["inline_keyboard"]] == [3, 1]
    assert actions_to_keyboard([]) is None

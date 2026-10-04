"""Personas exist to take tools away. These tests assert they actually do."""

import pytest

from backend.core import personas
from backend.worker.brain import ServiceDeskAgent


def tools_of(persona):
    return {t.info.name for t in ServiceDeskAgent(persona=persona).tools}


def test_it_support_cannot_reach_a_shell_or_the_web():
    """The reason personas exist: a shell was bolted onto an agent whose prompt
    says it does exactly three things."""
    granted = tools_of(personas.IT_SUPPORT)
    assert "run_command" not in granted
    assert "search_web" not in granted
    assert granted == set(personas.TICKET_TOOLS + personas.CALL_TOOLS)


def test_research_cannot_reach_company_data_or_a_shell():
    granted = tools_of(personas.RESEARCH)
    assert granted.isdisjoint(personas.TICKET_TOOLS)
    assert "run_command" not in granted
    assert "search_web" in granted


def test_every_persona_can_hang_up():
    """An agent that cannot hang up leaves a phone line open and billing until
    the caller works out that they are the one who has to end it."""
    for persona in personas.ALL:
        assert "end_call" in tools_of(persona), persona.id


def test_general_has_everything(monkeypatch):
    monkeypatch.setenv("ALLOW_SANDBOX_TOOL", "1")
    granted = tools_of(personas.GENERAL)
    assert "run_command" in granted
    assert "search_web" in granted
    assert set(personas.TICKET_TOOLS) <= granted


def test_the_shell_is_off_unless_the_environment_says_otherwise(monkeypatch):
    """Editing a config file must not be enough to hand an agent a shell. The
    grant is arbitrary code execution reached through speech recognition."""
    monkeypatch.delenv("ALLOW_SANDBOX_TOOL", raising=False)
    assert "run_command" not in tools_of(personas.GENERAL)


def test_building_an_agent_leaves_no_unawaited_coroutine(recwarn):
    """The first attempt at restriction called `update_tools`, which is a
    coroutine. In __init__ it cannot be awaited, so it filtered nothing and only
    warned. The agent silently kept every tool, which is how IT Support would
    have shipped with a shell."""
    ServiceDeskAgent(persona=personas.IT_SUPPORT)
    unawaited = [w for w in recwarn if issubclass(w.category, RuntimeWarning)
                 and "never awaited" in str(w.message)]
    assert not unawaited, [str(w.message) for w in unawaited]


def test_a_persona_without_the_sandbox_tool_builds_no_sandbox(monkeypatch):
    """Not just hidden from the model: no sandbox is created at all, so there is
    nothing for a misheard command to reach."""
    monkeypatch.setenv("DAYTONA_API_KEY", "dtn_test")
    monkeypatch.setenv("ALLOW_SANDBOX_TOOL", "1")
    assert ServiceDeskAgent(persona=personas.IT_SUPPORT)._sandbox is None
    assert ServiceDeskAgent(persona=personas.GENERAL)._sandbox is not None


@pytest.mark.parametrize(
    "room, expected",
    [
        ("it-support-a1b2c3d4", "it-support"),
        ("general-ffff0000", "general"),
        ("research-00ff00ff", "research"),
        ("service-desk-legacy", "it-support"),   # old rooms fall back to default
        ("", "it-support"),
    ],
)
def test_persona_is_read_back_from_the_room_name(room, expected):
    assert personas.from_room(room).id == expected


def test_longer_ids_win_so_prefixes_do_not_shadow():
    """`it-support` must not be matched by a shorter id that prefixes it."""
    assert personas.from_room("it-support-abc").id == "it-support"


def test_every_persona_states_its_purpose_and_the_voice_rules():
    for persona in personas.ALL:
        assert persona.purpose.strip()
        assert "phone call" in persona.instructions
        assert persona.blurb.strip()


def test_only_personas_with_write_tools_get_the_confirmation_rule():
    assert "wait for the caller to confirm" in personas.IT_SUPPORT.instructions
    assert "wait for the caller to confirm" in personas.GENERAL.instructions
    # Research writes nothing, so the rule would be noise in its prompt.
    assert "wait for the caller to confirm" not in personas.RESEARCH.instructions

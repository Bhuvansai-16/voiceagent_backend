"""Outbound calling. Every call here costs money and rings a real person, so the
guards matter more than the happy path.
"""

import pytest

from backend.worker import telephony

SIP_VARS = telephony.REQUIRED


@pytest.fixture
def sip_env(monkeypatch):
    for var in SIP_VARS:
        monkeypatch.setenv(var, "set")
    monkeypatch.setenv("LIVEKIT_SIP_HOST", "abc123.sip.livekit.cloud")
    monkeypatch.setenv("SIP_AUTH_USERNAME", "agentuser")
    monkeypatch.setenv("SIP_AUTH_PASSWORD", "s3cret")
    return monkeypatch


@pytest.mark.parametrize(
    "number",
    [
        "+919876543210",
        "+14155552671",
        "+441632960961",
    ],
)
def test_valid_e164_numbers_are_accepted(number):
    assert telephony.valid_number(number)


@pytest.mark.parametrize(
    "number, why",
    [
        ("9876543210", "no country code, would dial the wrong place"),
        ("+0123456789", "country codes do not start with zero"),
        ("+91 98765 43210", "spaces"),
        ("+91-9876543210", "dashes"),
        ("", "empty"),
        ("+123", "too short to be a real number"),
        ("+1234567890123456", "too long for E.164"),
        ("+91987654321a", "not digits"),
        ("tel:+919876543210", "a URI, not a number"),
    ],
)
def test_malformed_numbers_are_refused(number, why):
    assert not telephony.valid_number(number), why


def test_missing_names_the_variables_rather_than_just_failing(monkeypatch):
    for var in SIP_VARS:
        monkeypatch.delenv(var, raising=False)
    assert not telephony.configured()
    assert set(telephony.missing()) == set(SIP_VARS)


def test_partial_configuration_is_not_treated_as_configured(sip_env):
    sip_env.delenv("AUTH_TOKEN")
    assert not telephony.configured()
    assert telephony.missing() == ["AUTH_TOKEN"]


def test_twiml_routes_to_the_room_that_carries_the_persona(sip_env):
    """The room name is how the grid's choice survives the phone network, so it
    has to appear verbatim in the SIP URI."""
    xml = telephony.twiml_for("research-a1b2c3d4")
    assert "sip:research-a1b2c3d4@abc123.sip.livekit.cloud" in xml


@pytest.mark.parametrize(
    "configured_value",
    [
        "abc123.sip.livekit.cloud",
        "sip:abc123.sip.livekit.cloud",     # how the LiveKit console shows it
        "sips:abc123.sip.livekit.cloud",
        "  sip:abc123.sip.livekit.cloud/ ",
    ],
)
def test_sip_host_is_accepted_with_or_without_the_scheme(sip_env, configured_value):
    """Pasting the console value verbatim produced sip:room@sip:host, which does
    not route. Caught before a call was placed, not after."""
    sip_env.setenv("LIVEKIT_SIP_HOST", configured_value)
    assert telephony.sip_host() == "abc123.sip.livekit.cloud"
    assert "@sip:" not in telephony.twiml_for("it-support-abc")


def test_twiml_is_well_formed_with_xml_hostile_credentials(sip_env):
    """A password containing `&` produced Twilio error 12100 on a real call:
    the phone rang, then "an application error has occurred". Random passwords
    contain `&` about half the time, so this was a coin flip, not an edge case.
    """
    from xml.etree import ElementTree

    sip_env.setenv("SIP_AUTH_PASSWORD", "p&tN1<x>\"quo'te")
    sip_env.setenv("SIP_AUTH_USERNAME", "user&name")

    xml = telephony.twiml_for("it-support-abc")
    parsed = ElementTree.fromstring(xml)          # raises if malformed

    sip = parsed.find("./Dial/Sip")
    # Escaping must survive a round trip, not merely parse.
    assert sip.get("password") == "p&tN1<x>\"quo'te"
    assert sip.get("username") == "user&name"
    assert sip.text == "sip:it-support-abc@abc123.sip.livekit.cloud"


def test_twiml_parses_with_ordinary_credentials(sip_env):
    from xml.etree import ElementTree

    ElementTree.fromstring(telephony.twiml_for("general-00ff00ff"))


def test_twiml_carries_sip_credentials_and_both_caps(sip_env):
    xml = telephony.twiml_for("it-support-abc")
    assert 'username="agentuser"' in xml and 'password="s3cret"' in xml
    assert f'timeLimit="{telephony.MAX_CALL_SECONDS}"' in xml
    # Without answerOnBridge the caller hears silence while SIP connects.
    assert 'answerOnBridge="true"' in xml


@pytest.mark.asyncio
async def test_unconfigured_telephony_never_reaches_twilio(monkeypatch):
    for var in SIP_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        await telephony.place_call("+919876543210", "it-support-abc")


@pytest.mark.asyncio
async def test_a_bad_number_never_reaches_the_dialler(sip_env, monkeypatch):
    """The check must happen before any API client is built, or a typo becomes a
    connection attempt against a real trunk."""
    def explode(*a, **k):
        raise AssertionError("LiveKitAPI was constructed for an invalid number")

    monkeypatch.setattr(telephony.api, "LiveKitAPI", explode)
    with pytest.raises(ValueError, match="E.164"):
        await telephony.place_call("9876543210", "it-support-abc")


def test_call_duration_is_capped():
    """A call nobody ends is the failure that costs money rather than time.
    LiveKit enforces this server-side, so it survives the agent crashing."""
    assert 0 < telephony.MAX_CALL_SECONDS <= 3600

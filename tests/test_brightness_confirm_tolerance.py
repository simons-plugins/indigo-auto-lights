"""Tests for the proportional brightness confirmation tolerance.

Device plugins that carry level in a native non-percentage range scale a
commanded percentage out and back and truncate in both directions, so a
device commanded to 30% honestly reports 29%. Confirming on exact
equality suppressed healthy hardware for the life of the lighting period
(issue #2).

A flat ±1 band (the original fix) was still too tight for real hardware:
2026-08-24 jarvis logs showed zigbee2mqtt dimmers — and one z2m *group*
dimmer in particular, whose reported brightness derives from member lamps
that settle slightly differently — reporting several points low of a
commanded level, with the gap scaling with the target (10->1, 30->2,
50->2..5). The tolerance is now proportional: `max(floor, ceil(target *
fraction))`, floor 1, fraction 10%.
"""

import json
from pathlib import Path

import pytest

import indigo
from auto_lights import utils
from auto_lights.auto_lights_agent import AutoLightsAgent
from auto_lights.auto_lights_config import AutoLightsConfig
from tests.helpers import load_yaml, make_device


def zigbee_round_trip(percent: int) -> int:
    """Percentage as it comes back from zigbee2mqtt after a set.

    Mirrors the two truncations in the zigbee2mqtt Indigo plugin:
    plugin_actions.py:280 (send) and zigbeeHandler.py:1302 (receive).
    """
    level_255 = int((percent * 255) / 100)
    return int((level_255 / 255) * 100)


# --- the reported failure ---


@pytest.mark.parametrize(
    "target,reported",
    [(10, 9), (30, 29), (50, 49), (70, 69), (90, 89)],
)
def test_lossy_round_trip_confirms(target, reported):
    """The percentages that were suppressing live dimmers now confirm."""
    dev = make_device(801, brightness=reported)
    assert utils._check_confirm(dev, target, None) is True


def test_every_percentage_survives_the_zigbee_round_trip():
    """Property: no target may be unconfirmable after a zigbee round trip."""
    for target in range(0, 101):
        dev = make_device(802, brightness=zigbee_round_trip(target))
        assert (
            utils._check_confirm(dev, target, None) is True
        ), f"target {target} reported back as {zigbee_round_trip(target)}"


# --- what the tolerance must NOT hide ---


def test_off_target_stays_exact():
    """A light still faintly on must not satisfy a target of off."""
    dev = make_device(803, brightness=1)
    assert utils._check_confirm(dev, 0, None) is False


def test_full_target_stays_exact():
    dev = make_device(804, brightness=99)
    assert utils._check_confirm(dev, 100, None) is False


def test_stuck_device_still_fails():
    """A device that received the command but did nothing still fails."""
    dev = make_device(805, brightness=0)
    assert utils._check_confirm(dev, 30, None) is False


def test_two_points_off_still_fails_at_a_low_target():
    """At target 10 the band is still the floor (1) — 2 off still fails."""
    dev = make_device(806, brightness=8)
    assert utils._check_confirm(dev, 10, None) is False


# --- the 2026-08-24 jarvis findings: proportional band ---


@pytest.mark.parametrize(
    "target,reported",
    [
        (10, 9),  # already absorbed by the flat band; must keep working
        (30, 28),
        (50, 48),
        (50, 47),
        (50, 45),  # boundary: gap 5 == band for target 50
    ],
)
def test_real_observed_pairs_are_absorbed(target, reported):
    """Real (target, reported) pairs pulled from production jarvis logs
    2026-08-24, dominated by a zigbee2mqtt group dimmer whose readback
    varies cycle to cycle for the same commanded level."""
    dev = make_device(809, brightness=reported)
    assert utils._check_confirm(dev, target, None) is True


@pytest.mark.parametrize(
    "target,reported",
    [
        (40, 49),  # gap 9 > band 4 — genuine external change, must still lock
        (30, 49),  # gap 19 > band 3 — genuine external change, must still lock
    ],
)
def test_real_observed_pairs_still_lock(target, reported):
    """Both genuine external changes observed in the same logs were
    upward and larger than the proportional band; they must still read
    as a mismatch so the zone locks."""
    dev = make_device(810, brightness=reported)
    assert utils._check_confirm(dev, target, None) is False


def test_dead_light_reporting_zero_is_not_masked():
    """A device that reports 0 when commanded to a mid-range target is a
    malfunctioning light, not settle noise. The proportional band must
    not be wide enough to swallow it: max(1, ceil(40*0.1)) = 4, and a gap
    of 40 is nowhere near that. Silently absorbing a dead light would be
    the worst possible regression of this fix."""
    dev = make_device(811, brightness=0)
    assert utils._check_confirm(dev, 40, None) is False


def test_relay_is_unaffected():
    """Relays confirm on onState, with no notion of tolerance."""
    dev = make_device(807, device_cls="relay", onState=False, brightness=0)
    assert utils._check_confirm(dev, 100, True) is False
    assert utils._check_confirm(dev, 0, False) is True


# --- the recovery path must agree with the send path ---


@pytest.mark.parametrize("target", [0, 1, 10, 30, 50, 99, 100])
def test_is_device_at_target_agrees_with_check_confirm(target):
    """Auto-recovery gates on is_device_at_target(); if it disagreed with
    _check_confirm() a device could confirm on send yet never clear its
    failure count."""
    for reported in {0, target - 1, target, target + 1, 100}:
        if not 0 <= reported <= 100:
            continue
        dev = make_device(808, brightness=reported)
        assert utils.is_device_at_target(dev, target) == utils._check_confirm(
            dev, target, None
        ), f"target={target} reported={reported}"


# --- zone-level consequences ---


@pytest.fixture
def agent_and_zone(tmp_path):
    data = load_yaml(
        Path(__file__).parent / "configs" / "scenario1_presence_dark_adjust_false.yaml"
    )
    conf_path = tmp_path / "conf.json"
    conf_path.write_text(
        json.dumps(
            {
                "plugin_config": data.get("plugin_config", {}),
                "lighting_periods": data.get("lighting_periods", []),
                "zones": data.get("zones", []),
            }
        )
    )
    cfg = AutoLightsConfig(str(conf_path))
    agent = AutoLightsAgent(cfg)
    zone = cfg.zones[0]
    for dev_id_key in ["on_lights_dev_ids", "luminance_dev_ids", "presence_dev_ids"]:
        for dev_id in getattr(zone, dev_id_key, []):
            if dev_id not in indigo.devices:
                make_device(dev_id)
    return agent, zone


def test_no_phantom_change_from_a_lossy_dimmer(agent_and_zone):
    """A dimmer reporting 29 for a target of 30 is not an external change,
    so it can neither lock the zone nor be rewritten every evaluation."""
    agent, zone = agent_and_zone
    dev_id = zone.on_lights_dev_ids[0]
    make_device(dev_id, brightness=29, onState=True)
    zone.target_brightness = [{"dev_id": dev_id, "brightness": 30}]

    assert zone.has_brightness_changes() is False


def test_zone_still_sees_a_genuinely_stuck_dimmer(agent_and_zone):
    agent, zone = agent_and_zone
    dev_id = zone.on_lights_dev_ids[0]
    make_device(dev_id, brightness=0, onState=False)
    zone.target_brightness = [{"dev_id": dev_id, "brightness": 30}]

    assert zone.has_brightness_changes() is True


def test_zone_absorbs_a_group_dimmer_two_points_low(agent_and_zone):
    """Reproduces the production symptom: a z2m group dimmer commanded to
    30 reads back 28. Under the old flat band this locked the zone every
    cycle; the proportional band (target 30 -> band 3) absorbs it."""
    agent, zone = agent_and_zone
    dev_id = zone.on_lights_dev_ids[0]
    make_device(dev_id, brightness=28, onState=True)
    zone.target_brightness = [{"dev_id": dev_id, "brightness": 30}]

    assert zone.has_brightness_changes() is False


def test_zone_still_locks_on_a_genuine_external_raise(agent_and_zone):
    """A device reporting far above its target (someone raised it by hand)
    must still register as a change and be able to lock the zone."""
    agent, zone = agent_and_zone
    dev_id = zone.on_lights_dev_ids[0]
    make_device(dev_id, brightness=49, onState=True)
    zone.target_brightness = [{"dev_id": dev_id, "brightness": 30}]

    assert zone.has_brightness_changes() is True

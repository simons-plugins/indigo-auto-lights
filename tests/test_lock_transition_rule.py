"""The manual-override lock rule, one promise per test.

`has_lock_occurred(previous_dev, current_dev)` locks a zone only when THIS
event moved THIS device from at-target to off-target. Everything below is a
mutation test: each docstring names the promise and the specific wrong
implementation the test kills, because the two previous implementations of
this rule each looked fine against the happy-path suite.

  - The whole-zone LIVE read ("is anything off target right now?") lost a
    genuine override to a concurrent revert and self-locked whenever any
    OTHER device in the zone was still ramping.
  - Judging the NEW value alone ("is the device off target now?") locked the
    zone on every intermediate step of the plugin's own dimmer ramp.
"""

import json
import logging
from pathlib import Path

import pytest

import indigo
from auto_lights.auto_lights_config import AutoLightsConfig
from auto_lights.auto_lights_agent import AutoLightsAgent
from tests.helpers import (
    load_yaml,
    make_device,
    make_snapshot,
    settle_zone,
    suppress_device,
    target_for,
)


def _build(tmp_path, config_name):
    data = load_yaml(Path(__file__).parent / "configs" / config_name)
    config_json = {
        "plugin_config": data.get("plugin_config", {}),
        "lighting_periods": data.get("lighting_periods", []),
        "zones": data.get("zones", []),
    }
    conf_path = tmp_path / "conf.json"
    conf_path.write_text(json.dumps(config_json))
    cfg = AutoLightsConfig(str(conf_path))
    agent = AutoLightsAgent(cfg)
    zone = cfg.zones[0]
    for pres_id in zone.presence_dev_ids:
        make_device(pres_id, onState=True)
    for lum_id in zone.luminance_dev_ids:
        make_device(lum_id, sensorValue=0)
    return agent, zone


@pytest.fixture
def scenario1(tmp_path):
    """Single dimmer on-light (101), presence on, dark → target 100."""
    agent, zone = _build(tmp_path, "scenario1_presence_dark_adjust_false.yaml")
    dev = zone.on_lights_dev_ids[0]
    make_device(dev, brightness=0)
    return agent, zone, dev


@pytest.fixture
def multi_device(tmp_path):
    """Two dimmer on-lights (101, 102) plus an off-light (103)."""
    agent, zone = _build(tmp_path, "scenario_multi_device.yaml")
    for dev_id in zone.on_lights_dev_ids + zone.off_lights_dev_ids:
        make_device(dev_id, brightness=0, onState=False)
    return agent, zone


def _set_target(zone, dev_id, value):
    """Pin one device's target directly, bypassing the plan."""
    zone.target_brightness = [{"dev_id": dev_id, "brightness": value}]


def _fire(agent, dev, before, after, key="brightness"):
    """Fire one device-change event as a detached before/after snapshot pair."""
    previous = make_snapshot(dev, **{key: before})
    current = make_snapshot(dev, **{key: after})
    return agent.process_device_change(current, {key: after}, previous)


def _assert_settled(zone, dev, expected):
    assert (
        indigo.devices[dev].brightness == expected
    ), "precondition: device must be at target before the event"
    assert not zone.locked, "precondition: zone must start unlocked"


# ---------------------------------------------------------------- M2


def test_own_dimmer_ramp_never_locks(scenario1):
    """M2 — the plugin's own ramp must never lock the zone.

    Kills "the new value alone is off target, so lock". Every intermediate
    step of a ramp toward 100 is off target; only the previous state tells
    them apart from a manual move, and each step's previous state is also
    off target because Auto Lights only ever commands a device that is not
    at target.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    _assert_settled(zone, dev, 100)

    for before, after in ((0, 40), (40, 80), (80, 100)):
        _fire(agent, dev, before, after)
        assert not zone.locked, (
            f"the plugin's own ramp step {before}→{after} created a lock; "
            "the rule must look at the PREVIOUS state, not just the new one"
        )


# ---------------------------------------------------------------- M3


def test_late_reporter_never_locks(scenario1):
    """M3 — a device reporting late against an off-target history cannot lock.

    Kills "drop the previous-state check". The zone now wants 0; the device
    was at 30 (already off target) and reports 0, then 12. Neither event is a
    transition off target, so neither is a manual override.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    _set_target(zone, dev, 0)

    for before, after in ((30, 0), (30, 12)):
        _fire(agent, dev, before, after)
        assert not zone.locked, (
            f"a late report {before}→{after} against target 0 created a lock; "
            "the previous state was already off target, so nothing changed hands"
        )


# ---------------------------------------------------------------- M4


def test_turn_on_flash_never_locks(scenario1):
    """M4 — a lamp that flashes to full on the way to a dim target must not lock.

    Some drivers report 100 momentarily when switched on before settling to
    the commanded level. Previous state 0 is off target (target 30), so the
    flash is not a transition off target.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    _set_target(zone, dev, 30)

    _fire(agent, dev, 0, 100)

    assert not zone.locked, "a turn-on flash from an off-target state created a lock"


# ---------------------------------------------------------------- M5


def test_other_device_mid_write_does_not_lock_this_one(multi_device):
    """M5 — a sibling still mid-write must not lock the device that reported.

    Kills "restore the whole-zone live check". Device 102 is sitting at 40
    while its write is still in flight; device 101 reports its own settled
    value. A whole-zone live read sees 102 off target and locks the zone on
    101's event. The rule is per device, so it does not.
    """
    agent, zone = multi_device
    settle_zone(agent, zone)
    dev_a, dev_b = zone.on_lights_dev_ids[0], zone.on_lights_dev_ids[1]
    _assert_settled(zone, dev_a, 100)

    # Device 102 is mid-write: off its target, not yet settled.
    indigo.devices[dev_b].brightness = 40
    indigo.devices[dev_b].states["brightness"] = 40
    assert target_for(zone, dev_b) == 100

    _fire(agent, dev_a, 100, 100)

    assert not zone.locked, (
        "another device being mid-write locked the zone on an unrelated "
        "device's event — the rule must judge only the device that changed"
    )


def test_genuine_change_still_locks_while_sibling_mid_write(multi_device):
    """M5 sibling — the per-device rule must not go blind, only narrow.

    Same mid-write sibling, but this time device 101 genuinely moves off its
    own target. That must still lock.
    """
    agent, zone = multi_device
    settle_zone(agent, zone)
    dev_a, dev_b = zone.on_lights_dev_ids[0], zone.on_lights_dev_ids[1]
    _assert_settled(zone, dev_a, 100)

    indigo.devices[dev_b].brightness = 40
    indigo.devices[dev_b].states["brightness"] = 40

    _fire(agent, dev_a, 100, 30)

    assert zone.locked, "a genuine manual change on device 101 failed to lock"


# ---------------------------------------------------------------- M6


def test_manual_change_locks_even_while_checked_out(scenario1):
    """M6 — a manual change during the plugin's own write burst must lock.

    Kills "keep the `if self.checked_out: return False` guard". That guard
    existed to stop self-locking, which the transition rule now prevents by
    construction — but it also threw away real user input arriving exactly
    when the user is most likely to act, with the lights visibly moving.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    _assert_settled(zone, dev, 100)

    zone.check_out()
    try:
        _fire(agent, dev, 100, 30)
        assert zone.locked, (
            "a manual change that arrived while the zone was checked out was "
            "discarded; the checked_out guard must not gate this path"
        )
    finally:
        zone.check_in()


# ---------------------------------------------------------------- M7


def test_relay_turned_off_locks_and_turned_on_does_not(tmp_path):
    """M7 — relays go through the same rule via their bool target.

    Off→on from rest is the plugin's own work (previous state off target);
    on→off from the settled target is the user reaching for the switch.
    """
    agent, zone = _build(tmp_path, "scenario1_presence_dark_adjust_false.yaml")
    dev = zone.on_lights_dev_ids[0]
    make_device(dev, device_cls="relay", onState=False)

    settle_zone(agent, zone)
    assert target_for(zone, dev) is True
    assert indigo.devices[dev].onState is True
    assert not zone.locked

    previous = make_snapshot(dev, device_cls="relay", onState=True)
    current = make_snapshot(dev, device_cls="relay", onState=False)
    agent.process_device_change(current, {"onState": False}, previous)
    assert zone.locked, "switching a relay off from its target must lock the zone"

    # Reset and let the plugin turn it back on, then replay the echo.
    zone.reset_lock("test reset")
    settle_zone(agent, zone)
    previous = make_snapshot(dev, device_cls="relay", onState=False)
    current = make_snapshot(dev, device_cls="relay", onState=True)
    agent.process_device_change(current, {"onState": True}, previous)
    assert not zone.locked, "the plugin's own turn-on echo must not lock the zone"


# ---------------------------------------------------------------- M8


def test_tolerance_band_decides_what_counts_as_a_change(scenario1):
    """M8 — the lock rule uses the same tolerance band as the send path.

    Target 30 carries a band of 3 (max(1, ceil(30 * 0.10))). A report of 28
    is still at target — real zigbee dimmers read back a few points low — so
    it is not a change. A report of 20 is outside the band and is.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    _set_target(zone, dev, 30)
    indigo.devices[dev].brightness = 30
    indigo.devices[dev].states["brightness"] = 30
    assert not zone.locked

    _fire(agent, dev, 30, 28)
    assert not zone.locked, "a 2-point readback wobble inside the band created a lock"

    _fire(agent, dev, 30, 20)
    assert zone.locked, "a 10-point drop outside the band failed to create a lock"


# ---------------------------------------------------------------- M9


def test_no_previous_state_means_no_lock(scenario1, caplog):
    """M9 — an event with no previous state is not judged at all.

    Kills "fall back to judging the new value alone when previous_dev is
    None". Without a before-state there is no transition to see, and the
    fallback is exactly the mutation M2 rules out. The skip must be visible
    in the log rather than silent.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    _assert_settled(zone, dev, 100)

    current = make_snapshot(dev, brightness=30)
    with caplog.at_level(logging.DEBUG, logger="Plugin"):
        agent.process_device_change(current, {"brightness": 30}, None)

    assert not zone.locked, "an event with no previous state must not create a lock"
    assert any(
        "no previous state" in record.getMessage() for record in caplog.records
    ), "the skipped judgement must be logged, not silently swallowed"


# ---------------------------------------------------------------- M10


def test_excluded_device_does_not_lock(scenario1):
    """M10a — exclude_from_lock_dev_ids still short-circuits the rule."""
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    zone.exclude_from_lock_dev_ids = [dev]

    _fire(agent, dev, 100, 30)

    assert not zone.locked


def test_suppressed_device_does_not_lock(scenario1):
    """M10b — a suppressed device cannot lock the zone.

    A device over the failure threshold is already known-broken; its reports
    are noise, and letting them lock the zone would stall the healthy lights.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    # A suppressed device is, by definition, not sitting at its target — keep
    # the live state off target so note_device_event does not un-suppress it.
    indigo.devices[dev].brightness = 30
    indigo.devices[dev].states["brightness"] = 30
    suppress_device(agent, zone, dev)
    assert zone._is_device_suppressed(dev), "precondition: device must be suppressed"

    _fire(agent, dev, 100, 30)

    assert not zone.locked


def test_device_without_a_target_does_not_lock(scenario1):
    """M10c — no target entry for the device means nothing to compare against."""
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    # The setter rebuilds from a list, so clear the backing store directly.
    zone._target_brightness = []

    _fire(agent, dev, 100, 30)

    assert not zone.locked


def test_disabled_zone_does_not_lock(scenario1):
    """M10d — a disabled zone never locks."""
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    zone.enabled = False

    _fire(agent, dev, 100, 30)

    assert not zone.locked


# ---------------------------------------------------------------- M11


def test_target_moved_before_the_callback_is_an_accepted_miss(scenario1):
    """M11 — a target that moved inside the callback latency loses the lock.

    Documented, accepted gap. The zone re-planned to 60 (say, a lighting
    period boundary) between the user's change and this callback, so the
    previous state of 100 now reads as off target and condition 2 fails.
    The manual change is missed once and self-heals on the next change.
    Pinned here so that changing it is a deliberate decision, not a
    surprise: if this test starts failing, the gap was closed on purpose.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    _assert_settled(zone, dev, 100)

    zone.target_brightness = [{"dev_id": dev, "brightness": 60}]

    _fire(agent, dev, 100, 30)

    assert not zone.locked, (
        "accepted gap changed behaviour: a target that moved before the "
        "callback now locks the zone — update this test deliberately"
    )


# ---------------------------------------------------------------- degradation


class _UnreadableDimmer(indigo.DimmerDevice):
    """A dimmer whose brightness read fails, on demand.

    The raise is armed after construction because the stub Device's __init__
    reads brightness while building its states dict.
    """

    _explode = False

    @property
    def brightness(self):
        if self._explode:
            raise RuntimeError("brightness read failed")
        return self._brightness

    @brightness.setter
    def brightness(self, value):
        self._brightness = value


def test_unreadable_device_warns_and_does_not_lock(scenario1, caplog):
    """Degradation — a device the rule cannot evaluate must say so out loud.

    Returning "no lock" is the right action, but doing it silently turns a
    broken device into one that can never lock its zone and never explains
    why. The WARNING is the promise being tested; the missing lock alone
    would be indistinguishable from a normal at-target report.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    _assert_settled(zone, dev, 100)

    previous = make_snapshot(dev, brightness=100)
    current = make_snapshot(dev, device_cls=_UnreadableDimmer, brightness=30)
    current._explode = True

    with caplog.at_level(logging.WARNING, logger="Plugin"):
        agent.process_device_change(current, {"brightness": 30}, previous)

    assert not zone.locked, "an unreadable device must not create a lock"
    assert any(
        "cannot judge" in record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ), "an unreadable device must be reported, not silently ignored"

"""Zone re-evaluation is gated on the sensor reading, not on device activity.

`process_device_change` used to call `process_zone` for ANY update to a device
listed in a zone's `presence_dev_ids` or `luminance_dev_ids`. An Occupatum
occupancy device used as a presence sensor updates a `delay_timer` state and
its display string every ~1.2s while its off-delay counts down, so the Kitchen
zone was fully re-planned about once a second: each run re-applied the period
levels and reverted the user's manual dimmer change within a second, while the
load on the callback thread delayed the light's own change notification by
~10s. Neither presence on/off nor the light sensor value had changed.

Each test below names the promise it pins. "Must not re-evaluate" is asserted
by making `process_zone` fatal, and every such test first asserts a positive
precondition (the device really is in the zone's list) so it cannot pass
because the device was simply unknown to the zone.
"""

import json
from pathlib import Path

import pytest

from auto_lights.auto_lights_config import AutoLightsConfig
from auto_lights.auto_lights_agent import AutoLightsAgent
from tests.helpers import load_yaml, make_device, make_snapshot


def _build(tmp_path, config_name="scenario1_presence_dark_adjust_false.yaml"):
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
    make_device(zone.on_lights_dev_ids[0], brightness=0)
    return agent, zone


@pytest.fixture
def scenario1(tmp_path):
    """Presence device 301, luminance device 201, on-light 101."""
    agent, zone = _build(tmp_path)
    return agent, zone, zone.presence_dev_ids[0], zone.luminance_dev_ids[0]


def _forbid_process_zone(monkeypatch, agent):
    monkeypatch.setattr(
        agent,
        "process_zone",
        lambda *a, **k: pytest.fail("process_zone must not run for this event"),
    )


def _count_process_zone(monkeypatch, agent):
    """Replace process_zone with a counting stub; returns the mutable count."""
    calls = []

    def _stub(zone, clock=None):
        calls.append(zone)
        return True

    monkeypatch.setattr(agent, "process_zone", _stub)
    return calls


# ---------------------------------------------------------------- presence


def test_presence_countdown_tick_does_not_reevaluate(scenario1, monkeypatch):
    """Kills "re-evaluate on any update of a presence device" — the live bug.

    The Occupatum device rewrites its display string and delay timer roughly
    once a second while counting down. onState is unchanged across the event,
    so the zone has nothing new to act on.
    """
    agent, zone, pres_id, _ = scenario1
    assert zone._has_device(pres_id) == "presence_dev_ids"

    previous = make_snapshot(pres_id, onState=True)
    current = make_snapshot(pres_id, onState=True)
    _forbid_process_zone(monkeypatch, agent)

    agent.process_device_change(
        current,
        {"displayStateValUi": "Delay 298.7", "lastChanged": object()},
        previous,
    )


def test_presence_turning_on_reevaluates(scenario1, monkeypatch):
    """A real presence transition off→on must still re-evaluate the zone."""
    agent, zone, pres_id, _ = scenario1
    previous = make_snapshot(pres_id, onState=False)
    current = make_snapshot(pres_id, onState=True)
    calls = _count_process_zone(monkeypatch, agent)

    agent.process_device_change(current, {"onState": True}, previous)

    assert len(calls) == 1


def test_presence_turning_off_reevaluates(scenario1, monkeypatch):
    """And the reverse: on→off is the transition that turns the lights off."""
    agent, zone, pres_id, _ = scenario1
    previous = make_snapshot(pres_id, onState=True)
    current = make_snapshot(pres_id, onState=False)
    calls = _count_process_zone(monkeypatch, agent)

    agent.process_device_change(current, {"onState": False}, previous)

    assert len(calls) == 1


def test_presence_change_carried_only_in_states_onoffstate(scenario1, monkeypatch):
    """The gate must read the same states the zone reads, not just onState.

    `has_presence_detected()` ORs states["onState"] with states["onOffState"],
    so a custom device that only moves onOffState is still a real presence
    change. A gate that compared the onState attribute alone would swallow it
    and the lights would never come on for that device type.
    """
    agent, zone, pres_id, _ = scenario1
    previous = make_snapshot(pres_id, onState=True)
    current = make_snapshot(pres_id, onState=True)
    previous.states["onOffState"] = False
    current.states["onOffState"] = True
    assert previous.onState is True and current.onState is True
    calls = _count_process_zone(monkeypatch, agent)

    agent.process_device_change(current, {"onOffState": True}, previous)

    assert len(calls) == 1


# ---------------------------------------------------------------- luminance


def test_luminance_unchanged_does_not_reevaluate(scenario1, monkeypatch):
    """A luminance device that reports without moving its value changes nothing.

    Same shape as the presence bug: sensors re-report on a poll interval, and
    re-planning the zone on each report is what reverted manual changes.
    """
    agent, zone, _, lum_id = scenario1
    assert zone._has_device(lum_id) == "luminance_dev_ids"

    previous = make_snapshot(lum_id, sensorValue=30)
    current = make_snapshot(lum_id, sensorValue=30)
    _forbid_process_zone(monkeypatch, agent)

    agent.process_device_change(current, {"lastChanged": object()}, previous)


def test_luminance_change_reevaluates(scenario1, monkeypatch):
    """A genuine lux change is exactly what the zone needs to re-plan on."""
    agent, zone, _, lum_id = scenario1
    previous = make_snapshot(lum_id, sensorValue=30)
    current = make_snapshot(lum_id, sensorValue=65)
    calls = _count_process_zone(monkeypatch, agent)

    agent.process_device_change(current, {"sensorValue": 65}, previous)

    assert len(calls) == 1


# ---------------------------------------------------------------- degradation


def test_no_previous_snapshot_keeps_the_old_behaviour(scenario1, monkeypatch):
    """Documented degradation choice: no before-state means no gate.

    A caller that cannot supply `previous_dev` (the suppression retry path,
    older call sites, a manual re-check) has no transition to compare against.
    Suppressing re-evaluation there would be a lights-never-respond failure,
    which is far worse than an occasional redundant re-plan, so the gate
    deliberately does not apply.
    """
    agent, zone, pres_id, _ = scenario1
    current = make_snapshot(pres_id, onState=True)
    calls = _count_process_zone(monkeypatch, agent)

    agent.process_device_change(current, {"onState": True}, None)

    assert len(calls) == 1


# ---------------------------------------------------------------- ordering


def test_gate_is_per_event_not_per_device_history(scenario1, monkeypatch):
    """One suppressed tick must not desensitise the device to the next change.

    The gate judges a single before/after pair. A zone that stopped
    re-evaluating a device after its first no-op update would go permanently
    deaf to it — the failure mode the gate itself is most at risk of.
    """
    agent, zone, pres_id, _ = scenario1
    calls = _count_process_zone(monkeypatch, agent)

    # A countdown tick: no change to the presence reading.
    agent.process_device_change(
        make_snapshot(pres_id, onState=True),
        {"displayStateValUi": "Delay 298.7"},
        make_snapshot(pres_id, onState=True),
    )
    assert calls == []

    # Then a real transition on the same device.
    agent.process_device_change(
        make_snapshot(pres_id, onState=False),
        {"onState": False},
        make_snapshot(pres_id, onState=True),
    )
    assert len(calls) == 1

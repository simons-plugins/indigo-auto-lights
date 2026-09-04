"""The manual-override lock must survive a concurrent revert.

`has_lock_occurred()` used to decide by re-reading LIVE device state and
comparing it to target. In a busy zone a presence- or luminance-triggered
`process_zone` run can revert a manual change before the device-change handler
gets to look, so the live read says "at target" and the override becomes
invisible: no lock is created and the user's change is silently undone.

The failure is a race, so it is intermittent in the field — a quiet zone with
one presence sensor locks reliably while a busy zone with four presence sensors
and a chatty luminance sensor loses most of the time. That is precisely the
shape that a happy-path test blesses: a test that mutates the live device and
then fires the event leaves live state and event snapshot agreeing, so it
passes either way.

These tests force them to DISAGREE, which is the only state in which the bug is
observable. The rule under test judges the event's own before/after pair, so
the live device is deliberately left back at target.
"""

import json
from pathlib import Path

import pytest

import indigo
from auto_lights.auto_lights_config import AutoLightsConfig
from auto_lights.auto_lights_agent import AutoLightsAgent
from auto_lights import utils
from tests.helpers import (
    load_yaml,
    make_device,
    make_snapshot,
    settle_zone,
    target_for,
)


@pytest.fixture
def agent_and_zone(tmp_path):
    data = load_yaml(
        Path(__file__).parent / "configs" / "scenario1_presence_dark_adjust_false.yaml"
    )
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
    dev_id = zone.on_lights_dev_ids[0]
    make_device(dev_id, brightness=0)
    for pres_id in zone.presence_dev_ids:
        make_device(pres_id, onState=True)
    for lum_id in zone.luminance_dev_ids:
        make_device(lum_id, sensorValue=0)
    return agent, zone, dev_id


def _level(target):
    """The brightness that represents `target` on a stub dimmer."""
    if isinstance(target, bool):
        return 100 if target else 0
    return int(target)


def _manual_level(target):
    """A brightness that is unambiguously NOT the target."""
    return 0 if _level(target) > 50 else 100


def test_lock_created_even_when_live_state_already_reverted(agent_and_zone):
    """The override erases its own evidence — we must still lock.

    Live device is back AT target (a concurrent process_zone already reverted
    it); only the event's own before/after pair still carries the manual value.
    Deciding from live state finds nothing and silently drops the lock.
    """
    agent, zone, dev_id = agent_and_zone
    settle_zone(agent, zone)

    target = target_for(zone, dev_id)
    manual = _manual_level(target)

    previous = make_snapshot(dev_id, brightness=_level(target))
    current = make_snapshot(dev_id, brightness=manual)

    # The live device is at target: this is the revert having already landed.
    assert utils.is_device_at_target(
        indigo.devices[dev_id], target
    ), "precondition: live device must look at-target for this test to mean anything"
    assert utils.is_device_at_target(
        previous, target
    ), "precondition: the pre-change snapshot must be at target"
    assert not utils.is_device_at_target(
        current, target
    ), "precondition: the post-change snapshot must show the manual change"

    agent.process_device_change(current, {"brightness": manual}, previous)

    assert zone.locked, (
        "manual override was silently dropped: the lock decision read live state, "
        "which a concurrent revert had already restored to target"
    )


def test_no_lock_when_the_change_was_toward_target(agent_and_zone):
    """The inverse promise: a change that lands ON target must not lock.

    Guards against 'fixing' the race by locking on any device event at all,
    which would lock the zone every time Auto Lights adjusts a light itself.
    """
    agent, zone, dev_id = agent_and_zone
    settle_zone(agent, zone)

    target = target_for(zone, dev_id)
    level = _level(target)
    previous = make_snapshot(dev_id, brightness=level)
    current = make_snapshot(dev_id, brightness=level)

    agent.process_device_change(current, {"brightness": level}, previous)

    assert not zone.locked, "a change that landed ON target must not create a lock"


def test_excluded_device_still_never_locks(agent_and_zone):
    """exclude_from_lock_dev_ids must keep working on the new code path."""
    agent, zone, dev_id = agent_and_zone
    settle_zone(agent, zone)

    zone.exclude_from_lock_dev_ids = [dev_id]

    target = target_for(zone, dev_id)
    manual = _manual_level(target)
    previous = make_snapshot(dev_id, brightness=_level(target))
    current = make_snapshot(dev_id, brightness=manual)

    agent.process_device_change(current, {"brightness": manual}, previous)

    assert (
        not zone.locked
    ), "excluded device must not create a lock via the transition path"

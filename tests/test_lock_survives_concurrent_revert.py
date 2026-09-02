"""The manual-override lock must survive a concurrent revert.

`has_lock_occurred()` used to decide by re-reading LIVE device state and
comparing it to target. In a busy zone a presence- or luminance-triggered
`process_zone` run can revert a manual change before the device-change handler
gets to look, so the live read says "at target" and the override becomes
invisible: no lock is created and the user's change is silently undone.

The failure is a race, so it is intermittent in the field — a quiet zone with
one presence sensor locks reliably while a busy zone with four presence sensors
and a chatty luminance sensor loses most of the time. That is precisely the
shape that a happy-path test blesses: the existing
test_process_device_change_creates_new_lock mutates the device and then fires
the event, so live state and event snapshot agree and it passes either way.

These tests force them to DISAGREE, which is the only state in which the bug is
observable.
"""

import json
from pathlib import Path

import pytest

import indigo
from auto_lights.auto_lights_config import AutoLightsConfig
from auto_lights.auto_lights_agent import AutoLightsAgent
from auto_lights import utils
from tests.helpers import load_yaml, make_device


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
    return agent, zone, dev_id


def _target_for(zone, dev_id):
    for tgt in zone.target_brightness:
        if tgt["dev_id"] == dev_id:
            return tgt["brightness"]
    raise AssertionError(f"no target recorded for device {dev_id}")


def _make_snapshot(dev_id, brightness):
    """Build a device object representing the event's own snapshot.

    make_device() installs into indigo.devices, so we capture the live object
    first and put it back — leaving `snapshot` as a genuinely separate object.
    """
    live = indigo.devices[dev_id]
    snapshot = make_device(dev_id, brightness=brightness)
    indigo.devices[dev_id] = live
    return snapshot


def test_lock_created_even_when_live_state_already_reverted(agent_and_zone):
    """The override erases its own evidence — we must still lock.

    Live device is back AT target (a concurrent process_zone already reverted
    it); only the event snapshot still carries the manual value. Deciding from
    live state finds nothing and silently drops the lock.
    """
    agent, zone, dev_id = agent_and_zone
    agent.process_zone(zone)
    assert not zone.locked

    target = _target_for(zone, dev_id)
    # Pick a manual value that is unambiguously not the target.
    if isinstance(target, bool):
        manual = 0 if target else 100
    else:
        manual = 0 if int(target) > 50 else 100

    snapshot = _make_snapshot(dev_id, manual)

    # The live device is at target: this is the revert having already landed.
    assert utils.is_device_at_target(
        indigo.devices[dev_id], target
    ), "precondition: live device must look at-target for this test to mean anything"
    assert not utils.is_device_at_target(
        snapshot, target
    ), "precondition: the event snapshot must show the manual change"

    agent.process_device_change(snapshot, {"brightness": manual})

    assert zone.locked, (
        "manual override was silently dropped: the lock decision read live state, "
        "which a concurrent revert had already restored to target"
    )


def test_no_lock_when_the_change_was_toward_target(agent_and_zone):
    """The inverse promise: our OWN writes must not create a lock.

    Guards against 'fixing' the race by locking on any device event at all,
    which would lock the zone every time Auto Lights adjusts a light itself.
    """
    agent, zone, dev_id = agent_and_zone
    agent.process_zone(zone)
    assert not zone.locked

    target = _target_for(zone, dev_id)
    level = 100 if (target is True) else (0 if target is False else int(target))
    snapshot = _make_snapshot(dev_id, level)

    agent.process_device_change(snapshot, {"brightness": level})

    assert not zone.locked, "a change that landed ON target must not create a lock"


def test_excluded_device_still_never_locks(agent_and_zone):
    """exclude_from_lock_dev_ids must keep working on the new code path."""
    agent, zone, dev_id = agent_and_zone
    agent.process_zone(zone)
    assert not zone.locked

    zone.exclude_from_lock_dev_ids = [dev_id]

    target = _target_for(zone, dev_id)
    if isinstance(target, bool):
        manual = 0 if target else 100
    else:
        manual = 0 if int(target) > 50 else 100
    snapshot = _make_snapshot(dev_id, manual)

    agent.process_device_change(snapshot, {"brightness": manual})

    assert (
        not zone.locked
    ), "excluded device must not create a lock via the snapshot path"

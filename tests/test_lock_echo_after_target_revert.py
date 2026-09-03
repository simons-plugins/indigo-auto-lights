"""M12 — our own echo must not lock the zone after the target moves back.

The transition rule alone is not quite enough. The device is at 0, the zone
plans 100 and commands it; the writer confirms, checks the zone back in and
re-runs `process_zone`; presence or lux has moved in the meantime, so the plan
is now 0 again and 0 is commanded. The `deviceUpdated` for the *first* command
(0 → 100) is still queued. When it lands, its previous state of 0 sits exactly
on the new target and its new state of 100 does not — a textbook
at-target → off-target transition, and a spurious lock.

`Zone._recent_commands` closes it: a report landing on a value we commanded
within `COMMAND_ECHO_WINDOW_SECONDS` is our own echo, whatever the target has
done since. The window is what keeps that from becoming a blanket excuse — a
manual change to a value we commanded minutes ago still locks.
"""

import json
from pathlib import Path

import pytest

import indigo
from auto_lights.auto_lights_config import AutoLightsConfig
from auto_lights.auto_lights_agent import AutoLightsAgent
from auto_lights.zone import COMMAND_ECHO_WINDOW_SECONDS
from tests.helpers import load_yaml, make_device, make_snapshot, settle_zone


@pytest.fixture
def scenario1(tmp_path):
    """Single dimmer on-light (101), presence on, dark → target 100."""
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
    for pres_id in zone.presence_dev_ids:
        make_device(pres_id, onState=True)
    for lum_id in zone.luminance_dev_ids:
        make_device(lum_id, sensorValue=0)
    dev = zone.on_lights_dev_ids[0]
    make_device(dev, brightness=0)
    return agent, zone, dev


def _fire(agent, dev, before, after):
    previous = make_snapshot(dev, brightness=before)
    current = make_snapshot(dev, brightness=after)
    return agent.process_device_change(current, {"brightness": after}, previous)


def _replan_to_zero(zone, dev):
    """Model the re-plan that lands between our command and its echo."""
    zone.target_brightness = [{"dev_id": dev, "brightness": 0}]


def test_echo_of_our_own_command_does_not_lock_after_the_target_reverts(scenario1):
    """M12 — kills the mutation "drop the recent-command check".

    Without it the first assertion below fails: the queued 0 → 100 echo of our
    own write reads as a manual move away from the freshly re-planned target
    of 0. The second assertion is the other half of the promise — the excuse
    is for values we actually commanded, not for any report at all.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    assert indigo.devices[dev].brightness == 100

    commanded = [value for value, _ts in zone._recent_commands.get(dev, ())]
    assert 100 in commanded, (
        "precondition: settling the zone must record 100 as a recent command; "
        "without that record this test cannot exercise the echo path"
    )

    _replan_to_zero(zone, dev)

    _fire(agent, dev, 0, 100)
    assert not zone.locked, (
        "the queued echo of our own 100 locked the zone after the target "
        "reverted to 0; a report landing on a value we just commanded is ours"
    )

    _fire(agent, dev, 0, 60)
    assert zone.locked, (
        "a move to 60 — a value Auto Lights never commanded — failed to lock; "
        "the echo window must not excuse arbitrary values"
    )


def test_echo_excuse_expires_with_the_window(scenario1):
    """The excuse is time-boxed: an old command must not cover a new change.

    Same event as the first assertion above, but the recorded command is aged
    past `COMMAND_ECHO_WINDOW_SECONDS`. A user dialling a light to a level the
    plugin happened to use an hour ago is a manual override, not an echo.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)

    history = zone._recent_commands[dev]
    aged = [(value, ts - (COMMAND_ECHO_WINDOW_SECONDS + 1)) for value, ts in history]
    history.clear()
    history.extend(aged)

    _replan_to_zero(zone, dev)

    _fire(agent, dev, 0, 100)
    assert zone.locked, (
        "a command older than the echo window still excused a change; the "
        "window must expire or the rule stops detecting overrides entirely"
    )

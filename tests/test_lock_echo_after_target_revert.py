"""M12 — our own echo must not lock the zone after the target moves back.

The transition rule alone is not quite enough. The device is at 0, the zone
plans 100 and commands it; the writer confirms, checks the zone back in and
re-runs `process_zone`; presence or lux has moved in the meantime, so the plan
is now 0 again. The `deviceUpdated` for the *first* command (0 → 100) is still
queued. When it lands, its previous state of 0 sits exactly on the new target
and its new state of 100 does not — a textbook at-target → off-target
transition, and a spurious lock.

`Zone._recent_commands` closes it, and what it records is the state the device
was in when we commanded it, not the value we asked for. That is what the echo
transition STARTS from, whatever the device reports on the way out of it — a
dimmer ramping toward 100 reports 40 first, a value we never commanded.

Two things keep that from becoming a blanket excuse for real people:

  - each record excuses ONE transition and is then consumed, so a user who
    switches a light off, watches the zone put it back and switches it off
    again is not excused twice; and
  - the record expires after `COMMAND_ECHO_WINDOW_SECONDS`, which only has to
    cover the queued-echo race.
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

import indigo
from auto_lights.auto_lights_config import AutoLightsConfig
from auto_lights.auto_lights_agent import AutoLightsAgent
from auto_lights.zone import COMMAND_ECHO_WINDOW_SECONDS, Zone
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


def _recorded_states(zone, dev):
    """The pre-command states currently on record for one device."""
    return [pre for pre, _ts in zone._recent_commands.get(dev, ())]


def _set_presence(zone, present):
    for pres_id in zone.presence_dev_ids:
        make_device(pres_id, onState=present)


def _age_records(zone, dev, seconds):
    """Push every record for one device `seconds` further into the past."""
    history = zone._recent_commands[dev]
    aged = [(pre, ts - seconds) for pre, ts in history]
    history.clear()
    history.extend(aged)


def test_echo_of_our_own_command_does_not_lock_after_the_target_reverts(scenario1):
    """M12a — the queued echo is excused, and exactly once.

    Kills two mutations. "Match on the new value": the record here is the
    device's pre-command state of 0, and the report that has to be excused
    arrives as 0 → 100, so a rule comparing the NEW value against the record
    finds nothing and locks. "Do not consume the record": the second identical
    event is a user putting the light back to 100 after the zone dropped it to
    0, and a record left in place would excuse that one too, and the next, for
    the whole window.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    assert indigo.devices[dev].brightness == 100

    assert _recorded_states(zone, dev) == [0], (
        "precondition: settling the zone must record the state it commanded "
        "the device away from (0); without that record this test cannot "
        "exercise the echo path"
    )

    _replan_to_zero(zone, dev)

    _fire(agent, dev, 0, 100)
    assert not zone.locked, (
        "the queued echo of our own command locked the zone after the target "
        "reverted to 0; a transition starting from the state we commanded the "
        "device away from is ours"
    )

    _fire(agent, dev, 0, 100)
    assert zone.locked, (
        "the same transition was excused twice; one command excuses one "
        "transition, or a user fighting the zone is never heard"
    )


def test_first_ramp_step_is_excused_and_later_steps_need_no_excuse(scenario1):
    """M12b — a dimmer ramping out of the recorded state reports 40, not 100.

    Kills "match on the new value" from the other side: the excused report
    here is 0 → 40, a first ramp step toward the 100 we commanded, and 40 is a
    value Auto Lights never asked any device for. Only the state the
    transition STARTS from identifies it as ours. The later steps of the same
    ramp need no excuse at all — their previous state is already off target —
    which is why one record per command is enough.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    assert _recorded_states(zone, dev) == [0]

    _replan_to_zero(zone, dev)

    _fire(agent, dev, 0, 40)
    assert not zone.locked, (
        "the first step of our own ramp locked the zone after the target "
        "reverted to 0; the record is the pre-command state, not the "
        "commanded value, precisely so that a ramp step matches it"
    )
    assert (
        _recorded_states(zone, dev) == []
    ), "the ramp step must consume the record it matched"

    _fire(agent, dev, 40, 80)
    assert not zone.locked, (
        "a later ramp step locked the zone; with no record left it can only "
        "be excused by the ordinary rule — its previous state of 40 is "
        "already off the target of 0"
    )


def test_manual_move_onto_the_recorded_state_locks(scenario1):
    """M12c — the excuse is for transitions OUT of the recorded state.

    Kills "match the new state against the record". The zone commanded this
    light away from 0 and it is now settled at its target of 100. A user
    switching it off transitions 100 → 0, landing exactly on the recorded
    state — and must still lock. Comparing the new state instead of the
    previous one excuses it and the user loses the light.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)
    assert _recorded_states(zone, dev) == [0]

    _fire(agent, dev, 100, 0)
    assert zone.locked, (
        "switching the light off from its target failed to lock; the "
        "recorded state is where our command started, not where a change "
        "may end"
    )
    assert _recorded_states(zone, dev) == [
        0
    ], "a change that was not excused must not consume a record"


def test_flap_excuses_one_manual_change_and_the_window_bounds_it(scenario1):
    """M12d — the documented cost, and the two things that bound it.

    The reviewer's scenario: presence clears, the zone commands the light off
    (recording that it was at 100); presence returns, the zone commands it on
    again. The user now switches the light off — a 100 → 0 transition that
    starts on the state the OFF command recorded, so it is read as that
    command's echo and missed. That is the cost, and it is bounded twice
    over: the record expires with the window, and it is consumed by the one
    transition it excuses.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)

    _set_presence(zone, False)
    settle_zone(agent, zone)
    assert indigo.devices[dev].brightness == 0

    _set_presence(zone, True)
    settle_zone(agent, zone)
    assert indigo.devices[dev].brightness == 100

    history = zone._recent_commands[dev]
    fresh = list(history)
    assert 100 in [pre for pre, _ts in fresh], (
        "precondition: commanding the light off while it was at 100 must "
        "record 100 as a pre-command state"
    )

    # Bound 1 — the window. Outside it the same event is a plain override.
    assert COMMAND_ECHO_WINDOW_SECONDS <= 30, (
        "the window is the size of this cost. It only has to cover the "
        "queued-echo race — send_to_indigo()'s 2s settle plus callback "
        "latency — and every second beyond that is a second in which a real "
        "person moving a light off a state the plugin commanded it away from "
        "is silently ignored"
    )
    _age_records(zone, dev, COMMAND_ECHO_WINDOW_SECONDS + 1)
    _fire(agent, dev, 100, 0)
    assert zone.locked, (
        "an expired record still excused a manual change; the window is what "
        "keeps the flap scenario from swallowing overrides indefinitely"
    )

    # Inside the window the change IS missed — the accepted cost.
    zone.reset_lock("test: replay the same change inside the window")
    history.clear()
    history.extend(fresh)
    _fire(agent, dev, 100, 0)
    assert not zone.locked, (
        "documented cost: a manual change starting from a state we commanded "
        "the device away from inside the window is read as our echo"
    )

    # Bound 2 — consumption. The user's second attempt is heard.
    _fire(agent, dev, 100, 0)
    assert zone.locked, (
        "the second attempt was swallowed too; the cost is one missed change, "
        "not a two-minute window in which the user cannot be heard"
    )


def test_recording_a_command_never_costs_the_command(scenario1, caplog):
    """An unreadable device must lose its echo record, not its command.

    `_note_command` is called from inside the writer's try/except, one line
    before `send_to_indigo`. A read that raised out of it would abandon the
    send and be counted as a device failure — three of those and the device is
    suppressed, so bookkeeping for the lock rule would have switched a light
    off automation altogether. The device simply gets no record instead, and
    says so.
    """
    agent, zone, dev = scenario1

    with patch.object(
        Zone, "_normalize_dev_target_brightness", side_effect=RuntimeError("boom")
    ):
        with caplog.at_level(logging.ERROR, logger="Plugin"):
            zone._note_command(dev)

    assert not zone._recent_commands.get(dev), (
        "an unreadable pre-command state must leave no record; a wrong one "
        "would excuse a transition that was never ours"
    )
    assert any(
        "cannot read device" in record.getMessage() for record in caplog.records
    ), "a device that could not be read must say so, not fail silently"


def test_echo_excuse_expires_with_the_window(scenario1):
    """The excuse is time-boxed: an old command must not cover a new change.

    Same event as M12a's first assertion, but the record is aged past
    `COMMAND_ECHO_WINDOW_SECONDS`. A user moving a light off a state the
    plugin commanded it away from an hour ago is a manual override, not a
    queued echo.
    """
    agent, zone, dev = scenario1
    settle_zone(agent, zone)

    _age_records(zone, dev, COMMAND_ECHO_WINDOW_SECONDS + 1)

    _replan_to_zero(zone, dev)

    _fire(agent, dev, 0, 100)
    assert zone.locked, (
        "a command older than the echo window still excused a change; the "
        "window must expire or the rule stops detecting overrides entirely"
    )

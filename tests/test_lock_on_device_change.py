import json
from pathlib import Path

import pytest

import indigo
from auto_lights.auto_lights_config import AutoLightsConfig
from auto_lights.auto_lights_agent import AutoLightsAgent
from tests.helpers import load_yaml, make_device, make_snapshot, settle_zone


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
    # create the on-light device stub plus the sensors the zone reads, so the
    # zone computes a real "on" target rather than falling back to all-off.
    dev_id = zone.on_lights_dev_ids[0]
    make_device(dev_id, brightness=0)
    for pres_id in zone.presence_dev_ids:
        make_device(pres_id, onState=True)
    for lum_id in zone.luminance_dev_ids:
        make_device(lum_id, sensorValue=0)
    return agent, zone, dev_id


def test_process_device_change_creates_new_lock(agent_and_zone):
    """A manual dim from the settled target must lock the zone.

    The event carries a genuine before/after pair: previous AT target (100),
    current OFF target (50). The old version of this test mutated the one
    live stub object and handed it in as both sides of the event, so there
    was no transition to see — it passed against a whole-zone live read and
    would have passed against almost anything else too.
    """
    agent, zone, dev_id = agent_and_zone
    # Settle the zone so the device is genuinely at its target and unlocked.
    settle_zone(agent, zone)
    assert indigo.devices[dev_id].brightness == 100
    assert not zone.locked

    previous = make_snapshot(dev_id, brightness=100)
    current = make_snapshot(dev_id, brightness=50)

    agent.process_device_change(current, {"brightness": 50}, previous)

    assert zone.locked
    assert zone.name in agent._timers

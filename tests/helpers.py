def make_device(dev_id, device_cls="dimmer", **kwargs):
    """
    Create a dummy indigo.Device and insert into fake indigo.devices
    Supports: onState, brightness, sensorValue

    device_cls:
      - "dimmer" (default)
      - "relay"
      - "device"
      - a concrete Indigo stub class
    """
    import indigo

    if isinstance(device_cls, str):
        cls_map = {
            "device": indigo.Device,
            "dimmer": indigo.DimmerDevice,
            "relay": indigo.RelayDevice,
        }
        cls = cls_map[device_cls]
    else:
        cls = device_cls

    d = cls(
        dev_id,
        name=kwargs.get("name", ""),
        onState=kwargs.get("onState", False),
        brightness=kwargs.get("brightness", 0),
        sensorValue=kwargs.get("sensorValue", None),
    )
    # update any extra states
    for k, v in kwargs.items():
        if k not in ("name", "onState", "brightness", "sensorValue"):
            d.states[k] = v
    d.onOffState = d.onState
    d.states["onState"] = d.onState
    d.states["onOffState"] = d.onOffState
    d.states["brightness"] = d.brightness
    indigo.devices[dev_id] = d
    return d


def load_yaml(path):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def suppress_device(agent, zone, dev_id, count=None):
    """Seed the SuppressionManager with `count` consecutive failures.

    Defaults to the suppression threshold, so the device ends up suppressed.
    """
    from auto_lights.zone import MAX_CONSECUTIVE_FAILURES

    if count is None:
        count = MAX_CONSECUTIVE_FAILURES
    for _ in range(count):
        agent.suppression_manager.record_failure(dev_id, zone)


def device_fail_count(agent, dev_id):
    """Return the SuppressionManager's tracked failure count for a device."""
    entry = agent.suppression_manager._entries.get(dev_id)
    return entry.fail_count if entry else 0


def make_snapshot(dev_id, **kwargs):
    """Build a detached device object representing one event's own snapshot.

    make_device() installs into indigo.devices, so we capture the live object
    first and put it back — leaving the returned object as a genuinely
    separate stand-in for Indigo's origDev/newDev pair.
    """
    import indigo

    live = indigo.devices.get(dev_id)
    snapshot = make_device(dev_id, **kwargs)
    if live is not None:
        indigo.devices[dev_id] = live
    else:
        del indigo.devices[dev_id]
    return snapshot


def _apply_commanded_value(dev_id, desired):
    """Write a commanded target straight onto the stub device."""
    import indigo

    d = indigo.devices[dev_id]
    if isinstance(desired, bool):
        level = 100 if desired else 0
        on = desired
    else:
        level = int(desired)
        on = level > 0
    d.brightness = level
    d.states["brightness"] = level
    d.onState = on
    d.onOffState = on
    d.states["onState"] = on
    d.states["onOffState"] = on


def settle_zone(agent, zone, timeout=5.0):
    """Run the zone to completion with a send that always succeeds.

    Patches utils.send_to_indigo so the commanded value lands on the stub
    device (the stub indigo has no `indigo.dimmer`, so a real send would raise
    inside the writer thread and record failures), runs process_zone, then
    waits for the writer threads to finish. Leaves the zone at target and
    unlocked — the baseline every lock test needs before it can fire a
    meaningful event.

    Waiting for check-in is not enough on its own: each writer calls check_in()
    and *then* re-runs process_zone, so a test that returns on check-in can have
    its own setup (a re-planned target, a fired event) overtaken by that
    follow-up run. Waiting for the threads themselves to exit makes the zone
    genuinely quiescent when this returns.
    """
    import threading
    import time
    from unittest.mock import patch

    import indigo
    from auto_lights.utils import is_device_at_target

    def _send(dev_id, desired, **kwargs):
        _apply_commanded_value(dev_id, desired)
        return True

    pre_existing = set(threading.enumerate())

    with patch("auto_lights.utils.send_to_indigo", side_effect=_send):
        agent.process_zone(zone)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            writers = [
                t
                for t in threading.enumerate()
                if t not in pre_existing and t.is_alive()
            ]
            if not zone.checked_out and not writers:
                break
            time.sleep(0.02)

    assert not zone.checked_out, f"zone '{zone.name}' never checked back in"
    assert not zone.locked, f"zone '{zone.name}' locked while settling"
    for tgt in zone.target_brightness or []:
        dev_id = tgt["dev_id"]
        assert is_device_at_target(indigo.devices[dev_id], target_for(zone, dev_id)), (
            f"device {dev_id} did not land on its target while settling zone "
            f"'{zone.name}'; every lock test's baseline depends on this"
        )
    return zone


def target_for(zone, dev_id):
    """Return the zone's recorded target for one device."""
    for tgt in zone.target_brightness or []:
        if tgt["dev_id"] == dev_id:
            return tgt["brightness"]
    raise AssertionError(f"no target recorded for device {dev_id}")

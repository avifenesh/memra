#!/usr/bin/env python3
"""Resolve CUDA's logical device 0 to the physical GPU it names (memra#264 follow-up).

CUDA executables honor CUDA_VISIBLE_DEVICES. A release gate that queries nvidia-smi with a
hardcoded `-i 0` measures physical card 0 no matter which physical card a lease actually
selected for the engine to run on. An empty unselected card can mask insufficient headroom on
the selected card; a busy unselected card can falsely refuse an otherwise available selected
card. This module resolves logical device 0 (the only device an unqualified `-i 0` ever meant)
to its physical index and UUID, so the caller can query THAT card for headroom and tenants.

The parse/compare logic here takes device data as plain input (a list of dicts or the literal
text nvidia-smi would print) and never touches a GPU itself, so it is unit-tested without one.
An ambiguous or unresolvable selector raises ValueError; the caller's job is to refuse the
whole gate on that error, never to fall back to another card or skip the check.
"""

import argparse
import subprocess
import sys


class GpuResolutionError(ValueError):
    """Raised when CUDA's logical device 0 cannot be resolved to exactly one physical card."""


def parse_nvidia_smi_query(text):
    """Parse `nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits`.

    Returns a list of dicts: {"index": str, "uuid": str, "free_mib": int}, in the order
    nvidia-smi printed them (nvidia-smi's own device order, used when no selector narrows it).
    """
    devices = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise GpuResolutionError(f"malformed nvidia-smi device line: {line!r}")
        index, uuid, free = parts
        try:
            free_mib = int(free)
        except ValueError:
            raise GpuResolutionError(f"malformed free-memory field: {line!r}") from None
        devices.append({"index": index, "uuid": uuid, "free_mib": free_mib})
    return devices


def _is_uuid_selector(selector):
    upper = selector.upper()
    return upper.startswith("GPU-") or upper.startswith("MIG-")


def resolve_physical_device(devices, cuda_visible_devices, logical_index=0, cuda_device_order=None):
    """Resolve `logical_index` under `cuda_visible_devices` to its physical device record.

    `devices` is the full, unfiltered list from `parse_nvidia_smi_query` (or an equivalent
    stand-in), in nvidia-smi's own order (PCI bus order). `cuda_visible_devices` is the exact
    value of the CUDA_VISIBLE_DEVICES environment variable, or None if it is unset.
    `cuda_device_order` is the exact value of the CUDA_DEVICE_ORDER environment variable, or
    None if it is unset.

    Refuses (raises GpuResolutionError) rather than guessing when the variable is empty, mixes
    ordinal and UUID selectors, repeats a selector, names an unknown card, or leaves the
    requested logical index out of range. It never silently substitutes another card.

    ORDINAL ORDER IS NOT A GIVEN (revuto, PR #898). nvidia-smi always lists devices in PCI bus
    order. CUDA's own default numbering is FASTEST_FIRST, not PCI bus order, and an unset
    CUDA_VISIBLE_DEVICES or an ordinal selector is read in CUDA's order. On a host with more
    than one GPU model, CUDA's logical device 0 (or ordinal N) can therefore name a different
    physical card than nvidia-smi's index 0 (or index N) unless CUDA_DEVICE_ORDER is pinned to
    PCI_BUS_ID, the same pin tools/qualify-model-device-memory.py already sets next to
    CUDA_VISIBLE_DEVICES. A UUID selector names a physical card directly and is unaffected. So
    with more than one device present, an unset or ordinal selector REFUSES unless the caller
    has pinned CUDA_DEVICE_ORDER=PCI_BUS_ID; it never assumes the two orders agree.
    """
    if not devices:
        raise GpuResolutionError("no physical GPUs reported")

    by_index = {d["index"]: d for d in devices}
    by_uuid = {d["uuid"]: d for d in devices}

    if cuda_visible_devices is None:
        if len(devices) > 1 and cuda_device_order != "PCI_BUS_ID":
            raise GpuResolutionError(
                "CUDA_VISIBLE_DEVICES is unset with more than one GPU present: CUDA's default "
                "device order is FASTEST_FIRST, not nvidia-smi's PCI bus order, so logical "
                "device 0 cannot be assumed to be nvidia-smi index 0. Set "
                "CUDA_DEVICE_ORDER=PCI_BUS_ID or refuse."
            )
        visible = [d["index"] for d in devices]
    elif cuda_visible_devices == "":
        # CUDA's own contract: an explicitly empty selector means NO device is visible.
        raise GpuResolutionError(
            "CUDA_VISIBLE_DEVICES is set to the empty string: no GPU is visible"
        )
    else:
        selectors = [s.strip() for s in cuda_visible_devices.split(",") if s.strip() != ""]
        if not selectors:
            raise GpuResolutionError(
                f"CUDA_VISIBLE_DEVICES has no usable selectors: {cuda_visible_devices!r}"
            )
        uuid_flags = [_is_uuid_selector(s) for s in selectors]
        if len(devices) > 1 and not all(uuid_flags) and cuda_device_order != "PCI_BUS_ID":
            raise GpuResolutionError(
                f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r} uses an ordinal selector with "
                "more than one GPU present: CUDA reads ordinals in its own FASTEST_FIRST "
                "order, not nvidia-smi's PCI bus order, so the named ordinal cannot be "
                "assumed to match nvidia-smi's index of the same number. Set "
                "CUDA_DEVICE_ORDER=PCI_BUS_ID or refuse."
            )
        if any(uuid_flags) and not all(uuid_flags):
            raise GpuResolutionError(
                f"CUDA_VISIBLE_DEVICES mixes ordinal and UUID selectors: {cuda_visible_devices!r}"
            )
        visible = []
        for selector, is_uuid in zip(selectors, uuid_flags):
            if is_uuid:
                if selector not in by_uuid:
                    raise GpuResolutionError(
                        f"CUDA_VISIBLE_DEVICES names an unknown GPU UUID: {selector!r}"
                    )
                visible.append(by_uuid[selector]["index"])
            else:
                if selector not in by_index:
                    raise GpuResolutionError(
                        f"CUDA_VISIBLE_DEVICES names an unknown GPU ordinal: {selector!r}"
                    )
                visible.append(selector)
        if len(set(visible)) != len(visible):
            raise GpuResolutionError(
                f"CUDA_VISIBLE_DEVICES repeats a selector: {cuda_visible_devices!r}"
            )

    if logical_index < 0 or logical_index >= len(visible):
        raise GpuResolutionError(
            f"logical device {logical_index} is out of range for visible set {visible}"
        )
    return by_index[visible[logical_index]]


def query_nvidia_smi():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise GpuResolutionError(f"nvidia-smi query failed: {out.stderr.strip() or out.returncode}")
    return parse_nvidia_smi_query(out.stdout)


def main():
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-index", type=int, default=0)
    parser.add_argument(
        "--device-list-file",
        help="read nvidia-smi's own query text from this file instead of running nvidia-smi "
             "(testing / offline use only)",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="override CUDA_VISIBLE_DEVICES instead of reading it from the environment",
    )
    parser.add_argument(
        "--cuda-device-order",
        default=None,
        help="override CUDA_DEVICE_ORDER instead of reading it from the environment",
    )
    args = parser.parse_args()

    cvd = args.cuda_visible_devices if args.cuda_visible_devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    cdo = args.cuda_device_order if args.cuda_device_order is not None else os.environ.get("CUDA_DEVICE_ORDER")

    try:
        if args.device_list_file:
            with open(args.device_list_file, encoding="utf-8") as handle:
                devices = parse_nvidia_smi_query(handle.read())
        else:
            devices = query_nvidia_smi()
        device = resolve_physical_device(devices, cvd, args.logical_index, cdo)
    except GpuResolutionError as error:
        print(f"resolve-physical-gpu: refused: {error}", file=sys.stderr)
        return 1

    print(f"index={device['index']}")
    print(f"uuid={device['uuid']}")
    print(f"free_mib={device['free_mib']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Record measured hardware and keep local acceptance separate from official results."""
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import re
import subprocess
import statistics
import math

PROFILE_WARMUP = 80


def steady_samples(samples):
    if len(samples)<9 or any(not math.isfinite(value) or value<=0 for value in samples):
        return False
    width=len(samples)//3
    first,last=(statistics.median(v) for v in (samples[:width],samples[-width:]))
    return max(first,last)/min(first,last)<=1.05


def hardware_snapshot(soc):
    from tests.st.scripts.runtime import environment
    inventory_environment = environment()
    record = {"soc": soc, "hostname": platform.node(),
              "observed_at": datetime.now(timezone.utc).isoformat(),
              "cann_path": str(Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")).resolve())}
    try:
        inventory = subprocess.check_output(["npu-smi", "info"], text=True, timeout=10,
                                            env=inventory_environment, stderr=subprocess.STDOUT)
        cards = re.findall(r"\|\s*(\d+)\s+(910\S*)\s*\|", inventory)
        record["visible_cards"] = [{"physical_id": int(i), "sku": sku} for i, sku in cards]
        # A logical ACL device ID is not necessarily the physical npu-smi ID.
        # Do not guess the mapping on hosts exposing multiple cards.
        if len(cards) == 1:
            physical, sku = cards[0]
            common = subprocess.check_output(
                ["npu-smi", "info", "-t", "common", "-i", physical], text=True, timeout=10,
                env=inventory_environment, stderr=subprocess.STDOUT)
            record.update(physical_id=int(physical), sku=sku)
            for field, label in (("rated_core_mhz", "Aicore Freq(MHZ)"),
                                 ("current_core_mhz", "Aicore curFreq(MHZ)"),
                                 ("aicore_count", "Aicore Count")):
                match = re.search(re.escape(label) + r"\s*:\s*(\d+)", common)
                if match:
                    record[field] = int(match[1])
    except (OSError, subprocess.SubprocessError) as error:
        record["inventory_error"] = str(error)
    return record


def measured_soc(report):
    soc = report.get("environment", {}).get("soc") or report.get("execution_config", {}).get("soc")
    if not isinstance(soc, str) or not soc:
        raise ValueError("measured runtime SoC is missing; recapture the benchmark")
    return soc


def require_same_soc(reports):
    names = {measured_soc(report) for report in reports}
    if len(names) != 1:
        raise ValueError("performance comparison mixes hardware: " + ", ".join(sorted(names)))
    return names.pop()


def require_same_environment(reports):
    soc = require_same_soc(reports)
    fingerprints = set()
    for report in reports:
        environment = report.get("environment", {})
        if not environment.get("hostname"):
            raise ValueError("hardware provenance is missing; recapture the benchmark")
        fingerprints.add(tuple(environment.get(k) for k in
                               ("hostname", "rated_core_mhz", "aicore_count", "cann_path")))
    if len(fingerprints) != 1:
        raise ValueError("performance comparison mixes hosts, rated clocks or CANN installations")
    return soc


def official_scope(soc, expected="Ascend910B4"):
    return {"measured_soc": soc, "expected_official_soc": expected,
            "hardware_matches_official": soc == expected,
            "scope": "same-host Mock comparison only",
            "official_performance_verified": False}

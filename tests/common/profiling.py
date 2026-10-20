"""One device task parser for CANN and TileLang kernels."""
import csv, math
from pathlib import Path


def profile_samples(folder, warmup, repeats, backend="task", kernel_prefix="LinalgPinv"):
    rows = []
    if backend == "operator":
        samples = {}
        names = set()
        for path in sorted(Path(folder).rglob("OpBasicInfo*.csv")):
            with path.open(newline="") as stream:
                entries = list(csv.DictReader(stream))
            if len(entries) != 1:
                raise ValueError("operator sample must contain exactly one row")
            row = entries[0]
            name = row.get("Op Name", "")
            if not name.lower().startswith(kernel_prefix.lower()):
                raise ValueError("operator sample belongs to a different kernel")
            if not path.parent.name.isdecimal():
                raise ValueError("operator sample has no numeric launch index")
            index = int(path.parent.name)
            if index in samples:
                raise ValueError("duplicate operator launch index")
            samples[index] = row
            names.add(name)
        if set(samples) != set(range(warmup + repeats)) or len(names) != 1:
            raise ValueError("operator launches are missing or belong to different kernels")
        rows = [samples[index] for index in range(warmup + repeats)]
    elif backend == "task":
        for path in sorted(Path(folder).rglob("op_summary*.csv")):
            with path.open(newline="") as stream:
                rows.extend(row for row in csv.DictReader(stream)
                            if row.get("Op Name", "").lower().startswith(kernel_prefix.lower()))
        if not rows:
            for path in sorted(Path(folder).rglob("task_time*.csv")):
                with path.open(newline="") as stream:
                    rows.extend({**row, "Op Name": row["kernel_name"],
                                 "Task Duration(us)": row.get("task_time(us)", "")}
                                for row in csv.DictReader(stream)
                                if row.get("kernel_name", "").lower().startswith(kernel_prefix.lower())
                                and (row.get("kernel_type", "").startswith("KERNEL") or row.get("kernel_type", "") in ("MIX_AIC", "AI_VECTOR_CORE", "AI_CORE")))
    else:
        raise ValueError("unknown profiler backend")
    if len(rows) != warmup + repeats:
        raise ValueError(f"expected {warmup + repeats} {kernel_prefix} kernel samples, found {len(rows)}")
    aliases = (("Device_id", "Device ID"), ("stream_id", "Stream ID"), ("task_id", "Task ID"))
    identity_fields = [next((key for key in names if any(key in row for row in rows)), None) for names in aliases]
    if any(identity_fields):
        if not all(identity_fields) or any(not str(row.get(key, "")).strip() for key in identity_fields for row in rows):
            raise ValueError("device task identity is incomplete")
        identities = [tuple(row[key] for key in identity_fields) for row in rows]
        if len(set(identities)) != len(rows):
            raise ValueError("duplicate device task identity")
    if any("Task Duration(us)" not in row for row in rows):
        raise ValueError("kernel summary is missing its duration column")
    samples = [float(row["Task Duration(us)"]) for row in rows]
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("kernel duration must be finite and positive")
    return samples[warmup:]

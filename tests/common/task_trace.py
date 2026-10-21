"""Read complete operation Tasks and correlate lower-level scheduling records."""
import csv
import math
from pathlib import Path


def _table(folder, pattern, *, required):
    paths = list(Path(folder).rglob(pattern))
    if not paths and not required:
        return None
    if len(paths) != 1:
        raise ValueError("expected one " + pattern + " table")
    with paths[0].open(newline="") as stream:
        return list(csv.DictReader(stream))


def _identity(row, *, raw=False):
    fields = ("Device_id", "stream_id", "task_id") if raw else ("Device_id", "Stream ID", "Task ID")
    if any(not str(row.get(k, "")).strip() for k in fields):
        raise ValueError("complete device/stream/task identity is required")
    return tuple(str(row[k]).strip() for k in fields)


def _number(row, key, *, positive=False):
    value = float(row[key])
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("invalid trace timestamp or duration")
    return value


def complete_task_records(folder, *, kernel_prefix):
    """Never substitute raw MIX_AIV body duration for complete Task duration.

    Real captures can have ~1us before and ~1us after a MIX_AIV body.
    The summaries remain authoritative even when raw durations look faster.
    """
    summaries = _table(folder, "op_summary*.csv", required=True)
    raw = _table(folder, "task_time*.csv", required=False)
    raw_by_id = {}
    for row in raw or []:
        key = _identity(row, raw=True)
        if key in raw_by_id:
            raise ValueError("duplicate scheduling task identity")
        raw_by_id[key] = row
    records, seen = [], set()
    for row in summaries:
        name = row.get("Op Name", "")
        if not name.lower().startswith(kernel_prefix.lower()):
            continue
        key = _identity(row)
        if key in seen:
            raise ValueError("duplicate operation Task identity")
        seen.add(key)
        start = _number(row, "Task Start Time(us)")
        duration = _number(row, "Task Duration(us)", positive=True)
        record = {"identity": list(key), "name": name, "start_us": start,
                  "duration_us": duration, "duration_scope": "op_summary complete Task",
                  "task_type": row.get("Task Type")}
        if raw is not None:
            if key not in raw_by_id:
                raise ValueError("operation Task has no matching scheduling record")
            body = raw_by_id[key]
            body_name = body.get("kernel_name", "")
            if body.get("kernel_type") == "EVENT_RECORD" or (body_name != "N/A" and not body_name.lower().startswith(kernel_prefix.lower())):
                raise ValueError("operation Task matched an event or foreign kernel")
            body_start = _number(body, "task_start(us)")
            body_duration = _number(body, "task_time(us)", positive=True)
            prefix = body_start - start
            suffix = (start + duration) - (body_start + body_duration)
            # Exported epoch timestamps have sub-microsecond quantization.
            if min(prefix, suffix) < -0.5:
                raise ValueError("scheduling record lies outside the complete Task")
            record.update(raw_duration_us=body_duration, raw_task_type=body.get("kernel_type"),
                          prefix_us=prefix, suffix_us=suffix)
        records.append(record)
    if not records:
        raise ValueError("no matching complete Tasks")
    records.sort(key=lambda row: row["start_us"])
    return records

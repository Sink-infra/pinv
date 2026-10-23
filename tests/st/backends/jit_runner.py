"""TileLang runner implementing the same file/result protocol as ACLNN."""
import argparse
import ctypes as C
import fcntl
import json
import math
from pathlib import Path
from contextlib import nullcontext

import numpy as np

from code_dsl.op_host.geometry import geometry
from code_dsl.op_host.planner import make_plan
from code_dsl.op_host.schedule import launch_blocks
from tests.st.backends.acl import AclRuntime, bind_launch
from tests.st.backends.jit import UnsupportedStrategy, artifacts, select_strategy, verify
from tests.st.backends.task_profiler import TaskProfiler


def execute(args, report):
    shape = tuple(int(v) for v in args.shape.split(","))
    g = geometry(shape, atol=args.atol, rtol=None if args.rtol < 0 else args.rtol)
    count = math.prod(shape)
    if count > 1 << 28 or args.warmup < 0 or args.repeats < 1 or args.device < 0:
        raise ValueError("invalid tensor size, device or launch counts")
    data = Path(args.input).read_bytes()
    if len(data) != count * 4:
        raise ValueError("input byte count differs from shape")
    build_receipt = verify(args.build_dir)
    with open(f"/tmp/pinv-npu-device-{args.device}.lock", "a") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        runtime = AclRuntime(args.device)
        events = []
        try:
            report["soc"] = runtime.soc
            plan = make_plan(shape, atol=g.atol, rtol=g.rtol, soc=runtime.soc,
                             library=args.build_dir / "host-planner.so")
            strategy = select_strategy(plan)
            report.update(strategy=strategy, blocks=launch_blocks(plan, g.batch))
            report["provenance"] = build_receipt["libraries"][strategy]
            library_path, _ = artifacts(args.build_dir, strategy)
            library = C.CDLL(str(library_path.resolve()))
            blocks = report["blocks"]
            if strategy in {"peer_pipeline", "window_cube"}:
                if strategy=="peer_pipeline":
                    if blocks!=(g.batch*13+1)//2:
                        raise ValueError("invalid 13-AIV tile geometry")
                elif 2 * blocks % g.batch or 2 * blocks // g.batch < 3:
                    raise ValueError("invalid collective peer geometry")
                library.call.argtypes = [C.c_void_p] * 4 + [C.c_int32] * 5 + [C.c_float] * 2 + ([C.c_int32] if strategy=="window_cube" else []) + [C.c_void_p]
                library.call.restype = None
                def launch(buffers):
                    library.call(*buffers, blocks, g.n, g.batch, plan.fields["stride"], (plan.fields["local"] if strategy=="window_cube" else 0),
                                 g.atol, g.rtol, *([1] if strategy=="window_cube" else []), runtime.stream)
            else:
                call = bind_launch(library, module=True)
                def launch(buffers):
                    call(buffers, blocks, g.m, g.n, g.batch, g.atol, g.rtol, runtime.stream)
            a = np.frombuffer(data, dtype="<f4")
            buffers = [runtime.allocate(count, a), runtime.allocate(count),
                       runtime.allocate(plan.workspace_floats), runtime.allocate(plan.status_floats)]
            lib = runtime.lib
            lib.aclrtMemset.argtypes = [C.c_void_p, C.c_size_t, C.c_int32, C.c_size_t]
            lib.aclrtCreateEvent.argtypes = [C.POINTER(C.c_void_p)]
            lib.aclrtRecordEvent.argtypes = [C.c_void_p, C.c_void_p]
            lib.aclrtEventElapsedTime.argtypes = [C.POINTER(C.c_float), C.c_void_p, C.c_void_p]
            lib.aclrtDestroyEvent.argtypes = [C.c_void_p]
            for _ in range(2):
                event = C.c_void_p()
                runtime.check(lib.aclrtCreateEvent(C.byref(event)))
                events.append(event)
            def initialize():
                runtime.check(lib.aclrtMemset(buffers[1], count * 4, 255, count * 4))
            if args.output_init == "nan-once":
                initialize()
            samples = []
            with TaskProfiler(runtime, args.profile) if args.profile else nullcontext():
                for iteration in range(args.warmup + args.repeats):
                    if args.output_init == "nan":
                        initialize()
                    runtime.check(lib.aclrtRecordEvent(events[0], runtime.stream))
                    launch(buffers)
                    runtime.check(lib.aclrtRecordEvent(events[1], runtime.stream))
                    runtime.synchronize()
                    ms = C.c_float()
                    runtime.check(lib.aclrtEventElapsedTime(C.byref(ms), *events))
                    if iteration >= args.warmup:
                        samples.append(ms.value * 1000)
                    if iteration == args.warmup:
                        runtime.read(buffers[1], g.output_shape).astype("<f4").tofile(args.output)
                runtime.read(buffers[1], g.output_shape).astype("<f4").tofile(str(args.output) + ".final.bin")
            report.update(status="executed", event_micros=samples)
        finally:
            for event in events:
                code = runtime.lib.aclrtDestroyEvent(event)
                if code:
                    report["teardown"].append({"api": "aclrtDestroyEvent", "code": code})
            try:
                runtime.close()
            except RuntimeError as error:
                report["teardown"].append({"api": "ACL cleanup", "detail": str(error)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    for name in ("input", "output", "result"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output-init", choices=("runtime", "nan", "nan-once"), default="runtime")
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args(argv)
    report = {"status": "Harness Error", "backend": "jit", "output_init": args.output_init,
              "event_micros": [], "teardown": []}
    try:
        execute(args, report)
    except UnsupportedStrategy as error:
        report.update(status="Unsupported", detail=str(error))
    except RuntimeError as error:
        report.update(status="Runtime Error", detail=str(error))
    except (OSError, ValueError, KeyError) as error:
        report.update(status="Harness Error", detail=str(error))
    args.result.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "executed" and not report["teardown"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare identical input/reference files for all device backends."""
import hashlib
import numpy as np
from tests.common.cases import CASE_GROUPS
from tests.common.mock_cases import generate as generate_mock
from tests.common.tensor_io import _select_cases, generate_suite
from tests.st.scripts.gen_data import select_mock_cases


def prepare_cases(suite, names, destination):
    if suite == "mock":
        manifest = generate_mock(destination, select_mock_cases("success", names))
        prepared = []
        for case in manifest:
            arrays = {}
            for key in ("input", "reference"):
                arrays[key] = np.load(destination / case[key], allow_pickle=False)
                if hashlib.sha256(arrays[key].tobytes()).hexdigest() != case[key + "_sha256"]:
                    raise ValueError("Mock array digest differs")
            a, gold = arrays["input"], arrays["reference"]
            shape = tuple(case["shape"])
            if a.dtype != np.float32 or a.shape != shape or gold.dtype != np.float64:
                raise ValueError("Mock array declaration differs")
            if gold.shape != shape[:-2] + shape[-2:][::-1] or not np.isfinite(gold).all():
                raise ValueError("Mock reference shape or values differ")
            raw = destination / (case["name"] + ".bin")
            a.astype("<f4").tofile(raw)
            prepared.append({"name": case["name"], "shape": shape, "input": raw,
                             "atol": case["atol"], "rtol": case["rtol"], "gold": gold,
                             "output_init": case["output_init"]})
        return prepared
    cases = _select_cases(CASE_GROUPS[suite], "success", names)
    generate_suite(destination, cases)
    return [{"name": case.name, "shape": case.a_shape, "input": destination / case.name / "a.bin",
             "atol": case.atol, "rtol": -1.0 if case.rtol is None else case.rtol,
             "folder": destination / case.name} for case in cases]

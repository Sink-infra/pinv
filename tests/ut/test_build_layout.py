"""Source-layout integrity and shared Host/Device workspace arithmetic."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.common.tensor_io import store_text, store_bytes
from tests.st.scripts import build, runtime


class BuildLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.area = self.root / "build"
        self.code = self.root / "code"
        self.code.mkdir()
        store_text(self.code / "CMakeLists.txt", "project(fixture)\n")
        (self.code / "op_kernel").mkdir()
        store_text(self.code / "op_kernel/common.inc", "constexpr int fixture = 1;\n")
        self.runner = self.root / "runner.cpp"
        store_text(self.runner, "int main() { return 0; }\n")
        self.contexts = [
            patch.object(build, "ROOT", self.root),
            patch.object(runtime, "ROOT", self.root),
            patch.object(build, "RUNNER_SOURCE", self.runner),
        ]
        for context in self.contexts:
            context.start()
        source = self.area / "tests/source/code"
        shutil.copytree(self.code, source)
        artifact = self.area / "tests/linalg_pinv_runner"
        store_bytes(artifact, b"runner")
        store_bytes(self.area / "libcust_opapi.so", b"library")
        tiling = build.tiling_path(self.area)
        tiling.parent.mkdir(parents=True)
        store_bytes(tiling, b"tiling")
        self.receipt = {
            "source_sha256": runtime.source_digest(), "sdk": str(build.SDK),
            "compute_unit": "ascend910b",
            "library_sha256": runtime.sha256_file(self.area / "libcust_opapi.so"),
            "tiling_sha256": runtime.sha256_file(tiling),
            "runner_source_sha256": runtime.sha256_file(self.runner),
            "runner_sha256": runtime.sha256_file(artifact),
            "source_layout": "standalone",
            "layout_files": {p.relative_to(source).as_posix(): runtime.sha256_file(p)
                             for p in sorted(source.rglob("*")) if p.is_file()},
        }
        store_text(self.area / "tests/build.json", json.dumps(self.receipt))

    def tearDown(self):
        for context in reversed(self.contexts):
            context.stop()
        self.temporary.cleanup()

    def test_standalone_includes_nested_fragments(self):
        self.assertEqual(set(build.standalone_files()),
                         {"CMakeLists.txt", "op_kernel/common.inc"})
        self.assertEqual(build.verify_build(self.area), self.receipt)

    def test_changed_current_header_invalidates_the_receipt(self):
        store_text(self.code / "op_kernel/common.inc", "constexpr int fixture = 2;\n")
        with self.assertRaises(ValueError):
            build.verify_build(self.area)

    def test_missing_staged_header_invalidates_the_receipt(self):
        (self.area / "tests/source/code/op_kernel/common.inc").unlink()
        with self.assertRaises(ValueError):
            build.verify_build(self.area)

    def test_unrecorded_staged_file_invalidates_the_receipt(self):
        store_text(self.area / "tests/source/code/foreign.h", "foreign\n")
        with self.assertRaises(ValueError):
            build.verify_build(self.area)


class WorkspaceContractTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "a C++ compiler is required")
    def test_host_layout_covers_all_live_regions_and_cube_storage(self):
        source = r'''
#include "pinv_contract.h"
int main() {
    for (unsigned batch : {1u, 3u, 4u, 64u}) {
        for (unsigned cols = 1; cols <= 512; ++cols) {
            unsigned stride = ((2 * cols + 8 + 127) / 128) * 128;
            auto end = pinv::MetadataOffset(batch, cols, stride, batch);
            auto cube = pinv::CubeOffsetFloats(batch, cols, stride);
            if (cube < end || cube % pinv::kCubeAlignFloats) return 1;
            for (unsigned i = 0; i < batch; ++i) {
                auto begin = pinv::MetadataOffset(batch, cols, stride, i);
                if (begin + pinv::kMetadataFloats > cube) return 2;
            }
            auto exact = 2048ULL + 110ULL * cols * cols;
            if (pinv::CubeStorageBytes(cols) != exact) return 3;
            auto axis = pinv::RectAxis(cols);
            if (axis < cols || axis % 8 || (axis & (axis - 1))) return 4;
            for (unsigned rows : {1u, 15u, 65u, 129u, 512u}) {
                auto count = pinv::RectWorkspaceFloats(batch, rows, cols);
                auto width = pinv::RectWidth(rows, cols);
                if (width < rows + cols || width % 8) return 5;
                if (count != 2ULL * batch * cols * width) return 6;
                if (pinv::RectLocalFloats(rows, cols) * 4 > 128 * 1024) return 7;
            }
        }
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / "contract.cpp"
            store_text(path, "#include <initializer_list>\n" + source)
            executable = folder / "contract"
            subprocess.run(["c++", "-std=c++11", "-I" + str(runtime.ROOT / "code/op_kernel"),
                            str(path), "-o", str(executable)], check=True,
                           capture_output=True)
            subprocess.run([str(executable)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()

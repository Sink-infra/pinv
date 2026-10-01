# LinalgPinv 测试

测试按公共组件、系统测试数据和单元测试组织。各套件独立选择和生成。

| 套件 | 用例数 | 内容 |
| --- | ---: | --- |
| Coverage | 34 | 输入契约、Shape、数据类型、属性和边界，其中 18 个有效用例、16 个拒绝用例 |
| Correctness | 35 | 矩形、批量、秩亏、奇异值截断和数值构造 |
| Mock | 12 | 固定 FP32 矩阵及独立 FP64 参考 |

```text
tests/
├── common/                 # 用例加载、矩阵构造、FP64 参考和文件组件
├── st/
│   ├── cases/              # coverage、correctness、mock JSON 定义
│   ├── scripts/
│       ├── gen_data.py     # 套件选择与数据生成
│       ├── verify_result.py # 二进制输出校验
│       ├── build.py         # 当前源码与设备 runner 构建
│       ├── run_tests.py     # 实际执行、校验与事件计时
│       └── profile_mock_workloads.py # Mock 内核 Profiling
│   └── runner/             # 直接链接 ACLNN 的 C++ 设备程序
└── ut/                     # 用例、参考计算、文件完整性和命令行自检
```

## 环境与自检

在项目根目录执行：

```bash
python3 -m pip install -r tests/requirements.txt
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -m unittest discover -s tests/ut -v
python3 -m tests --details
```

单元测试检查输入契约、数学参考、生成文件和校验逻辑，不调用设备算子。
`python3 -m tests` 只读取用例定义，不依赖 NumPy。

## 用例选择与数据生成

```bash
python3 -m tests.st.scripts.gen_data --suite coverage --list
python3 -m tests.st.scripts.gen_data --suite coverage --expectation reject --list
python3 -m tests.st.scripts.gen_data --suite correctness \
  --case correctness_documented_full_rank --output build/tests/correctness
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -m tests.st.scripts.gen_data \
  --suite mock --output build/tests/mock
```

`--suite` 指定单个套件。`--case` 接受完整用例名，可重复指定；未知名称和跨套件名称返回状态码 2。
`--expectation` 可取 `all`、`success` 或 `reject`；Mock 均为有效输入。
生成目录必须不存在，生成完成后包含 `manifest.json`。

Coverage 和 Correctness 为每个用例生成独立目录：

- `a.bin`：按 C 顺序排列的原始输入。
- `reference_a_pinv_fp64.bin`：有效用例的 FP64 参考。
- `case.json`：Shape、类型、属性、预期行为和文件 SHA256。

拒绝用例保留原始输入及预期错误标签，不生成输出参考。
Mock 生成 `case_NN.npy` 和 `case_NN_reference.npy`，清单记录参数及数组内容摘要。
其固定构造参数位于 `st/cases/mock.json`，参考由实际 FP32 输入的 FP64 SVD 独立计算。

## 输出校验

Coverage 和 Correctness 的有效用例可用以下命令校验算子输出：

```bash
python3 -m tests.st.scripts.verify_result \
  --case-dir build/tests/correctness/correctness_documented_full_rank \
  --actual build/tests/actual.bin
```

输出必须为 C 顺序、小端 FP32 原始二进制，Shape 与输入的最后两维互换。
校验器检查清单、元数据和输入及参考的摘要，再逐元素检查
`abs(actual-reference) <= atol + rtol*abs(reference)`。
比较预算默认均为 `1e-4`，可指定更严格的预算。非有限输出和错误字节数不能通过。

返回状态码为：0 表示数值通过，1 表示数值不符，2 表示文件、参数或完整性错误。

## 数学约定

参考基于物化后的 FP32 输入计算，不以构造参数直接代替输入值。
伪逆仅保留严格大于 `max(atol, rtol * sigma_max)` 的奇异值。
默认相对阈值采用 `max(m, n) * eps(float32)`；正绝对阈值且未指定相对阈值时采用零相对阈值。
Coverage 和 Correctness 的有效构造同时检查保留与舍弃奇异值之间的间隔。

## NPU 构建与性能测试

在 Ascend Linux 主机的项目根目录执行。CANN SDK 默认位于
`/usr/local/Ascend/cann`，可通过 `PINV_CANN_PATH` 指定。
设备开发和测试使用 CANNJudge 的远程命令入口。

```bash
python3 -m tests.st.scripts.build --jobs 8
python3 -m tests.st.scripts.run_tests --suite coverage \
  --no-build --output build/tests/coverage-run
python3 -m tests.st.scripts.run_tests --suite correctness \
  --no-build --output build/tests/correctness-run
python3 -m tests.st.scripts.profile_mock_workloads \
  --no-build --output build/tests/mock-profile
```

`build.py` 构建当前 `code/`、本地 tiling 包和直接链接 ACLNN 的设备 runner。
默认构建目录为 `build/tests/npu`；`--build-dir` 可指定其他构建目录。
构建记录检查源码、SDK、算子库、tiling 库和 runner 的一致性。
`run_tests.py` 默认执行构建；`--no-build` 要求已有匹配的构建，不能复用不一致的产物。

设备入口使用 Coverage 与 Correctness 的有效用例，以及全部 Mock 用例。
`--case` 可重复指定完整名称。拒绝用例的输入契约由单元测试检查。
每个用例在独立进程中运行，同一设备上的测试通过文件锁串行执行。

`--device` 默认为 0，`--warmup` 默认为 3，`--repeats` 默认为 7。
预热样本不进入计时统计。runner 使用 ACL Event 记录算子执行区间，
汇总样本、均值、中位数和最小/最大值；主机文件操作及数据传输不在该区间内。
`--output-init auto` 按套件选择输出缓冲区策略：Mock 使用 `runtime`，
Coverage 和 Correctness 使用 `nan`。也可显式指定 `--output-init runtime`
或 `--output-init nan`。
`runtime` 保留设备运行时分配后的输出状态；`nan` 在每次调用前填充 NaN，
用于发现未赋值的输出位置。诊断填充在计时区间之外，结果文件记录实际使用的策略。

首个计时迭代和最后一个迭代的输出都必须通过数值校验。
数值结果保留错误元素数、总元素数和两位小数的错误率；通过状态以实际错误元素数为准。

`run_tests.py --profile` 同时采集内核 Profiling。
`profile_mock_workloads.py` 是 Mock 的 Profiling 入口，接受相同运行参数。
导出的 LinalgPinv 内核样本数必须等于预热数与重复数之和，舍弃预热后统计内核耗时。
事件耗时与内核耗时分别记录，不混用两种计时口径。

结果目录必须不存在。其中保存输入与参考、实际输出、命令、运行日志、
逐用例结果以及套件 `result.json`；Profiling 模式还保存原始采集和导出 CSV。
所有运行产物均位于 `build/` 等结果目录中，不属于测试套件源码。

运行结果保留 `Pass`、`Wrong Answer`、`Runtime Error`、`TLE` 和
`Harness Error`，运行时错误包含实际 API 和错误码。
套件返回 0 表示全部用例通过，1 表示存在执行、数值或采集失败，2 表示参数或构建错误。

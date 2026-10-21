# LinalgPinv 测试

测试按公共组件、系统测试数据和单元测试组织。各套件独立选择和生成。

| 套件 | 用例数 | 内容 |
| --- | ---: | --- |
| Coverage | 34 | 输入契约、Shape、数据类型、属性和边界，其中 18 个有效用例、16 个拒绝用例 |
| Correctness | 43 | 矩形、批量、秩亏、奇异值截断和数值构造 |
| Mock | 12 | 固定 FP32 矩阵及独立 FP64 参考 |

```text
tests/
├── common/                 # 用例加载、矩阵构造、FP64 参考和文件组件
├── st/
│   ├── cases/              # coverage、correctness、mock JSON 定义
│   ├── backends/            # CANN/ACLNN 与 TileLang JIT 后端及设备传输
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

Mock 数据构造使用 NumPy 1.26.4 和 2 个 BLAS 线程；设备测试命令使用同一环境。
单元测试检查输入契约、数学参考、生成文件和校验逻辑，不调用设备算子。
Correctness 包含条件数约 95 的双向矩形、低秩小尺度矩形、显式截断、混合批量秩与尺度，
以及独立的小尺度矩形。原生路径和后验误差界见 [原生数值说明](../docs/cann/NUMERICS.md)，
当前 DSL 生成路径的实现与验证见 [DSL 说明](../code_dsl/README.md)。
`python3 -m tests` 只读取用例定义，不依赖 NumPy。

## 统一后端与配对比较

统一入口 `python3 -m tests run` 的 `--backend ascend|jit|both` 选择 CANN/ACLNN、真实 TileLang JIT，或两者配对。
两条路径共用 `tests/common/workloads.py` 的用例生成、`observations.py` 的逐元素比较、
首末输出检查、`profiling.py` 的设备采样解析及同一份结果协议。
`both` 只生成一份输入和 FP64 参考，先运行 JIT，再运行 AscendC；不会为后端分别生成数据。

```bash
python3 -m tests build --backend jit --build-dir build/tests/backends/jit
python3 -m tests build --backend ascend --build-dir build/tests/backends/ascend
python3 -m tests run --backend jit --suite correctness \
  --build-dir build/tests/backends/jit --no-build --allow-unsupported \
  --transport npu-session --output build/tests/jit-correctness
python3 -m tests run --backend both --suite mock \
  --build-dir build/tests/backends --no-build --allow-unsupported \
  --processes 3 --warmup 1000 --repeats 9 --profile --profiler-backend task \
  --transport npu-session --output build/tests/paired-mock
```

Ascend 后端仅需 NumPy 和 CANN；JIT 构建额外需要项目使用的 tilelang-ascend
`0.1.1.010+e909216995c2acc527bdb252039ec93a920c052c`。
JIT 构建器编译 compact/local/rectangular 模块、带完整数值继续路径的 peer 模块，以及只处理
形状和启动参数的 host planner；另含完整窗口恢复模块。运行时加载真实 JIT 动态库，使用实际 SoC 的调度数据。
`--no-build` 校验 DSL、生成源码、动态库、planner、测试执行代码和 SDK 的身份。
中断后的 JIT 构建可用 `build --backend jit --resume` 重试；仅复用已校验的当前 DSL 模块，重新构建 planner。
JIT 默认构建位置为 `build/tests/jit`，Ascend 默认位置为 `build/tests/npu`；`both` 使用
`build/tests/backends/jit` 和 `build/tests/backends/ascend`。

当前 JIT 支持全部 12 个 Mock；尚未移植的非 Mock 配置仍报告 `Unsupported`，不会暗中改用 Ascend 后端。
默认遇到不支持的配置会使套件返回失败；`--allow-unsupported` 可运行已支持范围，但报告仍保留
`complete=false`、`passed=false` 和不支持用例名，仅 `scoped_passed` 表示已支持范围是否通过。
不能把阶段检查当作完整移植验收。

`--processes N` 为每例启动 N 个独立子进程。`both` 输出每个后端的 `result.json` 和根目录的
`comparison.json`，逐例校验输入/参考摘要、属性、输出初始化策略及实际硬件环境相同。
性能比较只使用内核 Task Duration，先取进程中位数，再跨进程取中位数；ACL Event 指标单独保留。
默认每例最多减速 5%，须至少三进程、每进程至少九项稳定样本，才能标为性能验收通过。
缺失策略、非稳态、错误输出和单进程采样都不能获得完整性能通过状态。

共享 NPU 服务环境使用 `--transport npu-session`，所有设备子进程和 profiler 均经过
`NpuSession.run`，无需修改服务或全局替换 `subprocess.run`。独占设备环境可使用默认的
`--transport direct`；两种传输使用同一套后端与结果处理。

日常正确性与配对性能测试使用上述统一入口。专用诊断命令位于 `tests/st/scripts/`，
可继续读取旧格式回执；[DSL 导出器](../code_dsl/README.md#导出与原生模板)与测试后端
通过 `code_dsl/source.py` 共用源码摘要，并使用 `tests/common/provenance.py` 校验 JIT 验收。
构建、采样、CSV 和历史回执均为运行产物，放入已忽略的 `build/` 或 `code_dsl/results/`。
`code/project_manifest.json` 是被 Git 忽略的本地转换清单，构建摘要、源码拷贝和产物校验均不依赖它。
旧构建回执需要重新生成，以使用当前源码与测试逻辑的身份校验。

## 本地与官方收益校准

每轮优化先相对**当前保留版本**验证 JIT 的新增收益，再导出并验证对应 CANN 包的新增收益。
两条路径必须在同一 Mock Case 上均获得稳定收益；每个进程的中位数都要优于对应基线。
JIT 快而 CANN 包不快、不同 Case 分别变快、历史 CAC 收益或正确性通过，都不能替代这一条件。
所有十二个 Mock 都要通过并逐项报告，不能用同形状的一个 fixture 外推另一个。

```bash
python3 -m tests.st.scripts.accept_optimization --stage jit \
  --baseline-jit build/retained/jit/result.json --candidate-jit build/candidate/jit/result.json \
  --output build/candidate/new-jit-benefit.json
# JIT 新增收益通过后，才进行转换、构建及 CANN 包 Mock 采样。
python3 -m tests.st.scripts.accept_optimization --stage package \
  --jit-acceptance build/candidate/new-jit-benefit.json \
  --baseline-jit build/retained/jit/result.json --candidate-jit build/candidate/jit/result.json \
  --baseline-cann build/retained/ascend/result.json --candidate-cann build/candidate/ascend/result.json \
  --baseline-manifest build/retained/project_manifest.json --candidate-manifest code/project_manifest.json \
  --output build/candidate/new-package-benefit.json
```

该入口核对原始采样与汇总一致性、输入与计时协议、三进程完整 Task，以及转换清单中的
DSL/JIT 源码和二进制摘要与 CANN 构建回执的源码清单。清单及验收报告仍是被忽略的运行产物。
这项验收与 CAC 提交门槛分别满足后才提交官方测试；官方结果返回前不宣称收益已兑现。

每轮官方结果返回后，逐 Case 比较相对于同一基线的本地和官方耗时降幅，使用绝对耗时，
不使用分数或 TBest。正确性、提交资格、本轮新增收益分别验收；历史收益不能为本轮退化背书。

```bash
python3 -m tests.st.scripts.audit_official_gains \
  --baseline-local build/baseline/result.json --candidate-local build/candidate/result.json \
  --baseline-official build/official-baseline.json --candidate-official build/official-candidate.json \
  --output build/calibration/comparison.json
```

本地 A/B 必须使用同一输入、属性、硬件与完整 Task 口径；每例至少三个独立进程，
每进程至少三个样本，小 Case（矩阵维度不超过 64）至少五个样本。报告保留进程中位数、
样本范围和 MAD。官方输入无需与本地相同，也不能据此声称复现了官方张量。
报告绑定四份原始回执的路径和摘要；缺少采样、非稳态和降幅偏差超过 3 个百分点均返回非零状态。

这一规则同样用于复查历史优化。每个超限项保持开放，直到有绑定源码、产物与计时协议的
本地复现实验、原因证据、修复及修复后的验证。恢复旧路径或另一 Case 提速不能关闭问题。
尚未完成这些步骤时，不将该路径的本地收益作为已经兑现的优化推进。

原型与统一入口必须读取同一计时域。`tests.common.task_trace.complete_task_records` 按
device/stream/task 标识关联 `op_summary` 和底层 `task_time`，始终返回前者的完整 Task 时长。
混合 Vector 内核的底层记录可能漏掉启动前后各约 1 μs；不能把这段内核主体时间标成完整 Task。
此类原型的旧收益必须从保留的完整 Task 表重新计算，不能用硬编码补时修正。

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
Mock 生成 `case_NN.npy` 和 `case_NN_reference.npy`，清单记录参数、输出初始状态及数组内容摘要。
其固定构造参数位于 `st/cases/mock.json`，包含条件数、逆矩阵稀疏约束、块谱和 FP32 元素约束。
块谱构造使用对称正定的逆矩阵分块及稠密耦合项。参考由实际 FP32 输入的 FP64 SVD 独立计算。
`batch_similarity_diagonals` 为每个批次指定正对角矩阵 D，输入矩阵按 D A D⁻¹ 构造。

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
python3 -m tests build --jobs 8
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -m tests run --suite coverage \
  --no-build --output build/tests/coverage-run
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -m tests run --suite correctness \
  --no-build --output build/tests/correctness-run
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -m tests.st.scripts.profile_mock_workloads \
  --no-build --output build/tests/mock-profile
```

`build.py` 构建当前 `code/`、本地 tiling 包和直接链接 ACLNN 的设备 runner。
当前 `code/` 包含已导出的 AscendC 源码，测试构建与运行无需安装 TileLang，
也无需预先运行 DSL 导出脚本；所需的项目头文件均随 `code/` 提供。
默认构建目录为 `build/tests/npu`；`--build-dir` 可指定其他构建目录。
`build.py --source-layout project` 构建完整工程。
`--source-layout standalone` 将完整 `code/` 文件集合复制到独立源码目录后构建，
包括公共头文件、实现片段和项目清单，用于检查源码的自包含性。
构建目录中的 `tests/source.json` 记录布局与实际参与构建的文件摘要。
构建记录检查源码、SDK、算子库、tiling 库和 runner 的一致性。
`run_tests.py` 默认执行构建；`--no-build` 要求已有匹配的构建，不能复用不一致的产物。

设备入口使用 Coverage 与 Correctness 的有效用例，以及全部 Mock 用例。
`--case` 可重复指定完整名称。拒绝用例的输入契约由单元测试检查。
每个用例在独立进程中运行。设备 runner 持有同一设备的独占文件锁，
覆盖设备初始化、执行和资源清理；采样器的主机准备和文件解析可并行。

`--device` 默认为 0，`--warmup` 默认为 3，`--repeats` 默认为 9。
预热样本不进入计时统计。runner 使用 ACL Event 记录算子执行区间，
汇总样本、均值、中位数和最小/最大值；主机文件操作及数据传输不在该区间内。
`--output-init auto` 按套件和用例声明选择输出缓冲区策略。
Mock 默认使用 `runtime`，`mock_05`、`mock_08` 和 `mock_11` 声明 `nan-once`；
Coverage 和 Correctness 使用 `nan`。
也可显式指定 `--output-init runtime`、`--output-init nan` 或 `--output-init nan-once`。
`runtime` 保留设备运行时分配后的输出状态；`nan-once` 在分配后填充一次 NaN，
保留后续调用产生的状态；`nan` 在每次调用前填充 NaN，用于检查每次调用的完整赋值。
填充在计时区间之外，逐例结果记录实际使用的策略。输出初始状态不参与输入或参考计算。

首个计时迭代和最后一个迭代的输出都必须通过数值校验。
数值结果保留错误元素数、总元素数和两位小数的错误率；通过状态以实际错误元素数为准。

`run_tests.py --profile` 同时采集内核 Profiling。
`profile_mock_workloads.py` 是 Mock 的 Profiling 入口，接受相同运行参数。
`--profiler-backend operator` 为默认采样方式，使用 `msprof op` 采集每次调用的
`OpBasicInfo`，读取 `Task Duration(us)`。`--profiler-replay kernel` 为默认重放方式，
也支持 `--profiler-replay application`；结果记录实际重放方式。
`--profiler-warmup` 默认为 5，设置每次内核采样前的重放预热次数；取 0 可关闭。
设备程序的 `--warmup` 与采样器的重放预热分别记录。
采样器跳过设备程序的预热调用，默认采集全部正式迭代；`--profile-samples N`
可选择预热后的前 N 个正式迭代，N 不超过 `--repeats`。
每条记录必须对应唯一的采样启动序号和相同的 LinalgPinv 内核，样本数必须与请求一致；
按启动序号排序后统计内核耗时。
`--profiler-backend task` 使用 ACL 任务级采样并导出 `op_summary` 或 `task_time`，
该模式保留预热与正式调用记录，舍弃预热后统计，样本数必须等于两者之和。
缺少样本不能通过采集检查。
事件耗时与内核耗时分别记录，不混用两种计时口径。operator 采样期间的事件区间
可能包含采样器的重放和主机侧调度等待，性能比较使用导出的内核耗时。

内核统计中的 `kernel.mean_us` 为全部采样的算术平均值，
`kernel.median_us` 为中位数；原始样本和最小/最大值分别记录。
冷启动测量在设备空闲时运行，稳态测量使用较长的重放预热。
两种采样配置使用独立结果目录：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -m tests.st.scripts.profile_mock_workloads \
  --no-build --warmup 0 --repeats 5 --profiler-warmup 0 \
  --output build/tests/mock-startup-profile
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -m tests.st.scripts.profile_mock_workloads \
  --no-build --warmup 3 --repeats 5 --profiler-warmup 100 \
  --output build/tests/mock-steady-profile
```

结果目录必须不存在。其中保存输入与参考、实际输出、命令、运行日志、
逐用例结果以及套件 `result.json`；Profiling 模式还保存实际采样方式、
原始采集和导出 CSV。operator 模式同时保留设备程序的原始命令。
所有运行产物均位于 `build/` 等结果目录中，不属于测试套件源码。

运行结果保留 `Pass`、`Wrong Answer`、`Runtime Error`、`TLE` 和
`Harness Error`，运行时错误包含实际 API 和错误码。
套件返回 0 表示全部用例通过，1 表示存在执行、数值或采集失败，2 表示参数或构建错误。

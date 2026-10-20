# pinv TileLang 实现

`op_kernel/` 用 TileLang 表达分解、重建、截断、访存、布局与同步；`op_host/` 只处理形状和启动调度。
独立 AscendC 工程位于 `../code/`，原目录迁移备份为 `../backup/`。形状、批量、步长和阈值通过运行时参数传入，
共享策略为 compact、local、rectangular、peer pipeline 和 window Cube。转换会生成一份本地清单
`code/project_manifest.json`，记录来源和文件摘要；该产物被 Git 忽略。

正确性与性能测试由 [tests](../tests/README.md) 统一管理。`tests/st/backends/` 实现后端构建身份、设备启动、
ACL 资源和传输；`tests/common/` 提供共同的数据、参考、输出校验、硬件记录和采样解析。
构建、运行、采样和验收命令均由 `tests/` 提供；DSL 导出器位于 `code_dsl/convert/`，
通过 `code_dsl/source.py` 与测试后端共用源码身份校验。

```bash
python3 -m tests build --backend jit --build-dir build/tests/backends/jit
python3 -m tests build --backend ascend --build-dir build/tests/backends/ascend
python3 -m tests run --backend both --suite mock \
  --build-dir build/tests/backends --no-build --allow-unsupported \
  --transport npu-session --processes 3 --warmup 1000 --repeats 9 \
  --profile --profiler-backend task --output build/tests/paired-mock
```

当前 JIT 覆盖全部 12 个 Mock 路径，256 阶窗口策略包含整数分解残差、Cube 修正和误差验收。
尚未移植的非 Mock 配置仍明确报告 `Unsupported`，完整 DSL 移植尚未完成。
高精度消元和 Jacobi 继续由 DSL 生成，快速路径拒绝时调用；数值计算没有 CPU 回退。
compact 使用四分之一 AIV 的共享调度策略，local 沿用原生调度。
窗口策略拒绝快速结果时，独立 JIT 使用 DSL Jacobi 继续计算；导出的 CANN 包保留原生数值回退，
避免串行 DSL Jacobi 在大矩阵上的高开销。两者都保留阈值截断。

## 导出与原生模板

`convert/window_export.py` 将全部 Mock 路径经过验证的实际 JIT 源码导出为 CANN 工程；它只适配声明、启动元数据和 include，
不手改求解器。导出仍要求对应 DSL 版本已通过先行 JIT 门槛，不能使用旧版本的通过回执。
转换专用的 CANN 启动适配文本也保存在 `convert/`；`op_host/schedule.py` 只负责 JIT 运行时调度。
模块整理会改变 DSL 源码摘要，后续转换仍须使用对应当前摘要的 JIT 验收记录。

原生模板固定为 Git 提交 `ec5ec7a2b5c9555a46636e08d25d9d05f46c0df2` 的 `code/`，
通过 `code_dsl/convert/native_source.py` 按需提取至临时目录，不再维护 `code_backup/` 副本。
该提交必须存在于本地 Git 对象库。`--native` 也可显式指定用户准备的原始源码目录。
数值设计说明保存在 [docs/cann/NUMERICS.md](../docs/cann/NUMERICS.md)。

```bash
python3 -m code_dsl.convert.window_export \
  --module-build build/tests/backends/jit/module --peer-build build/tests/backends/jit/peer \
  --window-build build/tests/backends/jit/window \
  --acceptance build/jit-validation/acceptance.json --output code \
  --framework-common /path/to/tilelang/src/tl_templates/ascend/common.h
```

## 历史结果

V55 官方阶段结果为 12/12 正确、57.67 分，相对固定原生官方基线最大减速 4.65%；
评测提交 `c1da598974278b8597890c5f79d9d8d4dd3fc424`，Run `91f0bb0e-ad86-4c38-ae7b-4ee9642453a9`。
这是已评测版本的记录，本次测试框架整合不构成一次新的官方评测。

`results/` 的已有本地文件保留，但不再由 Git 跟踪；构建与新报告写入 `build/`。
历史回执也可从提交 `4ca1c621bae064ef0603d1ef03370963aebcc49d` 的 `code_dsl/results/` 读取。
本地完整证据归档为 `build/pinv-v55-evidence-20261020.tar.gz`，SHA-256 为
`4e80f92f317d57d7f817c86057fb9cc7fd6edaa22517bf33e1c5bf7ee22b581e`。
本地项目清单保留导出时的源码身份和目录迁移记录，不把历史回执重写为新版本验证。

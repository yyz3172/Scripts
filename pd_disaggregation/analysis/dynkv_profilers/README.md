# DynamicKV Profilers

用于解析 vLLM-Ascend DynamicKV 相关日志的脚本集合。

## 脚本说明

| 脚本 | 功能 | 环境变量 |
|------|------|----------|
| **`run_dynkv_profilers.py`** | **一次运行下面四个解析脚本** | （无，仅聚合调用） |
| `parse_profile_execute_duration.py` | 解析 `ProfileExecuteDuration` 各阶段耗时；**prepare input 为 CPU wall**（与 `VLLM_DYNKV_PROFILE_PREPARE` 无关） | `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1` |
| `parse_dynkv_prepare_profile.py` | 解析 `_prepare_inputs` 阶段耗时（ON=DynamicKV 全字段；OFF=标准路径，DynKV 字段为 0） | `VLLM_DYNKV_PROFILE_PREPARE=1` |
| `parse_forward_profile.py` | 解析 `forward` 阶段的耗时分解 | `VLLM_DYNKV_PROFILE_FORWARD=1`（可选 `FIA` / `PA`） |
| `parse_offload_profile.py` | 解析 prefill 尾 **offload rewrite**：`rewrite_ms` / `finished_reqs` | `VLLM_DYNKV_PROFILE_FORWARD=1`（看 **prefill.log**） |
| `parse_model_acl_profile.py` | 解析 `model_acl`（`_update_attn_pa_params`）细分 | `VLLM_DYNKV_PROFILE_MODEL_ACL=1` |

## 使用方法

### 1. 启用日志

在 Decode worker 启动时设置对应的环境变量（``1P2_1D2_dynamickv_pa/run_decode.sh`` 已预置推荐组合）：

```bash
# 启用 ProfileExecuteDuration 基础计时
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1

# 启用 DynamicKV 性能日志（按需开启）
export VLLM_DYNKV_PROFILE_PREPARE=1
export VLLM_DYNKV_PROFILE_FORWARD=1
# PA decode：graph replay NPU sync（graph_npu_ms / fwd_block_npu_ms）+ 可选 eager 逐层 PA
export VLLM_DYNKV_PROFILE_PA=0   # PA graph 路径可选 1；与 FORWARD 联用
# model_acl 细分：ctx_lens / block_table / graph_update / event_record（B6 分析必开）
export VLLM_DYNKV_PROFILE_MODEL_ACL=1
# 可选：FIA eager 算子计时（32 sync/步，仅 PIA 目录 run_decode.sh）
# export VLLM_DYNKV_PROFILE_FIA=1
```

**additional_config** 需与代码一致（algo-2 Phase A/B）：

```json
"dynamic_kv": {
  "enabled": true,
  "uniform_kv_budget": "fixed_base"
}
```

启动后首步 decode 会打**一行**配置摘要（仅一次）：

```text
[DynamicKV][profile_status] dynkv_enabled=True impl=offload uniform_kv_budget=fixed_base ...
  PROFILE: PREPARE=1 FORWARD=1 MODEL_ACL=1 ...
```

可用 ``grep profile_status decode.log`` 确认 DynamicKV 与 profile 开关是否生效。

### 2. 运行推理并收集日志

```bash
# 将日志重定向到文件
python -m vllm.entrypoints.openai.api_server ... 2>&1 | tee decode.log
```

### 3. 解析日志

```bash
cd /path/to/Scripts/pd_disaggregation/analysis/dynkv_profilers

# 推荐：一次跑齐四个解析脚本（单份日志）
python run_dynkv_profilers.py /path/to/decode.log
python run_dynkv_profilers.py /path/to/decode.log --by-worker

# 单独解析（与上面等价的分项）
# 解析 ProfileExecuteDuration 各阶段（prepare input, forward, Sample, post process）
python parse_profile_execute_duration.py /path/to/decode.log

# 解析 prepare_inputs 阶段
python parse_dynkv_prepare_profile.py /path/to/decode.log

# 解析 forward 阶段（树形缩进：model_attn_op 在 model_attn 下）
python parse_forward_profile.py /path/to/decode.log

# 解析 prefill offload rewrite（高并发 TTFT 诊断）
python parse_offload_profile.py /path/to/prefill.log
python parse_offload_profile.py /path/to/prefill.log --by-worker

# 解析 model_acl（PA graph_task_update 细分）
python parse_model_acl_profile.py /path/to/decode.log
python parse_model_acl_profile.py decode_off.log decode_on.log

# 对比 OFF / ON 两份日志
python parse_forward_profile.py decode_off.log decode_on.log

# 按 Ray worker 分桶
python parse_dynkv_prepare_profile.py /path/to/decode.log --by-worker
```

## 日志格式示例

### `Profile execute duration`
```
Profile execute duration: tag=prepare input, duration=5.20ms
Profile execute duration: tag=forward, duration=48.50ms
Profile execute duration: tag=Sample, duration=0.70ms
Profile execute duration: tag=post process, duration=0.55ms
```

### `[DynamicKV][profile_status]`（一次性）

```
[DynamicKV][profile_status] dynkv_enabled=True impl=offload uniform_kv_budget=fixed_base ... PROFILE: PREPARE=1 FORWARD=1 MODEL_ACL=0 ...
```

### `[DynamicKV][prepare_profile]`

自 v0.13 DynamicKV 优化起，行首带 ``dynkv=0|1``（解析脚本会统计为字段 ``dynkv``，可忽略或用于过滤）：

```
[DynamicKV][prepare_profile] dynkv=1 layers=32 stack_init=0.02ms kv_list_build=0.01ms build_helper=0.25ms broadcast=0.37ms stacked_tensor=0.04ms layer_ctx_fill_batch=0.50ms layer_copy_meta=0.31ms layer_slot_remap=1.85ms layer_meta_assign=0.59ms total_loop=2.75ms
```

### `[DynamicKV][offload_profile]`（prefill，rewrite 分解）

prefill 侧 ``VLLM_DYNKV_PROFILE_FORWARD=1``，仅在某个 engine step 有请求完成 prefill 时打印：

```text
[DynamicKV][offload_profile] rewrite_ms=4521.33 finished_reqs=3
```

| 字段 | 说明 |
|------|------|
| `rewrite_ms` | 本 step 调用 ``run_offload_rewrite_and_build_updates`` 墙钟（ms） |
| `finished_reqs` | 本 step 完成 prefill、进入 rewrite 的请求数 |

```bash
python parse_offload_profile.py prefill.log
```

输出含 ``rewrite_ms`` / ``rewrite_ms_per_req`` 分位数、``finished_reqs`` 分布、``pile-up``（``finished_reqs>=2`` 占比）。

### `[DynamicKV][forward_profile]`

PA graph 为 ``[forward_profile][pa]``，同样带 ``dynkv=1``：

```
[DynamicKV][forward_profile][pa] dynkv=1 ctx_setup=0.12ms kv_setup=0.05ms dynkv_pre=0.03ms model_cpu=19.29ms model_acl=2.50ms graph_replay_wall=16.00ms graph_npu_ms=15.80ms ... profile_cpu_total=19.50ms
```

## 输出说明

### prepare_profile 字段

| 字段 | 说明 |
|------|------|
| `stack_init` | `_dynkv_stack` 缓冲区初始化 |
| `kv_list_build` | 构建 `rid_list_dyn` 和 `kv_list_dyn` |
| `build_helper` | 构建 `all_tmp_lens` 和 `slot_jobs_all` |
| `broadcast` | TP 广播 `built_dyn` |
| `stacked_tensor` | 构建 `stacked_dyn_lens_t` tensor |
| `layer_copy_meta` | 每层 `copy(attn_metadata)` |
| `layer_slot_remap` | 批量 slot remap（per-job ``[L,n_masked]`` scatter；P0b 全矩阵 gather 在 Ascend 上更慢已回退） |
| `layer_ctx_fill_batch` | ON：循环外批量写 graph ``context_lens_buf``（P1：ctx workspace 一次 ``[L,n]`` 写；需重 capture） |
| `layer_slot_assign` | stack remap 后 ``copy_`` 到 graph slot（统一 workspace 命中时为 0） |
| `layer_meta_assign` | ON：每层仅 ``setattr`` 指向已填 buffer（原 ``layer_other`` 主体） |
| `layer_other` | 与 ``layer_meta_assign`` 相同（兼容旧日志） |
| `total_loop` | 每层循环相关耗时合计（= 下述四项之和） |
| `  layer_slot_remap` | ↳ 批量 slot remap（循环前） |
| `  layer_slot_assign` | ↳ 批量写入 graph slot buffer（循环前） |
| `  layer_copy_meta` | ↳ 每层 copy metadata + slot 指针（循环内累加） |
| `  layer_meta_assign` | ↳ 每层 setattr dyn kv 元数据（循环内累加） |

解析脚本 ``parse_dynkv_prepare_profile.py`` 打印时，上述四项缩进显示在 ``total_loop`` 行之后。

### prepare_profile_ext 字段（P3）

| 字段 | 说明 |
|------|------|
| `update_states` | ``execute_model`` 内 ``_update_states`` |
| `prepare_core` | ``_prepare_inputs`` 中 attn 构建前（block_table/positions 等） |
| `prepare_attn_build` | ``builder.build`` 构建 ``attn_metadata_i`` |
| `prepare_kv_setup` | KV 组循环内、``builder.build`` 前（common metadata / block_table / padding） |
| `prepare_cos_sin` | ``update_cos_sin(positions)`` |
| `prepare_tail` | ``lmhead_tp`` pad 等收尾 |

### prepare_profile_reconcile（对账 Profile ``prepare input``）

每步一行，在 ``_prepare_inputs`` 结束打出：

```text
profile_prepare_est ≈ update_states + prepare_inputs_wall  ≈ Profile prepare input
inner_sum = prepare_core + prepare_kv_setup + prepare_attn_build
          + loop_sum + prepare_cos_sin + prepare_tail
prepare_gap = prepare_inputs_wall - inner_sum   # _prepare_inputs 内未细分部分
loop_sum    = 各 [prepare_profile] 行字段累加（stack/broadcast/ctx/total_loop 等）
```

| 字段 | 说明 |
|------|------|
| `prepare_inputs_wall` | 整段 ``_prepare_inputs`` 墙钟 |
| `loop_sum` | 所有 ``[prepare_profile]`` loop 字段之和（每 attn_group 累加） |
| `inner_sum` | 上述可加项合计 |
| `prepare_gap` | wall − inner_sum（计时孔洞 / Python 开销） |
| `profile_prepare_est` | update_states + wall，应对齐 ``parse_profile_execute_duration`` |

**统一 slot workspace（消除 ``layer_slot_assign``）**：FULL PA graph capture 时预分配
``[L, n_sm]`` 父 tensor，每层 ``slot_mapping`` 为 row view；runtime remap 写同一块内存。
**修改后须重启/重 capture graph**；否则 ``data_ptr`` 校验失败会回退 stack+assign。

### forward_profile 字段与层级（`parse_forward_profile.py`）

解析脚本按 **PA / PIA** 自动分表。字段后缀：`*` = ortho（勿与同级相加）、`#` = 非时间指标。

**PA graph（`[forward_profile][pa]`）— CPU 可相加分解：**

```text
profile_cpu_total ≈ ctx_setup + kv_setup + dynkv_pre + model_cpu + dynkv_post
model_cpu ≈ model_acl + model()墙钟
  model_acl
  model_npu_ms ⊇ graph_npu_ms
    graph_npu_ms*     # 同一次 replay 的 NPU
    graph_replay_wall*  # 同一次 replay 的 CPU，与 graph_npu_ms 勿相加
fwd_block_npu_ms†    # 独立 NPU 轴，≈ execute forward
pa_kv_tokens_avg#
```

**PIA eager + FIA（`[forward_profile]` + `fia_ms_total`）— CPU hook 可相加：**

```text
profile_cpu_total ≈ ctx_setup + kv_setup + dynkv_pre + model + dynkv_post
model ≈ model_core + model_sp_pcp
  model_core ≈ model_embed + model_norm + model_attn + model_mlp + model_layer_rms
    model_attn
      model_attn_op
        fia_ms_total*   # NPU FIA，与 model_attn_op 勿相加
        fia_kv_tokens_avg#
```

| 字段 | 说明 |
|------|------|
| `ctx_setup` | `set_ascend_forward_context` 设置 |
| `kv_setup` | `maybe_setup_kv_connector` |
| `dynkv_pre` | DynamicKV 预处理（Decode 端应接近 0）|
| `model` / `model_cpu` | PIA：`model` 墙钟；PA：`model_cpu` 同义 |
| `model_acl` | `_update_aclgraph_attn_params`（replay **之前**；PIA eager 常为 0） |
| `model_core` | `self.model(...)` 墙钟（PIA hook 父节点） |
| `graph_replay_wall` / `model_graph_replay` | `replay()` CPU（PA）；与 `graph_npu_ms` 交叉，**勿相加** |
| `graph_npu_ms` | `PROFILE_PA=1`：`replay()` NPU sync 整图时间 |
| `model_npu_ms` | 整段 `self.model()` NPU；**⊇** `graph_npu_ms`，勿与 `model_acl` 相加 |
| `fwd_block_npu_ms` | 整段 forward context NPU，≈ **execute forward** |
| `model_embed` / `model_norm` / `model_attn` / `model_mlp` | PIA：32 层 hook 累计（CPU） |
| `model_attn_op` | PIA：Attention 算子墙钟 |
| `fia_ms_total` | PIA：`PROFILE_FIA=1`，32 层 FIA NPU 合计 |
| `fia_kv_tokens_avg` / `pa_kv_tokens_avg` | metadata 统计（非时间） |
| `model_layer_rms` | `model_core - attn - mlp` 余量 |
| `model_sp_pcp` | SP all-gather / PCP restore |
| `dynkv_post` | DynamicKV 后处理 |
| `profile_cpu_total` | forward 块 CPU 总墙钟 |

### model_acl_profile 字段

| 字段 | 说明 |
|------|------|
| `layers` | 本步 PA graph 更新的层数（通常 32） |
| `ctx_lens` | `pa_dynamic_kv_context_lens_for_graph_update`（含 `copy_`） |
| `block_table` | 从 metadata 绑定 `block_tables` |
| `block_table_swap` | 本步中 `meta.block_tables` 与 capture 时 `attn_params` 指针不一致的层数 |
| `graph_update` | `gu_begin` + `gu_pa` + `gu_end` 合计 |
| `gu_begin` / `gu_pa` / `gu_end` | `graph_task_update_begin`、PA 重绑调用、`graph_task_update_end` |
| `event_record` | 每层 `ExternalEvent.record` |
| `loop_other` | `total` 减去上述分项（zip/解包等余量） |
| `total` | 整段 `_update_attn_pa_params` 墙钟 |
| `per_layer_ctx` / `per_layer_graph_update` | 每层均值（ms） |

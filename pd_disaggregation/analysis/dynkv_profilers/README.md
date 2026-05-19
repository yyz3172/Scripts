# DynamicKV Profilers

用于解析 vLLM-Ascend DynamicKV 相关日志的脚本集合。

## 脚本说明

| 脚本 | 功能 | 环境变量 |
|------|------|----------|
| `parse_profile_execute_duration.py` | 解析 `ProfileExecuteDuration` 各阶段耗时 | `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1` |
| `parse_dynkv_prepare_profile.py` | 解析 `_prepare_inputs` 阶段的 DynamicKV 耗时分解 | `VLLM_DYNKV_PROFILE_PREPARE=1` |
| `parse_forward_profile.py` | 解析 `forward` 阶段的耗时分解 | `VLLM_DYNKV_PROFILE_FORWARD=1` |

## 使用方法

### 1. 启用日志

在 Decode worker 启动时设置对应的环境变量：

```bash
# 启用 ProfileExecuteDuration 基础计时
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1

# 启用 DynamicKV 性能日志（按需开启）
export VLLM_DYNKV_PROFILE_PREPARE=1
export VLLM_DYNKV_PROFILE_FORWARD=1
```

### 2. 运行推理并收集日志

```bash
# 将日志重定向到文件
python -m vllm.entrypoints.openai.api_server ... 2>&1 | tee decode.log
```

### 3. 解析日志

```bash
cd /path/to/Scripts/pd_disaggregation/analysis/dynkv_profilers

# 解析 ProfileExecuteDuration 各阶段（prepare input, forward, Sample, post process）
python parse_profile_execute_duration.py /path/to/decode.log

# 解析 prepare_inputs 阶段
python parse_dynkv_prepare_profile.py /path/to/decode.log

# 解析 forward 阶段
python parse_forward_profile.py /path/to/decode.log

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

### `[DynamicKV][prepare_profile]`
```
[DynamicKV][prepare_profile] layers=32 stack_init=0.02ms kv_list_build=0.01ms build_helper=0.25ms broadcast=0.37ms stacked_tensor=0.04ms layer_copy_meta=0.31ms layer_slot_remap=1.85ms layer_other=0.59ms total_loop=2.75ms
```

### `[DynamicKV][forward_profile]`
```
[DynamicKV][forward_profile] ctx_setup=0.12ms kv_setup=0.05ms dynkv_pre=0.03ms model=48.50ms dynkv_post=0.02ms total=48.72ms
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
| `layer_slot_remap` | 每层 slot_mapping 重映射 |
| `layer_other` | 每层其他操作 |
| `total_loop` | 每层循环总耗时 |

### forward_profile 字段

| 字段 | 说明 |
|------|------|
| `ctx_setup` | `set_ascend_forward_context` 设置 |
| `kv_setup` | `maybe_setup_kv_connector` |
| `dynkv_pre` | DynamicKV 预处理（Decode 端应接近 0）|
| `model` | 实际模型 forward（所有层）|
| `dynkv_post` | DynamicKV 后处理（Decode 端应接近 0）|
| `total` | forward 块总耗时 |

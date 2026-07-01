import argparse
import os
import json
import time
import requests
import threading
from pathlib import Path
from collections import defaultdict, OrderedDict
from datetime import datetime
import statistics
from queue import Queue
import sys


CHAT_TIMEOUT = (1000, 2000)  # (连接超时, 读取超时)
RELEASE_TIMEOUT = 1000


_LOG_CTX = threading.local()


class Logger:
    """简单的日志工具，同时输出到终端和文件"""

    def __init__(self, log_file=None, enable_file_log=True):
        self.log_file = log_file
        self.enable_file_log = enable_file_log
        self.file_handle = None
        self._partial_line = ""

        if self.enable_file_log and self.log_file:
            self.file_handle = open(self.log_file, 'w', encoding='utf-8')

    def _prefix(self) -> str:
        tid = threading.get_ident()
        sid = getattr(_LOG_CTX, "session_id", None) or "NO_SESSION"
        return f"[T{tid}][{sid}] "

    def write(self, text):
        if not text:
            return

        data = self._partial_line + text
        self._partial_line = ""

        parts = data.splitlines(keepends=True)
        out_chunks: list[str] = []
        for part in parts:
            if part.endswith("\n"):
                out_chunks.append(self._prefix() + part)
            else:
                self._partial_line = part

        if out_chunks:
            out = "".join(out_chunks)
            sys.__stdout__.write(out)
            if self.enable_file_log and self.file_handle:
                self.file_handle.write(out)
                self.file_handle.flush()

    def flush(self):
        sys.__stdout__.flush()
        if self.file_handle:
            self.file_handle.flush()

    def close(self):
        if self.file_handle:
            self.file_handle.close()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = sys.__stdout__
        self.close()

class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.lock = threading.Lock()
        self.ttft_list = []  # Time To First Token
        self.total_time_list = []  # 总请求时间
        self.input_tokens_list = []  # 输入token数
        self.output_tokens_list = []  # 输出token数
        self.error_count = 0
        self.success_count = 0
        self.session_metrics = defaultdict(list)  # 按session统计
        self.request_start_times = []  # 记录每个请求的开始时间

    def add_metrics(self, session_id, turn_num, ttft, total_time, input_tokens=None, success=True, start_time=None, output_tokens=None):
        with self.lock:
            if success:
                self.ttft_list.append(ttft)
                self.total_time_list.append(total_time)
                if input_tokens is not None:
                    self.input_tokens_list.append(input_tokens)
                if output_tokens is not None:
                    self.output_tokens_list.append(output_tokens)
                self.success_count += 1
                self.session_metrics[session_id].append({
                    'turn_num': turn_num,
                    'session_id': session_id,
                    'ttft': ttft,
                    'total_time': total_time,
                    'input_tokens': input_tokens
                })
                if start_time is not None:
                    self.request_start_times.append(start_time)
            else:
                self.error_count += 1

    def get_summary(self):
        """获取统计摘要"""
        with self.lock:
            if not self.ttft_list:
                return {
                    'ttft_avg': 0,
                    'ttft_p50': 0,
                    'ttft_p90': 0,
                    'ttft_p95': 0,
                    'ttft_p99': 0,
                    'total_time_avg': 0,
                    'input_tokens_avg': 0,
                    'output_tokens_avg': 0,
                    'success_count': self.success_count,
                    'error_count': self.error_count
                }

            sorted_ttft = sorted(self.ttft_list)
            sorted_total = sorted(self.total_time_list)

            summary = {
                'ttft_avg': statistics.mean(self.ttft_list),
                'ttft_min': min(self.ttft_list),
                'ttft_max': max(self.ttft_list),
                'ttft_p50': sorted_ttft[len(sorted_ttft) // 2],
                'ttft_p90': sorted_ttft[int(len(sorted_ttft) * 0.9)],
                'ttft_p95': sorted_ttft[int(len(sorted_ttft) * 0.95)],
                'ttft_p99': sorted_ttft[int(len(sorted_ttft) * 0.99)],
                'total_time_avg': statistics.mean(self.total_time_list),
                'total_time_min': min(self.total_time_list),
                'total_time_max': max(self.total_time_list),
                'success_count': self.success_count,
                'error_count': self.error_count,
                'total_requests': self.success_count + self.error_count
            }

            if self.input_tokens_list:
                summary['input_tokens_avg'] = statistics.mean(self.input_tokens_list)
                summary['input_tokens_min'] = min(self.input_tokens_list)
                summary['input_tokens_max'] = max(self.input_tokens_list)
            else:
                summary['input_tokens_avg'] = None
                summary['input_tokens_min'] = None
                summary['input_tokens_max'] = None

            if self.output_tokens_list:
                summary['output_tokens_avg'] = statistics.mean(self.output_tokens_list)
                summary['output_tokens_min'] = min(self.output_tokens_list)
                summary['output_tokens_max'] = max(self.output_tokens_list)
            else:
                summary['output_tokens_avg'] = None
                summary['output_tokens_min'] = None
                summary['output_tokens_max'] = None

            # 计算实际QPS
            if self.request_start_times:
                sorted_times = sorted(self.request_start_times)
                if len(sorted_times) > 1:
                    time_range = sorted_times[-1] - sorted_times[0]
                    actual_qps = len(sorted_times) / time_range if time_range > 0 else 0
                    summary['actual_qps'] = actual_qps
                else:
                    summary['actual_qps'] = 0
            else:
                summary['actual_qps'] = 0

            return summary
    def to_dict(self):
        """转换为可序列化的字典"""
        with self.lock:
            return {
                'ttft_list': self.ttft_list.copy(),
                'total_time_list': self.total_time_list.copy(),
                'input_tokens_list': self.input_tokens_list.copy(),
                'output_tokens_list': self.output_tokens_list.copy(),
                'error_count': self.error_count,
                'success_count': self.success_count,
                'session_metrics': dict(self.session_metrics),
                'request_start_times': self.request_start_times.copy()
            }

    @classmethod
    def from_dict(cls, data):
        """从字典恢复MetricsCollector对象"""
        collector = cls()
        collector.ttft_list = data['ttft_list']
        collector.total_time_list = data['total_time_list']
        collector.input_tokens_list = data['input_tokens_list']
        collector.error_count = data['error_count']
        collector.success_count = data['success_count']
        collector.session_metrics = defaultdict(list, data['session_metrics'])
        collector.request_start_times = data.get('request_start_times', [])
        return collector


def save_metrics(summary, metrics_collector, output_file):
    """保存metrics到文件"""
    metrics_dict = metrics_collector.to_dict()

    data = {
        'summary': summary,
        'metrics_dict': metrics_dict
    }

    json_file = output_file.replace('.pkl', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Metrics已保存到: {json_file}")


def load_metrics(with_release_file, without_release_file):
    """加载保存的metrics"""
    print(f"正在加载metrics...")

    json_file_with = with_release_file.replace('.pkl', '.json')
    with open(json_file_with, 'r', encoding='utf-8') as f:
        with_data = json.load(f)

    summary_with = with_data['summary']
    metrics_with = MetricsCollector.from_dict(with_data['metrics_dict'])

    json_file_without = without_release_file.replace('.pkl', '.json')
    with open(json_file_without, 'r', encoding='utf-8') as f:
        without_data = json.load(f)

    summary_without = without_data['summary']
    metrics_without = MetricsCollector.from_dict(without_data['metrics_dict'])

    print(f"Metrics加载完成！")
    return summary_with, metrics_with, summary_without, metrics_without


def load_session_data(data_folder):
    """
    加载所有session数据并按session_id和turn_num组织
    返回: {session_id: [(turn_num, file_path, data), ...]}
    """
    session_data = defaultdict(list)
    json_files = []
    for item in os.listdir(Path(data_folder)):
        #if item.endswith('new'):
        new_data_folder = os.path.join(data_folder, item)
        json_files.extend(list(Path(new_data_folder).glob('messages*.json')))
    print(f"json_files:{json_files}")
    print(f"找到 {len(json_files)} 个JSON文件")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            session_id = data.get('session_id', '')
            turn_num = data.get('turn_num', 0)
            session_data[session_id].append((turn_num, json_file, data))
        except Exception as e:
            print(f"读取文件 {json_file} 失败: {e}")

    for session_id in session_data:
        session_data[session_id].sort(key=lambda x: x[0])

    sorted_sessions = sorted(session_data.items(), key=lambda x: x[0])
    ordered_session_data = OrderedDict(sorted_sessions)
    print(f"加载了 {len(ordered_session_data)} 个不同的session")

    # 打印前3个session的信息
    print("\n前3个session信息:")
    for i, (session_id, turns) in enumerate(list(ordered_session_data.items())[:3]):
        turn_nums = [t[0] for t in turns]
        print(f"  {i + 1}. Session '{session_id}': {len(turns)} turns, turn_nums={turn_nums}")

    return ordered_session_data


def send_chat_request(
    messages,
    model,
    chat_url,
    cache_salt=None,
    stream=True,
    max_tokens=5,
    cache_sharing=True,
):
    """
    发送chat请求并返回TTFT、总时间和input_tokens

    Args:
        messages: 消息列表
        cache_salt: session id（新接口使用cache_salt替代session_id）
        stream: 是否流式响应
        max_tokens: 收集的最大token数
        cache_sharing: 是否启用跨 session prefix cache 共享
    """
    req_data = {
        "model": model,
        "messages": messages,
        "stream": stream
    }

    if cache_salt is not None:
        req_data["cache_salt"] = cache_salt
        req_data["cache_sharing"] = cache_sharing

    headers = {
        "Content-Type": "application/json",
    }

    start_time = time.time()
    ttft = None
    input_tokens = None
    first_token_time = None
    token_count = 0

    session_log = cache_salt if cache_salt else "NO_SESSION"

    try:
        response = requests.post(
            chat_url,
            headers=headers,
            json=req_data,
            stream=stream,
            timeout=CHAT_TIMEOUT
        )
        response.raise_for_status()

        if stream:
            buffer = ""

            # 使用chunk_size=1逐字节读取
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if not chunk:
                    continue

                buffer += chunk

                # 按\n分割
                while '\n' in buffer:
                    line_end = buffer.find('\n')
                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1:]

                    # 跳过空行
                    if not line:
                        continue

                    # 处理SSE格式的数据
                    if line.startswith('data: '):
                        data_str = line[6:].strip()

                        # 跳过 [DONE]
                        if data_str == '[DONE]':
                            continue

                        try:
                            chunk_data = json.loads(data_str)

                            # 提取usage信息(仅首次)
                            if input_tokens is None:
                                usage = chunk_data.get('usage', {})
                                if usage:
                                    input_tokens = usage.get('prompt_tokens')

                            # 提取content
                            choices = chunk_data.get('choices', [])
                            if not choices:
                                continue

                            delta = choices[0].get('delta', {})
                            content = delta.get('content', '')
                            tool_calls = delta.get('tool_calls')

                            # 处理有效内容
                            if content or tool_calls:
                                # 记录首个token时间
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    ttft = first_token_time - start_time

                                token_count += 1

                                # 收集够指定数量token后停止
                                if token_count >= max_tokens:
                                    response.close()
                                    break

                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"[WARN] 处理数据块失败 ({session_log}): {e}")
                            continue

                # 如果已收集够token,跳出循环
                if token_count >= max_tokens:
                    break

            # 如果流式响应完成但没有记录到TTFT,使用总时间
            if ttft is None and token_count > 0:
                ttft = time.time() - start_time
                print("TTFT use total time")

        else:
            # 非流式响应
            resp_data = response.json()
            usage = resp_data.get('usage', {})
            if usage:
                input_tokens = usage.get('prompt_tokens')
            ttft = time.time() - start_time

        total_time = time.time() - start_time

        # 确保ttft有值
        if stream and ttft is None:
            ttft = total_time

        success = token_count > 0 if stream else True

        if not success:
            print(f"请求模型消息-----------------------")
            print(f"[CHAT WARN] 请求未收到有效内容 (cache_salt={cache_salt})")

        return ttft, total_time, input_tokens, success, start_time, token_count

    except requests.exceptions.Timeout:
        total_time = time.time() - start_time
        print(f"[CHAT ERROR] 请求超时 (cache_salt={cache_salt})")
        return total_time, total_time, None, False, start_time
    except Exception as e:
        total_time = time.time() - start_time
        print(f"[CHAT ERROR] 请求失败 (cache_salt={cache_salt}): {e}")
        return total_time, total_time, None, False, start_time


def send_release_request(
    messages,
    message_index_begin,
    cache_salt,
    model,
    release_url,
    cache_sharing=True,
):
    """
    发送release请求

    Args:
        messages: 消息列表
        message_index_begin: 开始释放的消息索引（0 表示从首条消息起释放）
        cache_salt: session id
        cache_sharing: 是否启用跨 session prefix cache 共享

    Returns:
        (success, block_released): HTTP 是否成功，以及服务端返回的 block_released
    """
    release_data = {
        "model": model,
        "messages": messages,
        "messages_released_index": message_index_begin,
        "cache_salt": cache_salt,
        "cache_sharing": cache_sharing,
    }
    headers = {
        "Content-Type": "application/json",
    }

    try:
        start = time.time()
        response = requests.post(
            release_url,
            headers=headers,
            json=release_data,
            timeout=RELEASE_TIMEOUT
        )
        release_time = time.time() - start
        block_released = None
        if response.status_code == 200:
            try:
                block_released = response.json().get("block_released")
            except (json.JSONDecodeError, ValueError):
                pass
        release_log = (
            f"[RELEASE] {release_time:.3f}s, "
            f"cache_salt={cache_salt}, "
            f"index={message_index_begin}, "
            f"status={response.status_code}, "
            f"block_released={block_released}"
        )
        if response.status_code != 200 or block_released is None:
            release_log += f", response={response.content!r}"
        print(release_log)
        return response.status_code == 200, block_released
    except Exception as e:
        print(f"[RELEASE ERROR] 请求失败 (cache_salt={cache_salt}): {e}")
        return False, None


def compute_release_plan(prev_msgs, curr_msgs):
    """
    根据前后两轮 messages 判断是否需要 release，以及 messages_released_index。

    Returns:
        (should_release, message_index_begin)
    """
    if not prev_msgs:
        return False, None

    prev_len = len(prev_msgs)
    curr_len = len(curr_msgs)

    # - curr_msgs 为空：不触发中间 release（旧逻辑 should_release 保持 False）
    # - 首条 message 发生变化（idx=0）：也不触发中间 release（旧逻辑 if msg_idx 跳过）
    if curr_len == 0:
        return False, None

    for idx in range(min(prev_len, curr_len)):
        if prev_msgs[idx] != curr_msgs[idx]:
            print(f"  [RELEASE REASON] Message modified at index {idx}")
            if idx == 0:
                return False, None
            return True, idx

    return False, None



def process_session(
    session_id,
    turns_data,
    metrics_collector,
    model,
    chat_url,
    release_url,
    enable_release=True,
    cache_sharing=True,
):
    """处理单个session的所有turns（串行模式）"""
    print(
        f"开始处理，共 {len(turns_data)} 个对话轮次, "
        f"enable_release={enable_release}, cache_sharing={cache_sharing}"
    )
    prev_msgs = []
    for num, file_path, data in turns_data:
        curr_msgs = data.get('message', [])

        # 在发送当前请求前，先按上一轮 messages 释放 KV
        if enable_release and prev_msgs:
            should_release, message_index_begin = compute_release_plan(
                prev_msgs, curr_msgs
            )
            if should_release:
                if message_index_begin >= len(prev_msgs):
                    print(
                        f"[WARN] {session_id} turn_{num}: "
                        f"message_index_begin={message_index_begin} 超出范围 "
                        f"(prev_messages长度={len(prev_msgs)})"
                    )
                else:
                    success, block_released = send_release_request(
                        prev_msgs,
                        message_index_begin,
                        session_id,
                        model,
                        release_url,
                        cache_sharing=cache_sharing,
                    )
                    if success:
                        pass
                    else:
                        print("release失败")

        # 发送chat请求
        cache_salt = session_id if enable_release else None
        print(f"发送第 {num} 个对话轮次 chat 请求")
        ttft, total_time, input_tokens, success, start_time, output_tokens = send_chat_request(
            curr_msgs,
            model,
            chat_url,
            cache_salt=cache_salt,
            cache_sharing=cache_sharing,
        )

        metrics_collector.add_metrics(
            session_id, num, ttft, total_time, input_tokens, success, start_time, output_tokens
        )

        prev_msgs = curr_msgs

    # Session 所有对话轮次完成后，释放该 session 剩余 KV cache
    if enable_release and prev_msgs:
        success, block_released = send_release_request(
            prev_msgs,
            0,
            session_id,
            model,
            release_url,
            cache_sharing=cache_sharing,
        )
        if success:
            print(
                "session结束 release成功, "
                f"index_begin:0, block_released:{block_released}"
            )
        else:
            print("session结束 release失败")


def worker_thread(
    task_queue,
    metrics_collector,
    model,
    chat_url,
    release_url,
    enable_release,
    cache_sharing=True,
):
    """工作线程 - 每个任务是一个完整的session"""
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        session_id, turns_data = task
        try:
            _LOG_CTX.session_id = session_id
            process_session(
                session_id,
                turns_data,
                metrics_collector,
                model,
                chat_url,
                release_url,
                enable_release,
                cache_sharing=cache_sharing,
            )
        except Exception as e:
            print(f"处理 session 时出错: {e}")
        finally:
            _LOG_CTX.session_id = None
            task_queue.task_done()


def apply_session_limits(session_data, max_sessions=None, max_turns=None):
    """按 session 数量与每 session 轮次上限裁剪数据。"""
    if max_sessions:
        session_ids = list(session_data.keys())[:max_sessions]
        session_data = {sid: session_data[sid] for sid in session_ids}

    if max_turns:
        session_data = {
            sid: turns[:max_turns]
            for sid, turns in session_data.items()
        }

    return session_data


def run_test(
    session_data,
    model,
    chat_url,
    release_url,
    num_threads=10,
    enable_release=True,
    cache_sharing=True,
):
    """
    运行测试（串行模式）

    Args:
        session_data: 已加载并裁剪后的 session 数据
        num_threads: 并发线程数
        enable_release: 是否启用cache释放
        cache_sharing: 是否启用跨 session prefix cache 共享
    """
    print(f"\n{'=' * 80}")
    print(f"测试配置:")
    print(f"  Session数: {len(session_data)}")
    print(f"  总对话轮次: {sum(len(turns) for turns in session_data.values())}")
    print(f"  并发线程数: {num_threads}")
    print(f"  测试模式: Session串行")
    print(f"  是否启用cache释放: {enable_release}")
    if enable_release:
        print(f"  cache_sharing: {cache_sharing}")
    if not enable_release:
        print(f"  注意: 不启用cache释放时，将不传入cache_salt")
    print(f"{'=' * 80}\n")

    metrics_collector = MetricsCollector()
    task_queue = Queue()

    # 按session放入队列
    for session_id, turns_data in session_data.items():
        task_queue.put((session_id, turns_data))

    print(f"串行模式: {len(session_data)} 个session按顺序执行\n")

    threads = []
    start_time = time.time()

    print(f"启动 {num_threads} 个工作线程...\n")
    for i in range(num_threads):
        t = threading.Thread(
            target=worker_thread,
            args=(
                task_queue,
                metrics_collector,
                model,
                chat_url,
                release_url,
                enable_release,
                cache_sharing,
            )
        )
        t.start()
        threads.append(t)

    # 等待所有任务完成
    task_queue.join()

    # 发送停止信号
    for _ in range(num_threads):
        task_queue.put(None)

    # 等待所有线程结束
    for t in threads:
        t.join()
    total_duration = time.time() - start_time

    summary = metrics_collector.get_summary()

    print(f"\n{'=' * 80}")
    print(f"测试完成！")
    print(f"{'=' * 80}")
    print(f"测试模式: Session串行")
    print(f"总耗时: {total_duration:.2f} 秒")
    print(f"总请求数: {summary['total_requests']}")
    print(f"成功请求: {summary['success_count']}")
    print(f"失败请求: {summary['error_count']}")
    print(f"平均QPS: {summary['total_requests'] / total_duration:.2f}")
    if 'actual_qps' in summary:
        print(f"实际并发QPS: {summary['actual_qps']:.2f}")
    print(f"\nTTFT (Time To First Token) 统计:")
    print(f"  平均值: {summary['ttft_avg']:.3f} 秒")
    print(f"  最小值: {summary['ttft_min']:.3f} 秒")
    print(f"  最大值: {summary['ttft_max']:.3f} 秒")
    print(f"  P50: {summary['ttft_p50']:.3f} 秒")
    print(f"  P90: {summary['ttft_p90']:.3f} 秒")
    print(f"  P95: {summary['ttft_p95']:.3f} 秒")
    print(f"  P99: {summary['ttft_p99']:.3f} 秒")
    print(f"\n总请求时间统计:")
    print(f"  平均值: {summary['total_time_avg']:.3f} 秒")
    print(f"  最小值: {summary['total_time_min']:.3f} 秒")
    print(f"  最大值: {summary['total_time_max']:.3f} 秒")

    if summary['input_tokens_avg'] is not None:
        print(f"\n输入Token统计:")
        print(f"  平均值: {summary['input_tokens_avg']:.0f}")
        print(f"  最小值: {summary['input_tokens_min']}")
        print(f"  最大值: {summary['input_tokens_max']}")

    print(f"{'=' * 80}\n")

    return summary, metrics_collector
def save_report(summary_with_release, summary_without_release,
                metrics_with_release, metrics_without_release,
                output_file="test_report.json"):
    """保存测试报告"""
    report = {
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "with_release": {
            "summary": summary_with_release,
            "session_count": len(metrics_with_release.session_metrics)
        },
        "without_release": {
            "summary": summary_without_release,
            "session_count": len(metrics_without_release.session_metrics)
        },
        "comparison": {
            "ttft_improvement_percentage": (
                    (summary_without_release['ttft_avg'] - summary_with_release['ttft_avg'])
                    / summary_without_release['ttft_avg'] * 100
            ) if summary_without_release['ttft_avg'] > 0 else 0,
            "ttft_p95_improvement_percentage": (
                    (summary_without_release['ttft_p95'] - summary_with_release['ttft_p95'])
                    / summary_without_release['ttft_p95'] * 100
            ) if summary_without_release['ttft_p95'] > 0 else 0,
            "ttft_p99_improvement_percentage": (
                    (summary_without_release['ttft_p99'] - summary_with_release['ttft_p99'])
                    / summary_without_release['ttft_p99'] * 100
            ) if summary_without_release['ttft_p99'] > 0 else 0,
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"测试报告已保存到: {output_file}")

    print(f"\n{'=' * 80}")
    print(f"对比结果:")
    print(f"{'=' * 80}")
    print(f"TTFT 平均值改善: {report['comparison']['ttft_improvement_percentage']:.2f}%")
    print(f"  无释放: {summary_without_release['ttft_avg']:.3f}s")
    print(f"  有释放: {summary_with_release['ttft_avg']:.3f}s")
    print(f"\nTTFT P95改善: {report['comparison']['ttft_p95_improvement_percentage']:.2f}%")
    print(f"  无释放: {summary_without_release['ttft_p95']:.3f}s")
    print(f"  有释放: {summary_with_release['ttft_p95']:.3f}s")
    print(f"\nTTFT P99改善: {report['comparison']['ttft_p99_improvement_percentage']:.2f}%")
    print(f"  无释放: {summary_without_release['ttft_p99']:.3f}s")
    print(f"  有释放: {summary_with_release['ttft_p99']:.3f}s")
    print(f"{'=' * 80}\n")

def positive_int(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} 不是正整数")
    return ivalue


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"{value!r} 不是合法布尔值，请使用 true/false"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="KV cache affinity 压测与对比工具")
    parser.add_argument("--url", default="10.41.38.238:8000", help="服务地址 host:port")
    parser.add_argument("--model", default="qwen3_32b", help="模型名称")
    parser.add_argument("--thread-num", type=int, default=1, help="并发线程数")
    parser.add_argument(
        "--repeat", type=positive_int, default=1,
        help="flag=1/2 时整套测试重复执行次数，默认 1",
    )
    parser.add_argument("--max-session", type=int, default=None, help="Session 数量上限，默认全部")
    parser.add_argument("--max-turns", type=int, default=None, help="每个 Session 的对话轮次上限，默认全部")
    parser.add_argument("--output-report-json", default="compare.json", help="对比报告输出路径 (flag=3)")
    parser.add_argument(
        "--flag", type=int, choices=[1, 2, 3], default=1,
        help="1=with_release, 2=without_release, 3=generate_report",
    )
    parser.add_argument(
        "--input-dataset",
        default="/root/xrx/l00444740_temp/msgs_50turn/",
        help="输入数据集目录",
    )
    parser.add_argument("--log-dir", default="local_run_analysis", help="日志与 metrics 输出目录")
    parser.add_argument("--metrics-with-release", help="with_release metrics 文件路径 (flag=3 必填)")
    parser.add_argument("--metrics-without-release", help="without_release metrics 文件路径 (flag=3 必填)")
    parser.add_argument(
        "--cache-sharing",
        type=str_to_bool,
        default=True,
        help="是否启用跨 session prefix cache 共享 (true/false)，默认 true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = args.model
    chat_url = f"http://{args.url}/v1/chat/completions"
    release_url = f"http://{args.url}/release_kv_cache"

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_file = args.input_dataset
    input_name = input_file.rstrip("/").split("/")[-1]

    flag = args.flag
    TN = args.thread_num
    MS = args.max_session
    MT = args.max_turns
    ms_tag = MS if MS is not None else "all"
    mt_tag = MT if MT is not None else "all"

    log_file = f"{input_name}_run_log_{timestamp}_TN{TN}_MS{ms_tag}_MT{mt_tag}_flag{flag}.txt"

    with Logger(log_file=os.path.join(log_dir, log_file), enable_file_log=True):
        print(f"{'=' * 80}")
        print(f"开始执行测试 - Flag={flag}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 80}\n")

        if flag == 1:
            print(f"执行阶段1: 启用cache释放测试 (重复执行 {args.repeat} 次)")
            print("=" * 80)

            session_data = load_session_data(input_file)
            total_turns_before = sum(len(turns) for turns in session_data.values())
            session_data = apply_session_limits(session_data, args.max_session, args.max_turns)
            total_turns_after = sum(len(turns) for turns in session_data.values())
            if args.max_session or args.max_turns:
                print(f"限制后: {len(session_data)} 个 session, 共 {total_turns_after} 个对话轮次 "
                      f"(原始 {total_turns_before} 个)\n")

            for run_idx in range(1, args.repeat + 1):
                if args.repeat > 1:
                    print(f"\n--- 第 {run_idx}/{args.repeat} 次重复 ---\n")
                summary_with, metrics_with = run_test(
                    session_data,
                    model,
                    chat_url,
                    release_url,
                    num_threads=args.thread_num,
                    enable_release=True,
                    cache_sharing=args.cache_sharing,
                )

                if args.repeat == 1:
                    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    metrics_file = os.path.join(
                        log_dir,
                        f"metrics_with_release_TN{TN}_MS{ms_tag}_MT{mt_tag}_{run_timestamp}.json",
                    )
                    save_metrics(summary_with, metrics_with, metrics_file)

            print("\n" + "=" * 80)
            print("阶段1完成！")
            print("\n" + "=" * 80)

        elif flag == 2:
            print(f"执行阶段2: 不启用cache释放测试 (重复执行 {args.repeat} 次)")
            print("=" * 80)
            print("请确认已重启vLLM服务清除cache！")
            print("本阶段将不传入cache_salt，避免session管理的影响")
            print("=" * 80 + "\n")

            session_data = load_session_data(input_file)
            total_turns_before = sum(len(turns) for turns in session_data.values())
            session_data = apply_session_limits(session_data, args.max_session, args.max_turns)
            total_turns_after = sum(len(turns) for turns in session_data.values())
            if args.max_session or args.max_turns:
                print(f"限制后: {len(session_data)} 个 session, 共 {total_turns_after} 个对话轮次 "
                      f"(原始 {total_turns_before} 个)\n")

            for run_idx in range(1, args.repeat + 1):
                if args.repeat > 1:
                    print(f"\n--- 第 {run_idx}/{args.repeat} 次重复 ---\n")
                summary_without, metrics_without = run_test(
                    session_data,
                    model,
                    chat_url,
                    release_url,
                    num_threads=args.thread_num,
                    enable_release=False,
                )

                if args.repeat == 1:
                    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    metrics_file = os.path.join(
                        log_dir,
                        f"metrics_without_release_TN{TN}_MS{ms_tag}_MT{mt_tag}_{run_timestamp}.json",
                    )
                    save_metrics(summary_without, metrics_without, metrics_file)

            print("\n" + "=" * 80)
            print("阶段2完成！")
            print("\n" + "=" * 80)

        elif flag == 3:
            if not args.metrics_with_release or not args.metrics_without_release:
                print("错误: flag=3 需要同时指定 --metrics-with-release 和 --metrics-without-release")
                sys.exit(1)

            print("执行阶段3: 生成对比报告")
            print("=" * 80)
            summary_with, metrics_with, summary_without, metrics_without = load_metrics(
                args.metrics_with_release,
                args.metrics_without_release,
            )
            save_report(
                summary_with, summary_without,
                metrics_with, metrics_without,
                output_file=args.output_report_json,
            )
            print("阶段3完成！")
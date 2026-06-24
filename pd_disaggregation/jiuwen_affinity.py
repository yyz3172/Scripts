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



class Logger:
    """简单的日志工具，同时输出到终端和文件"""

    def __init__(self, log_file=None, enable_file_log=True):
        self.log_file = log_file
        self.enable_file_log = enable_file_log
        self.file_handle = None

        if self.enable_file_log and self.log_file:
            self.file_handle = open(self.log_file, 'w', encoding='utf-8')

    def write(self, text):
        sys.__stdout__.write(text)
        if self.enable_file_log and self.file_handle:
            self.file_handle.write(text)
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


def send_chat_request(messages, cache_salt=None, stream=True, max_tokens=5):
    """
    发送chat请求并返回TTFT、总时间和input_tokens

    Args:
        messages: 消息列表
        cache_salt: session id（新接口使用cache_salt替代session_id）
        stream: 是否流式响应
        max_tokens: 收集的最大token数
    """
    req_data = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": stream
    }

    # 如果提供了cache_salt，则启用缓存共享
    if cache_salt is not None:
        req_data["cache_salt"] = cache_salt
        req_data["cache_sharing"] = True

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
            CONFIG["chat_url"],
            headers=headers,
            json=req_data,
            stream=stream,
            timeout=CONFIG['chat_timeout']
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


def send_release_request(messages, message_index_begin, cache_salt):
    """
    发送release请求

    Args:
        messages: 消息列表
        message_index_begin: 开始释放的消息索引
        message_index_end: 结束释放的消息索引
        cache_salt: session id
    """
    release_data = {
        "model": CONFIG["model"],
        "messages": messages,
        "messages_released_index": message_index_begin,
        # "tools_released_index": 0,
        # "tools": "",
        # "tools_released_index": "",
        # "message_index_end": message_index_end,
        "cache_salt": cache_salt,
        "cache_sharing": True,
    }
    headers = {
        "Content-Type": "application/json",
    }

    try:
        start = time.time()
        response = requests.post(
            CONFIG["release_url"],
            headers=headers,
            json=release_data,
            timeout=CONFIG['release_timeout']
        )
        release_time = time.time() - start
        print(f"[RELEASE] {cache_salt}: {release_time:.3f}s")
        print(f"[RELEASE] response: {response.content}")
        return response.status_code == 200
    except Exception as e:
        print(f"Release请求失败 (cache_salt={cache_salt}): {e}")
        return False



def process_session(session_id, turns_data, metrics_collector, enable_release=True):
    """处理单个session的所有turns（串行模式）"""
    print(f"[{session_id}] 开始处理，共 {len(turns_data)} 轮, enable_release={enable_release}")
    prev_msgs = []
    for num, file_path, data in turns_data:
        should_release = False
        curr_msgs = data.get('message', [])
        if enable_release and (prev_msgs or curr_msgs):
            prev_len = len(prev_msgs)
            curr_len = len(curr_msgs)
            if prev_len > 0:  # 只要前一轮有 messages，就需要设置 msg_idx
                # 默认前缀相同，不需要释放
                msg_idx = prev_len
                if curr_len > 0:  # 当前轮也有 messages，比较前缀
                    for idx in range(min(prev_len, curr_len)):
                        ''' 一般来说当前messages比前一轮长.
                        但是如果当前messages比前一轮短，也是可以从第一个不同考虑release。
                        当当前message比前一轮短时，如果当前轮与前一轮的前半部分一样就没必要release'''
                        if prev_msgs[idx] != curr_msgs[idx]:
                            should_release = True
                            msg_idx = idx  # 从第一个不同的消息开始释放
                            print(f"  [RELEASE REASON] Message modified at index {idx}")
                            break
                else:  # 当前轮没有 messages（从有变无）
                    # 前缀为空，算相同，不需要 release
                    # 应对bug情况
                    msg_idx = prev_len
                    # should_release 保持 False

        # 在发送当前请求前,先处理上一轮的release
        if enable_release and should_release and prev_msgs is not None:
            if msg_idx:
                # 使用新接口的索引方式：message_index_begin 和 message_index_end
                message_index_begin = msg_idx
                if message_index_begin >= len(prev_msgs):
                    print(f"[WARN] {session_id} turn_{num}: "
                          f"message_index_begin={message_index_begin} 超出范围 "
                          f"(prev_messages长度={len(prev_msgs)})")
                elif message_index_begin < len(prev_msgs):
                    success = send_release_request(
                        prev_msgs,
                        message_index_begin,
                        session_id
                    )
                    if success:
                        print(f"[{session_id}] release成功, turn:{num}, index_begin:{message_index_begin}")
                    else:
                        print(f"[{session_id}] release失败")

        # 发送chat请求
        cache_salt = session_id if enable_release else None
        print(f"发送第{num}轮chat msg")
        ttft, total_time, input_tokens, success, start_time, output_tokens = send_chat_request(
            curr_msgs,
            cache_salt=cache_salt
        )

        metrics_collector.add_metrics(
            session_id, num, ttft, total_time, input_tokens, success, start_time, output_tokens
        )

        prev_msgs = curr_msgs


def worker_thread(task_queue, metrics_collector, enable_release):
    """工作线程 - 每个任务是一个完整的session"""
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        session_id, turns_data = task
        try:
            process_session(session_id, turns_data, metrics_collector, enable_release)
        except Exception as e:
            print(f"处理session {session_id} 时出错: {e}")
        finally:
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


def run_test(data_folder, num_threads=10, enable_release=True, max_sessions=None, max_turns=None):
    """
    运行测试（串行模式）

    Args:
        data_folder: 数据文件夹
        num_threads: 并发线程数
        enable_release: 是否启用cache释放
        max_sessions: 最大 session 数，None 表示使用全部 session
        max_turns: 每个 session 最大对话轮次，None 表示使用全部轮次
    """
    print(f"\n{'=' * 80}")
    print(f"测试配置:")
    print(f"  数据文件夹: {data_folder}")
    print(f"  并发线程数: {num_threads}")
    print(f"  测试模式: Session串行")
    print(f"  是否启用cache释放: {enable_release}")
    if not enable_release:
        print(f"  注意: 不启用cache释放时，将不传入cache_salt")
    print(f"  Session数量: {max_sessions or '全部'}")
    print(f"  每Session轮次: {max_turns or '全部'}")
    print(f"{'=' * 80}\n")

    session_data = load_session_data(data_folder)
    total_turns_before = sum(len(turns) for turns in session_data.values())
    session_data = apply_session_limits(session_data, max_sessions, max_turns)
    total_turns_after = sum(len(turns) for turns in session_data.values())

    if max_sessions or max_turns:
        print(f"限制后: {len(session_data)} 个 session, 共 {total_turns_after} 轮请求 "
              f"(原始 {total_turns_before} 轮)\n")

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
            args=(task_queue, metrics_collector, enable_release)
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

if __name__ == "__main__":
    URL = '10.41.38.238:8000'

    # 配置
    CONFIG = {
        'chat_url': f'http://{URL}/v1/chat/completions',
        'release_url': f'http://{URL}/release_kv_cache',

        # 模型
        'model': 'qwen3_32b',

        "thread_num": 1,  # 并发线程数
        "max_session": 1,  # Session 数量上限，None 表示全部
        "max_turns": 5,  # 每个 Session 的对话轮次上限，None 表示全部
        "output_report_json": "compare.json",
        "flag": 1,  # 1=with_release, 2=without_release, 3=generate_report

        "input_dataset": "/root/xrx/l00444740_temp/msgs_50turn/",
        "log_dir": "local_run_analysis_1223_56session_8B",

        # 超时配置
        "chat_timeout": (1000, 2000),  # (连接超时, 读取超时)
        "release_timeout": 1000,
    }

    log_dir = CONFIG["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_file = CONFIG["input_dataset"]
    input_name = input_file.split("/")[-1]

    flag = CONFIG["flag"]
    TN = CONFIG["thread_num"]
    MS = CONFIG["max_session"]
    MT = CONFIG["max_turns"]
    ms_tag = MS if MS is not None else "all"
    mt_tag = MT if MT is not None else "all"

    log_file = f"{input_name}_run_log_{timestamp}_TN{TN}_MS{ms_tag}_MT{mt_tag}_flag{flag}.txt"

    with Logger(log_file=os.path.join(log_dir, log_file), enable_file_log=True):
        print(f"{'=' * 80}")
        print(f"开始执行测试 - Flag={flag}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 80}\n")

        if flag == 1:
            print("执行阶段1: 启用cache释放测试")
            print("=" * 80)
            summary_with, metrics_with = run_test(
                input_file,
                num_threads=CONFIG["thread_num"],
                enable_release=True,
                max_sessions=CONFIG["max_session"],
                max_turns=CONFIG["max_turns"],
            )

            metrics_file = os.path.join(log_dir,
                                        f"metrics_with_release_TN{TN}_MS{ms_tag}_MT{mt_tag}_{timestamp}.json")
            save_metrics(summary_with, metrics_with, metrics_file)

            print("\n" + "=" * 80)
            print("阶段1完成！")
            print("=" * 80)
            print("\n请按以下步骤继续:")
            print("1. 重启vLLM服务以清除cache")
            print("2. 修改代码中的 flag=2")
            print("3. 重新运行此脚本")
            print("=" * 80 + "\n")

        elif flag == 2:
            print("执行阶段2: 不启用cache释放测试")
            print("=" * 80)
            print("请确认已重启vLLM服务清除cache！")
            print("本阶段将不传入cache_salt，避免session管理的影响")
            print("=" * 80 + "\n")

            summary_without, metrics_without = run_test(
                input_file,
                num_threads=CONFIG["thread_num"],
                enable_release=False,
                max_sessions=CONFIG["max_session"],
                max_turns=CONFIG["max_turns"],
            )

            metrics_file = os.path.join(log_dir,
                                        f"metrics_without_release_TN{TN}_MS{ms_tag}_MT{mt_tag}_{timestamp}.json")
            save_metrics(summary_without, metrics_without, metrics_file)

            print("\n" + "=" * 80)
            print("阶段2完成！")
            print("=" * 80)
            print("\n请按以下步骤继续:")
            print("1. 修改代码中的 flag=3")
            print("2. 更新 WITH_RELEASE_FILE 和 WITHOUT_RELEASE_FILE 路径")
            print("3. 重新运行此脚本生成对比报告")
            print("=" * 80 + "\n")
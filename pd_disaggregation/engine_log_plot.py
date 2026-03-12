"""
Engine 日志解析与绘图
从指定目录下的 prefill.log / decode.log 中解析 Engine 指标行，汇总为原始数据并绘制折线图，
输出到单个 Excel 文件（含原始数据 + 三张折线图）。

日志行示例：
  Engine 000: Avg prompt throughput: 80.2 tokens/s, Avg generation throughput: 0.1 tokens/s,
  Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.6%, Prefix cache hit rate: 74.7%,
  External prefix cache hit rate: 0.0%

用法：
  python engine_log_plot.py <log_dir> [--output OUTPUT.xlsx]
  python engine_log_plot.py log/sharegpt_10_yyz_260312/batch_40
"""

import re
import os
import argparse
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.series import SeriesLabel


# 解析正则：从包含 "Engine 000:" 的行中提取各字段
ENGINE_LINE_PATTERN = re.compile(
    r"Engine\s+\d+:"
    r"\s*Avg prompt throughput:\s*([\d.]+)\s*tokens/s,"
    r"\s*Avg generation throughput:\s*([\d.]+)\s*tokens/s,"
    r"\s*Running:\s*(\d+)\s*reqs,"
    r"\s*Waiting:\s*(\d+)\s*reqs,"
    r"\s*GPU KV cache usage:\s*([\d.]+)%,"
    r"\s*Prefix cache hit rate:\s*([\d.]+)%,"
    r"\s*External prefix cache hit rate:\s*([\d.]+)%",
    re.IGNORECASE,
)

COLUMNS = [
    "avg_prompt_throughput",
    "avg_generation_throughput",
    "running",
    "waiting",
    "gpu_kv_cache_usage_pct",
    "prefix_cache_hit_rate_pct",
    "external_prefix_cache_hit_rate_pct",
]


def parse_engine_line(line):
    """从一行日志中解析出指标字典，不匹配则返回 None。"""
    m = ENGINE_LINE_PATTERN.search(line)
    if not m:
        return None
    return {
        "avg_prompt_throughput": float(m.group(1)),
        "avg_generation_throughput": float(m.group(2)),
        "running": int(m.group(3)),
        "waiting": int(m.group(4)),
        "gpu_kv_cache_usage_pct": float(m.group(5)),
        "prefix_cache_hit_rate_pct": float(m.group(6)),
        "external_prefix_cache_hit_rate_pct": float(m.group(7)),
    }


def load_log_series(log_path):
    """从单个日志文件中按行解析所有 Engine 指标，返回列表 of dict。"""
    rows = []
    if not os.path.isfile(log_path):
        return rows
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            d = parse_engine_line(line)
            if d is not None:
                rows.append(d)
    return rows


def load_prefill_decode(log_dir):
    """从 log_dir 下读取 prefill.log 和 decode.log，返回 (prefill_rows, decode_rows)。"""
    log_dir = os.path.abspath(log_dir)
    prefill_path = os.path.join(log_dir, "prefill.log")
    decode_path = os.path.join(log_dir, "decode.log")
    prefill_rows = load_log_series(prefill_path)
    decode_rows = load_log_series(decode_path)
    return prefill_rows, decode_rows


# ── Excel 样式 ─────────────────────────────────────────────────────────────────
FONT_NAME = "Arial"
HDR_FILL = PatternFill("solid", start_color="1F4E79")
HDR_FONT = Font(name=FONT_NAME, bold=True, color="FFFFFF", size=10)
BORDER = Border(
    left=Side(border_style="thin", color="BFBFBF"),
    right=Side(border_style="thin", color="BFBFBF"),
    top=Side(border_style="thin", color="BFBFBF"),
    bottom=Side(border_style="thin", color="BFBFBF"),
)
CENTER = Alignment(horizontal="center", vertical="center")


def _max_len(prefill_rows, decode_rows):
    return max(len(prefill_rows), len(decode_rows), 1)


def _pad(rows, length, fill=None):
    """将 rows 填充到 length 长度，不足用 fill。"""
    return list(rows) + [fill] * (length - len(rows))


def write_raw_sheet(wb, prefill_rows, decode_rows):
    """写入原始数据：两个 sheet Prefill / Decode。"""
    # Prefill
    ws_prefill = wb.create_sheet("Prefill 原始数据", 0)
    headers = ["index"] + COLUMNS
    for c, h in enumerate(headers, 1):
        cell = ws_prefill.cell(row=1, column=c, value=h)
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = CENTER
        cell.border = BORDER
    for r, row in enumerate(prefill_rows, 2):
        ws_prefill.cell(row=r, column=1, value=r - 2)
        for c, key in enumerate(COLUMNS, 2):
            v = row.get(key)
            ws_prefill.cell(row=r, column=c, value=v)
    # Decode
    ws_decode = wb.create_sheet("Decode 原始数据", 1)
    for c, h in enumerate(headers, 1):
        cell = ws_decode.cell(row=1, column=c, value=h)
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = CENTER
        cell.border = BORDER
    for r, row in enumerate(decode_rows, 2):
        ws_decode.cell(row=r, column=1, value=r - 2)
        for c, key in enumerate(COLUMNS, 2):
            v = row.get(key)
            ws_decode.cell(row=r, column=c, value=v)


# 图 sheet：第一个数据区起始行；第二个数据区紧接在第一个之后
CHART_SHEET_DATA_START_ROW = 27
# 第二个图锚点：与第一个图左右并排（P 列约在第一个图右侧）
CHART2_ANCHOR = "P1"

# 采样图最多显示的点数
MAX_CHART_POINTS = 120


def _downsample_indices(n, max_pts):
    """
    返回用于绘图的索引列表。若 n <= max_pts 则全选；否则均匀采样，保留首尾，共约 max_pts 个点。
    """
    if n <= max_pts:
        return list(range(n))
    step = (n - 1) / (max_pts - 1)
    indices = [0]
    for i in range(1, max_pts - 1):
        indices.append(int(round(i * step)))
    indices.append(n - 1)
    return indices


def _write_data_block(ws, row0, headers, columns_subset, p_vals_by_col, d_vals_by_col, indices):
    """在 sheet 的 row0 起写入一块表头+数据（indices 为行索引列表，若为 None 表示 0..len-1 全要）。"""
    n_rows = len(indices)
    idx_col = 1
    data_start_col = 2
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=row0, column=c, value=h)
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = CENTER
        cell.border = BORDER
    for ri in range(n_rows):
        idx = indices[ri]
        sheet_row = row0 + 1 + ri
        ws.cell(row=sheet_row, column=idx_col, value=idx)
        for ci, col in enumerate(columns_subset):
            prefill_col = data_start_col + ci * 2
            decode_col = data_start_col + ci * 2 + 1
            pv = p_vals_by_col[col][idx]
            dv = d_vals_by_col[col][idx]
            ws.cell(row=sheet_row, column=prefill_col, value=pv if pv is not None else "")
            ws.cell(row=sheet_row, column=decode_col, value=dv if dv is not None else "")
    return n_rows


def _build_line_chart(ws, row0, n_rows, columns_subset, titles, chart_title, num_metrics, palette):
    """根据已有数据区 (row0 表头, row0+1..row0+n_rows 数据) 建折线图并返回 chart。"""
    data_start_col = 2
    chart = LineChart()
    chart.title = chart_title
    chart.style = 10
    chart.y_axis.title = titles[0] if len(titles) == 1 else "Value"
    chart.width = 22
    chart.height = 12
    cats = Reference(ws, min_col=1, min_row=row0 + 1, max_row=row0 + n_rows)
    chart.set_categories(cats)
    for ci in range(num_metrics):
        prefill_col = data_start_col + ci * 2
        decode_col = data_start_col + ci * 2 + 1
        title_prefill = f"prefill {titles[ci]}"
        title_decode = f"decode {titles[ci]}"
        ref_p = Reference(ws, min_col=prefill_col, min_row=row0, max_row=row0 + n_rows)
        ref_d = Reference(ws, min_col=decode_col, min_row=row0, max_row=row0 + n_rows)
        chart.add_data(ref_p, titles_from_data=True)
        chart.series[-1].tx = SeriesLabel(v=title_prefill)
        chart.series[-1].smooth = True
        chart.add_data(ref_d, titles_from_data=True)
        chart.series[-1].tx = SeriesLabel(v=title_decode)
        chart.series[-1].smooth = True
    _style_series_pairwise(chart, num_metrics, palette)
    return chart


def _chart_data_sheet(wb, prefill_rows, decode_rows, name, columns_subset, titles):
    """
    一个 sheet 两张图：上为原始数据折线图，下为采样折线图；下方对应两块数据区。
    """
    n = _max_len(prefill_rows, decode_rows)
    indices_sampled = _downsample_indices(n, MAX_CHART_POINTS)
    n_plot = len(indices_sampled)
    headers = ["index"]
    for col in columns_subset:
        headers.append(f"prefill_{col}")
        headers.append(f"decode_{col}")

    p_vals_by_col = {col: _pad([row[col] for row in prefill_rows], n) for col in columns_subset}
    d_vals_by_col = {col: _pad([row[col] for row in decode_rows], n) for col in columns_subset}

    ws = wb.create_sheet(name)
    idx_col = 1
    data_start_col = 2
    num_metrics = len(columns_subset)
    palette = _chart_palette(num_metrics)

    # 数据区1：原始（全部 n 行）
    row0_full = CHART_SHEET_DATA_START_ROW
    _write_data_block(ws, row0_full, headers, columns_subset, p_vals_by_col, d_vals_by_col, list(range(n)))
    n_full = n

    # 数据区2：采样（n_plot 行）
    row0_sampled = row0_full + n_full + 2
    _write_data_block(ws, row0_sampled, headers, columns_subset, p_vals_by_col, d_vals_by_col, indices_sampled)
    n_sampled = n_plot

    # 图1：原始数据，放在顶部
    chart1 = _build_line_chart(ws, row0_full, n_full, columns_subset, titles, f"{name} (原始数据)", num_metrics, palette)
    ws.add_chart(chart1, "A1")
    # 图2：采样，放在图1下方
    chart2 = _build_line_chart(ws, row0_sampled, n_sampled, columns_subset, titles, f"{name} (采样)", num_metrics, palette)
    ws.add_chart(chart2, CHART2_ANCHOR)
    return ws


# 线宽（EMU）：数值越小越细，约 25000～30000 为较细折线
LINE_WIDTH_EMU = 28000


def _chart_palette(num_metrics):
    """
    按指标数量返回颜色对列表：每指标一色系，(深色, 浅色) 用于 prefill / decode。
    不同指标用不同色系，同指标同色系便于对比。
    """
    # 每行 (深色hex, 浅色hex)：蓝、绿、红、紫、橙
    full = [
        ("1B5E9E", "64B5F6"),   # 蓝
        ("2E7D32", "81C784"),   # 绿
        ("C62828", "E57373"),   # 红
        ("6A1B9A", "B39DDB"),   # 紫
        ("E65100", "FFB74D"),   # 橙
    ]
    return full[:num_metrics]


def _style_series_pairwise(chart, num_metrics, colors):
    """
    按「指标」成对设置样式：同指标同色系（prefill 深色、decode 浅色），全部实线，折线较细。
    series 顺序为 [prefill_m0, decode_m0, prefill_m1, decode_m1, ...]
    """
    for i in range(len(chart.series)):
        metric_idx = i // 2
        is_decode = (i % 2) == 1
        pair = colors[metric_idx] if metric_idx < len(colors) else ("333333", "999999")
        color = pair[1] if is_decode else pair[0]
        line = chart.series[i].graphicalProperties.line
        line.solidFill = color
        line.dashStyle = "solid"
        line.width = LINE_WIDTH_EMU


def write_charts(wb, prefill_rows, decode_rows):
    """创建三张图的数据表并插入折线图。"""
    # 图1: Avg prompt throughput, Avg generation throughput
    _chart_data_sheet(
        wb,
        prefill_rows,
        decode_rows,
        name="图1_Throughput",
        columns_subset=["avg_prompt_throughput", "avg_generation_throughput"],
        titles=["Avg prompt throughput (tokens/s)", "Avg generation throughput (tokens/s)"],
    )
    # 图2: Running, Waiting
    _chart_data_sheet(
        wb,
        prefill_rows,
        decode_rows,
        name="图2_Running_Waiting",
        columns_subset=["running", "waiting"],
        titles=["Running (reqs)", "Waiting (reqs)"],
    )
    # 图3: GPU KV cache usage, Prefix cache hit rate, External prefix cache hit rate
    _chart_data_sheet(
        wb,
        prefill_rows,
        decode_rows,
        name="图3_Cache",
        columns_subset=[
            "gpu_kv_cache_usage_pct",
            "prefix_cache_hit_rate_pct",
            "external_prefix_cache_hit_rate_pct",
        ],
        titles=[
            "GPU KV cache usage (%)",
            "Prefix cache hit rate (%)",
            "External prefix cache hit rate (%)",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="解析 prefill/decode 日志并生成 Excel 报表与折线图")
    parser.add_argument("log_dir", type=str, help="包含 prefill.log 和 decode.log 的目录路径")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出 Excel 路径，默认在 log_dir 下 engine_log_analysis.xlsx",
    )
    args = parser.parse_args()
    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"错误：目录不存在 {log_dir}")
        return 1
    out_path = args.output
    if not out_path:
        out_path = os.path.join(log_dir, "engine_log_analysis.xlsx")
    else:
        out_path = os.path.abspath(out_path)

    prefill_rows, decode_rows = load_prefill_decode(log_dir)
    print(f"Prefill 解析行数: {len(prefill_rows)}")
    print(f"Decode 解析行数: {len(decode_rows)}")
    if not prefill_rows and not decode_rows:
        print("未解析到任何 Engine 指标行，请确认日志格式。")
        return 1

    wb = openpyxl.Workbook()
    # 删除默认 sheet，由我们按顺序创建
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    write_raw_sheet(wb, prefill_rows, decode_rows)
    write_charts(wb, prefill_rows, decode_rows)
    wb.save(out_path)
    print(f"已保存: {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())

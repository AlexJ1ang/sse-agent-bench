"""Trace 持久化与回放支撑。

每次评测运行结束后，把每个 case 的完整 trace 序列化为 JSONL
（一行一个 case），内容包括：

- 输入 payload（构建后的请求体）
- 全部 SSE 事件（事件名、payload、接收时间戳）
- 工具调用链（名称、顺序、参数、耗时）
- stage timings（各工作流阶段耗时）
- 最终回答、评分结果、重复采样快照

目录结构：``harness_output/traces/<run_id>/<case_id>.json``

这些数据在 CaseRunner 采集阶段已全部具备，此处为「零额外采集成本」的
持久化层，支撑 ``replay`` 命令做失败用例一键回放与三维 diff。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_harness.models import CaseResult, RunResult, SSEStreamResult


def _serialize_stream(stream: SSEStreamResult) -> dict[str, Any]:
    """序列化 SSE 流结果（去掉 perf_counter 绝对时间戳的噪声，保留顺序）。"""
    return {
        "events": [
            {"event": e.event, "payload": e.payload, "received_at": e.received_at}
            for e in stream.events
        ],
        "answer_parts": list(stream.answer_parts),
        "full_answer": stream.full_answer,
        "tool_calls": [
            {
                "call_id": tc.call_id,
                "name": tc.name,
                "started_at": tc.started_at,
                "completed_at": tc.completed_at,
                "duration_ms": tc.duration_ms,
                "arguments": tc.arguments,
            }
            for tc in stream.tool_calls
        ],
        "stage_timings": [
            {
                "stage": st.stage,
                "phase": st.phase,
                "step_duration_ms": st.step_duration_ms,
                "elapsed_ms": st.elapsed_ms,
            }
            for st in stream.stage_timings
        ],
        "workflow_node_timings_ms": stream.workflow_node_timings_ms,
        "error_events": list(stream.error_events),
        "trace_id": stream.trace_id,
        "stream_interrupted": stream.stream_interrupted,
    }


def _serialize_case(case_result: CaseResult) -> dict[str, Any]:
    """序列化单条用例的完整 trace。"""
    case = case_result.case
    eval_score = case_result.eval_score
    # 优先持久化真实请求体；旧数据或缺失时退回由 case 定义重建。
    payload = case_result.request_payload or {
        "case_id": case.id,
        "question": case.question,
        "context": case.context,
        "expectations": case.expectations.model_dump(mode="json"),
        "tags": case.tags,
        "metadata": case.metadata,
    }

    return {
        "case_id": case.id,
        "payload": payload,
        "stream": _serialize_stream(case_result.stream),
        "latency": case_result.latency.model_dump(mode="json"),
        "eval_score": eval_score.model_dump(mode="json") if eval_score else None,
        "error": case_result.error,
        "success": case_result.success,
        "retries": case_result.retries,
        "repeat_count": case_result.repeat_count,
        "mean_score": case_result.mean_score,
        "std_score": case_result.std_score,
        "pass_rate": case_result.pass_rate,
        "flaky": case_result.flaky,
        "repeats": [
            {
                "attempt": r.attempt,
                "success": r.success,
                "overall_score": r.overall_score,
                "overall_passed": r.overall_passed,
                "latency_ms": r.latency_ms,
                "answer_preview": r.answer_preview,
                "error": r.error,
            }
            for r in case_result.repeats
        ],
    }


def save_run_traces(run_result: RunResult, traces_dir: Path) -> list[Path]:
    """把一次运行的所有 case trace 落盘为 JSON 文件。

    Args:
        run_result: 一次完整的评测运行结果
        traces_dir: traces 根目录（内部会按 run_id 分目录）

    Returns:
        已写入的文件路径列表
    """
    run_dir = traces_dir / run_result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for cr in run_result.case_results:
        path = run_dir / f"{cr.case.id}.json"
        data = _serialize_case(cr)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        paths.append(path)

    # 附带一份 run 级元数据
    meta_path = run_dir / "_run.json"
    meta = {
        "run_id": run_result.run_id,
        "mode": run_result.mode,
        "started_at": run_result.started_at,
        "completed_at": run_result.completed_at,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config_summary": run_result.config_summary,
        "summary": run_result.summary,
        "case_count": len(run_result.case_results),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    paths.append(meta_path)

    return paths


def load_trace(traces_dir: Path, run_id: str, case_id: str) -> dict[str, Any] | None:
    """加载指定 run 下某个 case 的 trace。

    Returns:
        trace dict；不存在时返回 None
    """
    path = traces_dir / run_id / f"{case_id}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_run_ids(traces_dir: Path) -> list[str]:
    """列出所有已持久化的 run id（按目录名）。"""
    if not traces_dir.exists():
        return []
    return sorted(
        p.name for p in traces_dir.iterdir() if p.is_dir()
    )


def list_case_ids(traces_dir: Path, run_id: str) -> list[str]:
    """列出某个 run 下的所有 case id（排除 _run.json 元数据）。"""
    run_dir = traces_dir / run_id
    if not run_dir.exists():
        return []
    return sorted(
        p.stem for p in run_dir.glob("*.json") if p.stem != "_run"
    )


# ── 三维 diff ────────────────────────────────────────────────


def _extract_tool_sequence(stream: dict[str, Any]) -> list[dict[str, Any]]:
    """从 trace 的 stream 段提取工具调用序列（名称 + 参数，忽略绝对时间）。"""
    tool_calls = stream.get("tool_calls", [])
    return [
        {"name": tc.get("name", "unknown"), "arguments": tc.get("arguments", {})}
        for tc in tool_calls
    ]


def _extract_stage_timings(stream: dict[str, Any]) -> dict[str, float]:
    """从 trace 的 stream 段提取 stage -> 耗时（step_duration_ms）。"""
    timings: dict[str, float] = {}
    for st in stream.get("stage_timings", []):
        key = f"{st.get('stage', '')}:{st.get('phase', '')}".strip(":")
        timings[key] = st.get("step_duration_ms", 0.0)
    # 兜底：workflow node timings 也纳入
    for node, ms in (stream.get("workflow_node_timings_ms") or {}).items():
        timings.setdefault(f"[node]{node}", ms)
    return timings


def build_trace_diff(old_trace: dict[str, Any], new_trace: dict[str, Any]) -> dict[str, Any]:
    """构建两个 trace 的三维 diff 报告。

    维度：
    1. 回答文本 diff（difflib 高亮，产出统一 diff 文本）
    2. 工具调用序列对比（顺序 + 参数差异）
    3. 耗时对比（旧 vs 新各阶段耗时表）
    """
    import difflib

    # 1. 回答 diff
    old_answer = old_trace.get("stream", {}).get("full_answer", "")
    new_answer = new_trace.get("stream", {}).get("full_answer", "")
    answer_diff = "\n".join(
        difflib.unified_diff(
            old_answer.splitlines(keepends=True),
            new_answer.splitlines(keepends=True),
            fromfile="old",
            tofile="new",
            lineterm="",
        )
    ) if old_answer != new_answer else ""

    # 2. 工具调用序列
    old_tools = _extract_tool_sequence(old_trace.get("stream", {}))
    new_tools = _extract_tool_sequence(new_trace.get("stream", {}))

    old_names = [t["name"] for t in old_tools]
    new_names = [t["name"] for t in new_tools]
    sm = difflib.SequenceMatcher(a=old_names, b=new_names)
    tool_opcodes = [
        {"tag": tag, "old": old_names[i1:i2], "new": new_names[j1:j2]}
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
    ]

    # 参数差异：同名工具逐一比较
    arg_diffs: list[dict[str, Any]] = []
    old_by_name: dict[str, dict[str, Any]] = {}
    new_by_name: dict[str, dict[str, Any]] = {}
    for t in old_tools:
        old_by_name.setdefault(t["name"], t)
    for t in new_tools:
        new_by_name.setdefault(t["name"], t)
    for name in set(old_by_name) & set(new_by_name):
        if old_by_name[name]["arguments"] != new_by_name[name]["arguments"]:
            arg_diffs.append({
                "tool": name,
                "old_arguments": old_by_name[name]["arguments"],
                "new_arguments": new_by_name[name]["arguments"],
            })

    tool_changed = (
        old_names != new_names or len(arg_diffs) > 0
    )

    # 3. 耗时对比
    old_stages = _extract_stage_timings(old_trace.get("stream", {}))
    new_stages = _extract_stage_timings(new_trace.get("stream", {}))
    stage_keys = sorted(set(old_stages) | set(new_stages))
    latency_diff = [
        {
            "stage": k,
            "old_ms": old_stages.get(k),
            "new_ms": new_stages.get(k),
        }
        for k in stage_keys
    ]
    # 只在存在任一 stage 的耗时数值变化时才算「耗时变化」
    latency_changed = any(
        item["old_ms"] != item["new_ms"] for item in latency_diff
    )

    # 4. 评分 / 状态差异
    old_score = old_trace.get("eval_score") or {}
    new_score = new_trace.get("eval_score") or {}
    score_change = {
        "old_overall": old_score.get("overall_score"),
        "new_overall": new_score.get("overall_score"),
        "old_passed": old_score.get("overall_passed"),
        "new_passed": new_score.get("overall_passed"),
    }

    changed = bool(answer_diff) or tool_changed or latency_changed

    return {
        "changed": changed,
        "answer_changed": bool(answer_diff),
        "answer_diff": answer_diff,
        "tool_changed": tool_changed,
        "tool_opcodes": tool_opcodes,
        "tool_arg_diffs": arg_diffs,
        "latency_changed": latency_changed,
        "latency_diff": latency_diff,
        "score_change": score_change,
    }

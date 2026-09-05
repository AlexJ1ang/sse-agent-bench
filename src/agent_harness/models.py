"""核心数据模型：测试用例、SSE 事件、执行结果、评分结果。

所有模型均为通用设计，不绑定特定业务领域。
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ── 测试用例 ──────────────────────────────────────────────────


class ExpectationSpec(BaseModel):
    """对一条用例的通用期望声明。

    所有字段均可选，未填的维度在评分时自动跳过。
    这是「能程序硬校验的绝不用 LLM 判」原则的载体：
    用户把确定性期望写在这里，程序照单校验。
    """

    # 意图/路由期望：被测 Agent 若在意图识别阶段输出分类标签，
    # 可在这里指定期望值。key 为层级名（如 main/sub），value 为期望标签。
    expected_intents: dict[str, str] = Field(default_factory=dict)

    # 工具调用期望：期望被调用的工具名列表（不要求顺序）。
    expected_tools: list[str] = Field(default_factory=list)

    # 回答内容期望：回答中必须出现的关键词/短语列表。
    expected_keywords: list[str] = Field(default_factory=list)

    # 回答内容期望：回答中必须匹配的正则表达式列表。
    expected_patterns: list[str] = Field(default_factory=list)

    # 参考答案：用于事实一致性校验（数值提取对比）和 LLM-Judge 参考。
    reference_answer: str = ""

    # 是否为开放式问题（开放式问题才走 LLM-Judge）。
    is_open_ended: bool = False


class TestCase(BaseModel):
    """通用测试用例 Schema。

    框架不预设字段含义。question 是唯一必填的用户输入，
    context 用于存放构建请求体所需的任意附加字段（用户ID、角色、设备等），
    expectations 声明确定性校验期望。
    """

    id: str
    question: str
    context: dict[str, Any] = Field(default_factory=dict)
    expectations: ExpectationSpec = Field(default_factory=ExpectationSpec)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestSuite(BaseModel):
    """测试用例集合。"""

    id: str
    name: str
    description: str = ""
    cases: list[TestCase] = Field(default_factory=list)


# ── SSE 事件 ──────────────────────────────────────────────────


class SSEEvent(BaseModel):
    """一条 SSE 事件。"""

    event: str
    payload: Any
    received_at: float = Field(default_factory=time.perf_counter)


class ToolCallRecord(BaseModel):
    """一次工具调用记录。"""

    call_id: str = ""
    name: str = "unknown"
    started_at: float = 0.0
    completed_at: float | None = None
    duration_ms: float | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class StageTiming(BaseModel):
    """一个工作流节点的耗时。"""

    stage: str
    phase: str = ""
    step_duration_ms: float = 0.0
    elapsed_ms: float = 0.0


class SSEStreamResult(BaseModel):
    """SSE 流的完整采集结果。"""

    events: list[SSEEvent] = Field(default_factory=list)
    answer_parts: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    stage_timings: list[StageTiming] = Field(default_factory=list)
    workflow_node_timings_ms: dict[str, float] = Field(default_factory=dict)
    error_events: list[str] = Field(default_factory=list)
    trace_id: str = ""
    stream_interrupted: bool = False

    @property
    def full_answer(self) -> str:
        return "".join(self.answer_parts)

    @property
    def event_count(self) -> int:
        return len(self.events)


# ── 延迟指标 ──────────────────────────────────────────────────


class LatencyMetrics(BaseModel):
    """一次请求的延迟指标。"""

    request_total_ms: float | None = None
    request_to_first_answer_ms: float | None = None
    request_to_answer_complete_ms: float | None = None
    answer_output_duration_ms: float | None = None
    answer_generation_total_ms: float | None = None
    workflow_node_timings_ms: dict[str, float] = Field(default_factory=dict)
    tool_timings: list[dict[str, Any]] = Field(default_factory=list)


# ── 评分结果 ──────────────────────────────────────────────────


class ScorerType(str, Enum):
    INTENT = "intent"
    TOOL = "tool"
    FACT = "fact"
    LLM_JUDGE = "llm_judge"


class DimensionScore(BaseModel):
    """单维度评分。"""

    scorer: ScorerType
    score: float
    max_score: float = 1.0
    passed: bool = True
    skipped: bool = False
    details: str = ""
    issues: list[str] = Field(default_factory=list)
    # Judge 自一致性校验失败标记（分数仍记录，仅供报告聚合分歧率）
    # 仅对 LLM_JUDGE 维度有意义，其余维度恒为 False
    low_confidence: bool = False


class EvalScore(BaseModel):
    """一条用例的完整评分。"""

    case_id: str
    dimensions: list[DimensionScore] = Field(default_factory=list)
    overall_score: float = 0.0
    overall_passed: bool = True
    latency: LatencyMetrics | None = None
    error: str = ""


# ── 执行结果 ──────────────────────────────────────────────────


class RepeatResult(BaseModel):
    """重复采样中单次执行的结果快照。"""

    attempt: int
    success: bool = False
    stream: SSEStreamResult = Field(default_factory=SSEStreamResult)
    overall_score: float = 0.0
    overall_passed: bool = False
    latency_ms: float | None = None
    answer_preview: str = ""
    error: str = ""


class CaseResult(BaseModel):
    """一条用例的完整执行结果。

    当 eval.repeat_count > 1 时，repeats 记录每次执行快照，
    本对象的 stream/latency/eval_score 为首次执行的代表值。
    """

    case: TestCase
    stream: SSEStreamResult = Field(default_factory=SSEStreamResult)
    eval_score: EvalScore | None = None
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    error: str = ""
    # 实际发送给被测 Agent 的请求体（trace 持久化与回放用）
    request_payload: dict[str, Any] = Field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = False
    # 本次执行内部的网络层重试次数（最终以最后一次结果为准）
    retries: int = 0

    # ── 重复采样聚合 ──
    repeats: list[RepeatResult] = Field(default_factory=list)
    repeat_count: int = 1
    mean_score: float = 0.0
    std_score: float = 0.0
    pass_rate: float = 0.0
    flaky: bool = False


class RunResult(BaseModel):
    """一次 Harness 运行的完整结果。"""

    run_id: str
    mode: str
    started_at: str = ""
    completed_at: str = ""
    config_summary: dict[str, Any] = Field(default_factory=dict)
    case_results: list[CaseResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


# ── 压测结果 ──────────────────────────────────────────────────


class LoadSample(BaseModel):
    """压测中单条请求的采样。"""

    case_id: str
    started_at: float
    completed_at: float
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    success: bool = True
    error: str = ""
    sse_event_count: int = 0
    answer_length: int = 0
    stream_interrupted: bool = False


class LoadTestResult(BaseModel):
    """压测结果。"""

    run_id: str
    scenario_name: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    samples: list[LoadSample] = Field(default_factory=list)
    latency_percentiles: dict[str, float] = Field(default_factory=dict)
    ttft_percentiles: dict[str, float] = Field(default_factory=dict)
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    interruption_rate: float = 0.0
    latency_breakdown: dict[str, Any] = Field(default_factory=dict)

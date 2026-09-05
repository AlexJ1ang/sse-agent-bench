"""全局配置加载。

从 YAML 配置文件 + 环境变量读取 Harness 运行参数。
所有与被测 Agent 的交互细节（请求体结构、SSE 事件语义、校验规则）
均通过 adapter 配置注入，框架本身不含任何业务绑定。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ── 被测系统适配 ──────────────────────────────────────────────


class RequestAdapterConfig(BaseModel):
    """如何把一条 TestCase 转成被测 Agent 的 HTTP 请求体。

    body_template 是一个 dict，value 支持 {field} 占位符，
    占位符从 TestCase 的通用字段 + context dict 中取值。
    例如：
        {"query": "{question}", "session": "{session_id}", "meta": "{context}"}
    """

    body_template: dict[str, Any] = Field(
        default_factory=lambda: {"query": "{question}"}
    )
    extra_headers: dict[str, str] = Field(default_factory=dict)


class SSEEventMapping(BaseModel):
    """SSE 事件名到框架内部语义的映射。

    被测 Agent 的 SSE 事件名五花八门，通过此配置告知框架：
    哪个事件携带回答文本、哪个表示工具调用、哪个表示结束、哪个表示错误。
    未配置的事件类型仍会被原样采集，只是不参与指标计算。
    """

    answer_event: str = "answer"
    answer_content_field: str = "content"
    tool_call_event: str = "tool_call"
    tool_name_field: str = "name"
    tool_args_field: str = "arguments"
    tool_call_id_field: str = "call_id"
    status_event: str = "status"
    status_stage_field: str = "stage"
    done_event: str = "done"
    error_event: str = "error"
    error_message_field: str = "message"


class TargetConfig(BaseModel):
    """被测 Agent 服务连接配置。"""

    base_url: str = "http://localhost:8000"
    chat_path: str = "/chat"
    method: str = "POST"
    connect_timeout: float = 10.0
    read_timeout: float = 180.0
    request_adapter: RequestAdapterConfig = Field(
        default_factory=RequestAdapterConfig
    )
    sse_mapping: SSEEventMapping = Field(default_factory=SSEEventMapping)


# ── LLM-as-Judge ──────────────────────────────────────────────


class JudgeConfig(BaseModel):
    """LLM-as-Judge 评分配置。"""

    enabled: bool = True
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    chat_path: str = "/chat/completions"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    cache_enabled: bool = True
    system_prompt: str = ""
    pass_threshold: float = 3.0
    # Chain-of-Thought 评分：先逐维度推理（引用原文佐证）再打分
    cot_enabled: bool = True
    # 自一致性校验：同一回答评两次（第二次维度顺序打乱 + 换措辞），
    # 两次加权总分差异 > 阈值则标注 low_confidence（position bias 检测）
    consistency_check: bool = False
    # 自一致性判定阈值（加权总分差异绝对值）
    consistency_threshold: float = 1.0


# ── 评测 / 压测 ───────────────────────────────────────────────


class EvalConfig(BaseModel):
    """质量评测配置。"""

    concurrency: int = 4
    repeat_count: int = 1
    # 单次执行内部的重试次数（默认 0 关闭）。仅对网络层瞬时错误重试：
    # 超时、请求连接错误、SSE 流中断。不会对 4xx/5xx 重试，也不产生新的
    # RepeatResult，因此不影响 flaky 统计（flaky 仍指 repeat 之间的不一致）。
    retry_count: int = 0
    # 重试间隔（毫秒）
    retry_delay_ms: int = 300
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    scorers_enabled: list[str] = Field(
        default_factory=lambda: ["intent", "tool", "fact", "llm_judge"]
    )
    dimension_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "intent": 0.25,
            "tool": 0.25,
            "fact": 0.30,
            "llm_judge": 0.20,
        }
    )


class LoadConfig(BaseModel):
    """性能压测配置。"""

    concurrency: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    think_time_ms: int = 0


class HarnessConfig(BaseModel):
    """Harness 顶层配置。"""

    target: TargetConfig = Field(default_factory=TargetConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    load: LoadConfig = Field(default_factory=LoadConfig)
    output_dir: Path = Path("harness_output")
    baseline_name: str = "latest"


def load_config(config_path: Path | str | None = None) -> HarnessConfig:
    """加载配置文件，环境变量优先于 YAML。"""
    raw: dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

    config = HarnessConfig(**raw)

    env_overrides = {
        "HARNESS_BASE_URL": ("target", "base_url"),
        "HARNESS_CHAT_PATH": ("target", "chat_path"),
        "HARNESS_JUDGE_API_KEY": ("eval", "judge", "api_key"),
        "HARNESS_JUDGE_API_BASE": ("eval", "judge", "api_base"),
        "HARNESS_JUDGE_MODEL": ("eval", "judge", "model"),
    }
    for env_key, parts in env_overrides.items():
        value = os.getenv(env_key)
        if not value:
            continue
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return config

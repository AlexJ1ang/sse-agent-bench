"""请求体构建。

把一条 TestCase 按 target.request_adapter.body_template 渲染为实际的 HTTP 请求体。
占位符解析逻辑在此统一定义，供 CaseRunner（功能评测）与 LoadEngine（性能压测）共用，
避免两处重复实现。
"""

from __future__ import annotations

import uuid
from typing import Any

from agent_harness.config import HarnessConfig
from agent_harness.models import TestCase


def build_payload(config: HarnessConfig, case: TestCase) -> dict[str, Any]:
    """按 target.request_adapter.body_template 构建请求体。

    模板 value 中的 {field} 占位符依次从 TestCase 字段、case.context 中解析。
    特殊占位符：
      {context}    -> 整个 context dict
      {question}   -> 用例问题
      {case_id}    -> 用例 ID
      {session_id} -> 自动生成的 session id（若模板用到）
      {user_id}    -> 自动生成 harness-<case_id>（若模板用到）
    """
    target = config.target
    template = target.request_adapter.body_template
    auto_session = f"harness-{uuid.uuid4().hex[:12]}"
    auto_user = f"harness-{case.id}"

    def resolve(value: Any) -> Any:
        if isinstance(value, str):
            if value == "{context}":
                return case.context
            if value == "{session_id}":
                return auto_session
            if value == "{user_id}":
                return auto_user
            if value == "{question}":
                return case.question
            if value == "{case_id}":
                return case.id
            if value.startswith("{") and value.endswith("}"):
                key = value[1:-1]
                if key in case.context:
                    return case.context[key]
                return value  # 未识别的占位符原样保留
            return value
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return {k: resolve(v) for k, v in template.items()}

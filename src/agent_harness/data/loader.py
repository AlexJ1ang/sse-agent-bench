"""测试用例数据加载器。

支持从 JSON / YAML / CSV 文件加载 TestSuite。
所有格式最终都归一化为通用 TestCase 模型。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from agent_harness.models import ExpectationSpec, TestCase, TestSuite


def load_suite(path: Path) -> TestSuite:
    """从文件加载测试用例集。

    支持的格式：
    - .yaml / .yml : 完整 TestSuite 结构（id/name/description/cases）
    - .json        : 同 YAML 结构，或纯用例数组
    - .csv         : 每行一条用例，列头映射到 TestCase 字段
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"suite file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml_suite(path)
    if suffix == ".json":
        return _load_json_suite(path)
    if suffix == ".csv":
        return _load_csv_suite(path)
    raise ValueError(f"unsupported suite format: {suffix}")


def load_cases(path: Path) -> list[TestCase]:
    """只加载用例列表（不需要 TestSuite 包装）。"""
    return load_suite(path).cases


# ── YAML ──────────────────────────────────────────────────────


def _load_yaml_suite(path: Path) -> TestSuite:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _dict_to_suite(data, default_id=path.stem)


# ── JSON ──────────────────────────────────────────────────────


def _load_json_suite(path: Path) -> TestSuite:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _dict_to_suite(data, default_id=path.stem)


def _dict_to_suite(data: Any, default_id: str) -> TestSuite:
    """把 dict/list 数据转为 TestSuite。"""
    if isinstance(data, list):
        # 纯用例数组
        return TestSuite(
            id=default_id,
            name=default_id,
            cases=[_dict_to_case(c) for c in data],
        )
    if isinstance(data, dict):
        cases_raw = data.get("cases", [])
        return TestSuite(
            id=str(data.get("id") or default_id),
            name=str(data.get("name") or default_id),
            description=str(data.get("description") or ""),
            cases=[_dict_to_case(c) for c in cases_raw],
        )
    raise ValueError(f"invalid suite data type: {type(data)}")


def _dict_to_case(raw: dict[str, Any]) -> TestCase:
    """把单个 dict 转为 TestCase，兼容多种字段命名。"""
    exp_raw = raw.get("expectations") or {}
    expectations = ExpectationSpec(
        expected_intents=dict(exp_raw.get("expected_intents") or {}),
        expected_tools=list(exp_raw.get("expected_tools") or []),
        expected_keywords=list(exp_raw.get("expected_keywords") or []),
        expected_patterns=list(exp_raw.get("expected_patterns") or []),
        reference_answer=str(exp_raw.get("reference_answer") or ""),
        is_open_ended=bool(exp_raw.get("is_open_ended", False)),
    )
    return TestCase(
        id=str(raw.get("id") or raw.get("case_id") or ""),
        question=str(raw.get("question") or raw.get("query") or ""),
        context=dict(raw.get("context") or {}),
        expectations=expectations,
        tags=list(raw.get("tags") or []),
        metadata=dict(raw.get("metadata") or {}),
    )


# ── CSV ───────────────────────────────────────────────────────


def _load_csv_suite(path: Path) -> TestSuite:
    """从 CSV 加载用例。

    CSV 列约定（大小写不敏感，多余列进 context）：
      id / case_id          -> 用例 ID
      question / query      -> 用户问题
      expected_keywords     -> 分号或逗号分隔
      expected_tools        -> 分号或逗号分隔
      expected_intent_main  -> 主意图
      expected_intent_sub   -> 子意图
      reference_answer      -> 参考答案
      is_open_ended         -> true/false/1/0
      tags                  -> 分号或逗号分隔
      其他列                -> 全部进入 context
    """
    cases: list[TestCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            cases.append(_csv_row_to_case(row, idx, path.stem))

    return TestSuite(
        id=path.stem,
        name=path.stem,
        description=f"loaded from {path.name}",
        cases=cases,
    )


def _split_list_cell(value: str | None) -> list[str]:
    """拆分 CSV 中的列表单元格（支持 ; 或 , 分隔）。"""
    if not value:
        return []
    parts = value.replace(";", ",").split(",")
    return [p.strip() for p in parts if p.strip()]


def _csv_row_to_case(row: dict[str, str], idx: int, suite_id: str) -> TestCase:
    """把一行 CSV 转为 TestCase。"""
    # 列名统一小写映射
    lower_row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}

    case_id = lower_row.get("id") or lower_row.get("case_id") or f"{suite_id}-{idx:03d}"
    question = lower_row.get("question") or lower_row.get("query") or ""

    # 意图期望
    expected_intents: dict[str, str] = {}
    for key, val in lower_row.items():
        if key.startswith("expected_intent_") and val:
            expected_intents[key.removeprefix("expected_intent_")] = val
    # 兼容单一层级的 "expected_intent" 列
    if "expected_intent" in lower_row and lower_row["expected_intent"]:
        expected_intents.setdefault("main", lower_row["expected_intent"])

    is_open_ended_raw = (lower_row.get("is_open_ended") or "").lower()
    is_open_ended = is_open_ended_raw in {"true", "1", "yes", "y"}

    expectations = ExpectationSpec(
        expected_intents=expected_intents,
        expected_tools=_split_list_cell(lower_row.get("expected_tools")),
        expected_keywords=_split_list_cell(lower_row.get("expected_keywords")),
        expected_patterns=_split_list_cell(lower_row.get("expected_patterns")),
        reference_answer=lower_row.get("reference_answer") or "",
        is_open_ended=is_open_ended,
    )

    # 保留列以外的字段进 context
    reserved = {
        "id", "case_id", "question", "query",
        "expected_keywords", "expected_tools", "expected_patterns",
        "expected_intent", "reference_answer", "is_open_ended", "tags",
    }
    reserved |= {k for k in lower_row if k.startswith("expected_intent_")}
    context = {k: v for k, v in lower_row.items() if k not in reserved and v}

    return TestCase(
        id=case_id,
        question=question,
        context=context,
        expectations=expectations,
        tags=_split_list_cell(lower_row.get("tags")),
    )

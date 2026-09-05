"""数据加载器单元测试：YAML / JSON / CSV 三种格式。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_harness.data.loader import load_suite


@pytest.fixture
def tmp_suite_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_load_yaml_full_suite(tmp_suite_dir: Path):
    path = tmp_suite_dir / "suite.yaml"
    path.write_text(
        """
id: my-suite
name: My Suite
description: desc
cases:
  - id: c1
    question: q1
    context:
      role_id: OWNER
    expectations:
      expected_intents:
        main: device_query
      expected_keywords:
        - 工时
    tags: [smoke]
""",
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert suite.id == "my-suite"
    assert len(suite.cases) == 1
    case = suite.cases[0]
    assert case.id == "c1"
    assert case.context["role_id"] == "OWNER"
    assert case.expectations.expected_intents == {"main": "device_query"}
    assert case.expectations.expected_keywords == ["工时"]


def test_load_json_pure_array(tmp_suite_dir: Path):
    path = tmp_suite_dir / "cases.json"
    path.write_text(
        json.dumps(
            [
                {"id": "a1", "question": "q1"},
                {"query": "q2", "case_id": "a2"},
            ]
        ),
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert suite.id == "cases"
    assert [c.id for c in suite.cases] == ["a1", "a2"]
    assert suite.cases[1].question == "q2"


def test_load_csv_with_column_mapping(tmp_suite_dir: Path):
    path = tmp_suite_dir / "cases.csv"
    path.write_text(
        "id,question,expected_intent_main,expected_keywords,role_id\n"
        "c1,q1,device_query,工时;累计,OWNER\n",
        encoding="utf-8-sig",
    )
    suite = load_suite(path)
    case = suite.cases[0]
    assert case.id == "c1"
    assert case.expectations.expected_intents == {"main": "device_query"}
    assert case.expectations.expected_keywords == ["工时", "累计"]
    # 多余列进 context
    assert case.context["role_id"] == "OWNER"


def test_missing_file_raises(tmp_suite_dir: Path):
    with pytest.raises(FileNotFoundError):
        load_suite(tmp_suite_dir / "nope.yaml")


def test_unsupported_format_raises(tmp_suite_dir: Path):
    path = tmp_suite_dir / "suite.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_suite(path)

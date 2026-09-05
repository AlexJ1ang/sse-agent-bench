"""合成测试数据（promptfoo 思路）。

基于种子用例，用 LLM 按三类策略生成语义等价的变体，扩充评测集：

- ``paraphrase`` 换问法：同义改写，覆盖更多口语表达。
- ``noise``      加噪声：口语化、错别字、多轮上下文包装。
- ``edge``       边界情况：空值、极端数值、歧义问法等。

核心约定：

- 期望字段自动继承种子（``expected_tools`` / ``expected_intents`` /
  ``expected_patterns`` / ``reference_answer`` / ``is_open_ended``）。
- 仅 ``question`` 与 ``expected_keywords`` 由 LLM 改写，保证改写后仍可硬校验。
- 复用现有 ``JudgeConfig`` 的 LLM 配置，不引入新的 provider 配置。
- 若未配置 Judge API Key（或 LLM 调用失败），退化为确定性的本地规则
  变体器，保证 ``synthesize`` 命令在无网环境下也可用。人工审核仍是
  合并入主用例集前的必选环节。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
import yaml

from agent_harness.config import JudgeConfig
from agent_harness.models import ExpectationSpec, TestCase

logger = logging.getLogger(__name__)

STRATEGIES = ("paraphrase", "noise", "edge")

SYNTH_SYSTEM_PROMPT = """\
你是 Agent 评测集扩充助手。给定一条种子测试用例，生成语义等价的变体问题，\
用于黑盒回归测试的鲁棒性覆盖。

## 三种变异策略

1. paraphrase（换问法）：同义改写，换一种自然的问法，语义完全不变。
2. noise（加噪声）：口语化、加入轻微口头禅或错别字、或带简短多轮上下文包裹。
3. edge（边界情况）：用极端数值、歧义表达或极端条件重新表述。

## 输出要求

严格只输出一个 JSON 数组，不要输出其他内容。数组每个元素是：
{"strategy": "paraphrase|noise|edge", "question": "改写后的问题", "keywords": ["关键词1", "关键词2"]}

其中 keywords 是改写后问题里应当保留/出现的 1~3 个关键实体或短语，\
用于确定性关键词校验；如无合适关键词则为空数组。
"""

SYNTH_USER_TEMPLATE = """\
## 种子问题

{question}

## 种子上下文

{context}

## 种子确定性期望（继承，无需改写）

expected_tools: {expected_tools}
expected_intents: {expected_intents}
expected_patterns: {expected_patterns}

## 任务

为上面的种子问题生成 {count} 个变体，策略依次为：{strategies}。
只改写问题与关键词，其他期望字段由程序自动继承。
"""


# ── LLM 变体生成 ──────────────────────────────────────────────


def _llm_payload(judge: JudgeConfig, system: str, user: str) -> dict[str, Any]:
    return {
        "model": judge.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": judge.temperature,
        "max_tokens": judge.max_tokens,
        "stream": False,
    }


async def _call_llm(judge: JudgeConfig, system: str, user: str) -> str | None:
    """调用 OpenAI 兼容接口，返回文本内容；失败返回 None。"""
    url = f"{judge.api_base}{judge.chat_path}"
    headers = {"Authorization": f"Bearer {judge.api_key}"} if judge.api_key else {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        try:
            resp = await client.post(url, json=_llm_payload(judge, system, user), headers=headers)
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            logger.exception("synthesize LLM call failed")
            return None
    choices = body.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message", {})
    return str(message.get("content") or "") or None


def _parse_variants(raw: str) -> list[dict[str, Any]]:
    """从 LLM 输出中解析变体 JSON 数组。"""
    text = raw.strip()
    # 剥离可能的 ```json 围栏
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 退化：取第一个 [ 到最后一个 ]
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and item.get("question")]


async def _llm_variants(
    case: TestCase,
    judge: JudgeConfig,
    strategies: list[str],
    count: int,
) -> list[dict[str, Any]]:
    """用 LLM 为单条种子生成变体。"""
    if not judge.api_key:
        return []
    user = SYNTH_USER_TEMPLATE.format(
        question=case.question,
        context=json.dumps(case.context, ensure_ascii=False) if case.context else "（无）",
        expected_tools=case.expectations.expected_tools,
        expected_intents=case.expectations.expected_intents,
        expected_patterns=case.expectations.expected_patterns,
        count=count,
        strategies="、".join(strategies),
    )
    raw = await _call_llm(judge, SYNTH_SYSTEM_PROMPT, user)
    if not raw:
        return []
    return _parse_variants(raw)


# ── 确定性本地变体器（无 LLM / LLM 失败时的回退）───────────────


_FILLER_PREFIX = ("请问一下", "麻烦帮我看看", "帮我看一下", "我想问下", "咨询一下")
_FILLER_SUFFIX = ("呗", "哈", "哦", "呀", "呢")
_TYPO_MAP = {"的": "滴", "吗": "嘛", "什么": "啥", "怎么": "咋", "可以": "能"}


def _rule_variants(
    case: TestCase,
    strategies: list[str],
    count: int,
) -> list[dict[str, Any]]:
    """基于确定性规则生成变体（不依赖 LLM）。"""
    q = case.question.strip()
    kws = list(case.expectations.expected_keywords)
    variants: list[dict[str, Any]] = []
    for i in range(count):
        strategy = strategies[i % len(strategies)]
        if strategy == "paraphrase":
            new_q = f"请帮我查询：{q}"
            new_kws = kws
        elif strategy == "noise":
            new_q = f"{_FILLER_PREFIX[i % len(_FILLER_PREFIX)]}，{q}{_FILLER_SUFFIX[i % len(_FILLER_SUFFIX)]}"
            typo_q = new_q
            for a, b in _TYPO_MAP.items():
                if a in typo_q:
                    typo_q = typo_q.replace(a, b, 1)
                    break
            new_q = typo_q
            new_kws = kws
        else:  # edge
            new_q = f"如果临时把所有设备都算进来，{q}（请给我一个明确结论）"
            new_kws = kws
        variants.append({"strategy": strategy, "question": new_q, "keywords": new_kws})
    return variants


# ── 组装 ──────────────────────────────────────────────────────


def _inherit_expectations(
    case: TestCase,
    new_keywords: list[str] | None,
) -> ExpectationSpec:
    """继承种子期望，仅关键词可用改写结果覆盖。"""
    base = case.expectations
    keywords = new_keywords if new_keywords else list(base.expected_keywords)
    return ExpectationSpec(
        expected_intents=dict(base.expected_intents),
        expected_tools=list(base.expected_tools),
        expected_keywords=[str(k) for k in keywords],
        expected_patterns=list(base.expected_patterns),
        reference_answer=base.reference_answer,
        is_open_ended=base.is_open_ended,
    )


def _to_case(
    seed: TestCase,
    variant: dict[str, Any],
    strategy: str,
    seq: int,
) -> TestCase:
    return TestCase(
        id=f"{seed.id}-{strategy[:4]}-{seq:03d}",
        question=str(variant["question"]).strip(),
        context=dict(seed.context),
        expectations=_inherit_expectations(seed, variant.get("keywords")),
        tags=list(seed.tags),
        metadata=dict(seed.metadata) | {"synthesized_from": seed.id, "strategy": strategy},
    )


async def synthesize(
    seeds: list[TestCase],
    judge: JudgeConfig,
    *,
    strategies: list[str],
    count: int | None = None,
    per_seed: int | None = None,
) -> list[TestCase]:
    """基于种子用例合成变体。

    变体数量由 ``per_seed`` 或 ``count`` 控制，二者互斥（均不传则用默认）：

    - ``per_seed=N``：每个种子生成 N 条变体（最直观的写法）。
    - ``count=N``：变体**总量**目标，在种子间均分（向上取整，总量可能略超 N）。
    - 均不传：每个种子生成「策略数」条（默认 3 条）。

    无论哪种方式，都不会让单个种子无限膨胀。

    Args:
        seeds: 种子用例列表
        judge: 复用的 LLM 配置（JudgeConfig）
        strategies: 参与的变异策略子集
        count: 变体总量目标（None 表示不受总量约束）
        per_seed: 每个种子生成的变体数（None 表示不用该约束）

    Returns:
        合成后的变体用例列表（不含种子本身）
    """
    strategies = [s for s in strategies if s in STRATEGIES] or list(STRATEGIES)
    n_seeds = len(seeds)
    if n_seeds == 0:
        return []
    if per_seed is not None:
        per_seed = max(1, per_seed)
    elif count is not None:
        per_seed = max(1, -(-count // n_seeds))  # 总量均分，向上取整
    else:
        per_seed = len(strategies)  # 默认：每个种子 = 策略数

    results: list[TestCase] = []
    for seed in seeds:
        variants = await _llm_variants(seed, judge, strategies, per_seed)
        used_fallback = False
        if not variants:
            variants = _rule_variants(seed, strategies, per_seed)
            used_fallback = True
        elif len(variants) < per_seed:
            # 补齐不足部分
            variants += _rule_variants(seed, strategies, per_seed - len(variants))
            used_fallback = True
        if used_fallback:
            logger.info("种子 %s 使用本地规则变体器补齐", seed.id)
        strategy_seq: dict[str, int] = {}
        for v in variants[:per_seed]:
            strat = v.get("strategy") or strategies[0]
            if strat not in strategies:
                strat = strategies[0]
            strategy_seq[strat] = strategy_seq.get(strat, 0) + 1
            results.append(_to_case(seed, v, strat, strategy_seq[strat]))
    return results


def build_suite_yaml(
    seeds: list[TestCase],
    synthesized: list[TestCase],
) -> str:
    """把「种子 + 合成变体」拼成一个 TestSuite 的 YAML 文本。"""

    def case_to_dict(case: TestCase) -> dict[str, Any]:
        exp = case.expectations
        d: dict[str, Any] = {"id": case.id, "question": case.question}
        if case.context:
            d["context"] = case.context
        expd: dict[str, Any] = {}
        if exp.expected_intents:
            expd["expected_intents"] = exp.expected_intents
        if exp.expected_tools:
            expd["expected_tools"] = exp.expected_tools
        if exp.expected_keywords:
            expd["expected_keywords"] = exp.expected_keywords
        if exp.expected_patterns:
            expd["expected_patterns"] = exp.expected_patterns
        if exp.reference_answer:
            expd["reference_answer"] = exp.reference_answer
        expd["is_open_ended"] = exp.is_open_ended
        if expd:
            d["expectations"] = expd
        if case.tags:
            d["tags"] = case.tags
        if case.metadata:
            d["metadata"] = case.metadata
        return d

    suite = {
        "id": "synthesized",
        "name": "Synthesized Eval Suite",
        "description": (
            f"由 {len(seeds)} 条种子合成 {len(synthesized)} 条变体；"
            "请人工审核后合并入主用例集"
        ),
        "cases": [case_to_dict(c) for c in synthesized],
    }
    return yaml.safe_dump(suite, allow_unicode=True, sort_keys=False, default_flow_style=False)


def write_suite_yaml(path: Path, text: str) -> None:
    """写出合成用例集 YAML。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# NOTE: 请人工审核后合并入主用例集\n" + text, encoding="utf-8")
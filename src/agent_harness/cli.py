"""Agent Harness CLI 入口。

用法：
    agent-harness eval --config config.yaml --suite cases.yaml
    agent-harness load --config config.yaml --suite cases.yaml
    agent-harness report --input harness_output/report_xxx.json --format html
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from agent_harness.config import load_config
from agent_harness.core.baseline import BaselineStore
from agent_harness.core.runner import BatchRunner
from agent_harness.data.loader import load_suite
from agent_harness.data.synthesizer import (
    STRATEGIES,
    build_suite_yaml,
    synthesize,
    write_suite_yaml,
)
from agent_harness.eval.pipeline import EvalPipeline
from agent_harness.load.engine import LoadEngine
from agent_harness.report.generator import ReportGenerator

logger = logging.getLogger(__name__)

STRATEGY_CHOICES = list(STRATEGIES)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="agent-harness",
        description="LLM Agent 质量保障 Harness：功能评测 + 性能压测一体化框架",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="输出调试日志"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # eval 命令
    eval_parser = subparsers.add_parser("eval", help="执行功能评测")
    eval_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="配置文件路径"
    )
    eval_parser.add_argument(
        "--suite", "-s", type=Path, required=True, help="测试用例集路径"
    )
    eval_parser.add_argument(
        "--concurrency", "-j", type=int, default=None, help="并发数（覆盖配置）"
    )
    eval_parser.add_argument(
        "--output", "-o", type=Path, default=None, help="报告输出目录"
    )
    eval_parser.add_argument(
        "--format", "-f", nargs="+", choices=["json", "html"], default=["json", "html"],
        help="报告格式",
    )
    eval_parser.add_argument(
        "--gate", action="store_true", help="启用 CI 质量门禁（不满足则退出码非 0）"
    )
    eval_parser.add_argument(
        "--gate-min-score", type=float, default=None, help="门禁：整体平均分下限"
    )
    eval_parser.add_argument(
        "--gate-min-pass-rate", type=float, default=None, help="门禁：通过率下限"
    )
    eval_parser.add_argument(
        "--gate-max-regressions", type=int, default=None, help="门禁：允许的最大显著回归用例数"
    )
    eval_parser.add_argument(
        "--baseline", type=str, default=None, help="对比基线名称（用于门禁/回归检测）"
    )

    # load 命令
    load_parser = subparsers.add_parser("load", help="执行性能压测")
    load_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="配置文件路径"
    )
    load_parser.add_argument(
        "--suite", "-s", type=Path, required=True, help="测试用例集路径"
    )
    load_parser.add_argument(
        "--concurrency", "-j", type=int, default=None, help="并发数（覆盖配置）"
    )
    load_parser.add_argument(
        "--duration", "-d", type=int, default=None, help="压测时长（秒，覆盖配置）"
    )
    load_parser.add_argument(
        "--output", "-o", type=Path, default=None, help="报告输出目录"
    )

    # report 命令（离线转换）
    report_parser = subparsers.add_parser("report", help="从 JSON 报告生成 HTML")
    report_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="JSON 报告文件路径"
    )
    report_parser.add_argument(
        "--output", "-o", type=Path, default=None, help="输出目录"
    )

    # replay 命令（失败用例回放 + 三维 diff）
    replay_parser = subparsers.add_parser(
        "replay", help="重新执行单个用例并与旧 trace 对比"
    )
    replay_parser.add_argument(
        "--config", "-c", type=Path, required=True, help="配置文件路径"
    )
    replay_parser.add_argument(
        "--suite", "-s", type=Path, required=True, help="测试用例集路径"
    )
    replay_parser.add_argument(
        "--run-id", type=str, required=True, help="旧 trace 的 run id"
    )
    replay_parser.add_argument(
        "--case-id", type=str, required=True, help="要回放的用例 id"
    )
    replay_parser.add_argument(
        "--output", "-o", type=Path, default=None, help="输出目录"
    )

    # list 命令（只读查询）
    list_parser = subparsers.add_parser(
        "list", help="列出已保存的基线 / trace run（只读）"
    )
    list_parser.add_argument(
        "--output", "-o", type=Path, default=None, help="输出目录"
    )

    # synthesize 命令（基于种子合成评测用例）
    synth_parser = subparsers.add_parser(
        "synthesize", help="基于种子用例合成变体（同义改写/噪声/边界）"
    )
    synth_parser.add_argument(
        "--seeds", type=Path, required=True, help="种子用例集路径（YAML/JSON/CSV）"
    )
    synth_parser.add_argument(
        "--per-seed", type=int, default=None,
        help="每个种子生成的变体数（与 --count 互斥，推荐）",
    )
    synth_parser.add_argument(
        "--count", "-n", type=int, default=None,
        help="变体总量目标（在种子间均分；与 --per-seed 互斥）",
    )
    synth_parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGY_CHOICES,
        default=STRATEGY_CHOICES,
        help="参与的变异策略（默认全部；决定不传数量参数时的默认生成量）",
    )
    synth_parser.add_argument(
        "--config", "-c", type=Path, default=None, help="配置文件路径（复用其 judge.LLM 配置）"
    )
    synth_parser.add_argument(
        "--out", "-o", type=Path, required=True, help="输出 YAML 路径"
    )

    # prune 命令（清理旧 trace，默认 dry-run）
    prune_parser = subparsers.add_parser(
        "prune", help="清理旧 trace（保留最近 N 次，默认只扫描不删除）"
    )
    prune_parser.add_argument(
        "--output", "-o", type=Path, default=None, help="输出目录"
    )
    prune_parser.add_argument(
        "--keep", "-k", type=int, default=10, help="保留最近 N 个 run（默认 10）"
    )
    prune_parser.add_argument(
        "--yes", "-y", action="store_true", help="确认执行删除（否则仅 dry-run）"
    )

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "eval":
            return asyncio.run(_cmd_eval(args))
        elif args.command == "load":
            return asyncio.run(_cmd_load(args))
        elif args.command == "report":
            return _cmd_report(args)
        elif args.command == "replay":
            return asyncio.run(_cmd_replay(args))
        elif args.command == "list":
            return _cmd_list(args)
        elif args.command == "prune":
            return _cmd_prune(args)
        elif args.command == "synthesize":
            return asyncio.run(_cmd_synthesize(args))
    except Exception as exc:
        logger.exception("命令执行失败")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


async def _cmd_eval(args: argparse.Namespace) -> int:
    """执行功能评测。"""
    logger.info("加载配置: %s", args.config)
    config = load_config(args.config)

    # 提前确定输出目录并写回 config，使 judge cache / trace / baseline 等
    # 派生路径统一跟随 --output（而非 config.output_dir 默认值）。
    output_dir = args.output or config.output_dir
    config.output_dir = output_dir

    logger.info("加载用例集: %s", args.suite)
    suite = load_suite(args.suite)
    logger.info("共 %d 条用例", len(suite.cases))

    # 执行
    runner = BatchRunner(config)
    logger.info("开始执行...")
    run_result = await runner.run(suite.cases, concurrency=args.concurrency)

    # 评分
    logger.info("执行评分...")
    pipeline = EvalPipeline(config)
    run_result = await pipeline.evaluate(run_result)

    # 生成报告
    # 持久化 trace（供回放，不依赖基线）
    from agent_harness.core.trace_store import save_run_traces
    save_run_traces(run_result, output_dir / "traces")

    reporter = ReportGenerator(output_dir)
    paths = reporter.generate(run_result, formats=args.format)

    logger.info("评测完成:")
    for p in paths:
        logger.info("  %s", p)

    summary = run_result.summary
    print(f"\n{'='*50}")
    print(f"评测结果: {summary.get('eval_passed', 0)}/{summary.get('total', 0)} 通过")
    print(f"通过率: {summary.get('eval_pass_rate', 0):.1%}")
    print(f"平均分: {summary.get('eval_avg_score', 0):.2f}")
    print(f"{'='*50}\n")

    # ── 基线对比（在读旧基线之后、保存新基线之前）──
    # 用于门禁或 --baseline 显式对比。此时基线文件仍是上一次运行的结果，
    # 避免了「自己跟自己比恒无回归」的问题。
    baseline_store = BaselineStore(output_dir / "baselines")
    comparison = None
    if args.baseline:
        comparison = baseline_store.compare(run_result, args.baseline)
        if "error" in comparison:
            logger.warning("基线对比失败: %s", comparison["error"])
            comparison = None

    # ── CI 质量门禁 ──
    if args.gate:
        from agent_harness.core.gate import (
            GateRule,
            evaluate_gate,
            save_gate_result,
        )

        if args.gate_max_regressions is not None and comparison is None:
            # 未显式指定 --baseline 但要比较回归数 → 用默认基线
            comparison = baseline_store.compare(run_result, config.baseline_name)
            if "error" in comparison:
                logger.warning("基线对比失败: %s", comparison["error"])
                comparison = None

        rule = GateRule(
            min_score=args.gate_min_score,
            min_pass_rate=args.gate_min_pass_rate,
            max_regressions=args.gate_max_regressions,
        )
        gate_result = evaluate_gate(summary, comparison, rule)
        gate_path = save_gate_result(gate_result, output_dir)

        print(f"门禁结果: {'通过' if gate_result.passed else '未通过'}")
        for c in gate_result.checks:
            mark = "✓" if c.passed else "✗"
            print(f"  {mark} {c.name}: {c.detail}")
        print(f"门禁明细: {gate_path}")
        print()

        if not gate_result.passed:
            logger.error("质量门禁未通过，退出（本次结果已保存 trace，但未刷新基线）")
            return 1

    # ── 门禁通过（或未启用门禁）后刷新基线 ──
    baseline_store.save(config.baseline_name, run_result)
    logger.info("基线已保存: %s", baseline_store.baseline_dir / f"{config.baseline_name}.json")

    return 0


async def _cmd_load(args: argparse.Namespace) -> int:
    """执行性能压测。"""
    logger.info("加载配置: %s", args.config)
    config = load_config(args.config)

    logger.info("加载用例集: %s", args.suite)
    suite = load_suite(args.suite)

    engine = LoadEngine(config)
    logger.info(
        "开始压测: concurrency=%d, duration=%ds",
        args.concurrency or config.load.concurrency,
        args.duration or config.load.duration_seconds,
    )
    result = await engine.run(
        suite.cases,
        concurrency=args.concurrency,
        duration_seconds=args.duration,
    )

    output_dir = args.output or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存压测结果
    import json
    path = output_dir / f"load_report_{result.run_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    logger.info("压测完成: %s", path)

    print(f"\n{'='*50}")
    print(f"压测结果: {result.successful_requests}/{result.total_requests} 成功")
    print(f"吞吐量: {result.throughput_rps} RPS")
    print(f"错误率: {result.error_rate:.1%}")
    print(f"P50 延迟: {result.latency_percentiles.get('p50', 0)}ms")
    print(f"P99 延迟: {result.latency_percentiles.get('p99', 0)}ms")
    print(f"{'='*50}\n")

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """从 JSON 生成 HTML 报告。"""
    import json

    logger.info("读取报告: %s", args.input)
    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    from agent_harness.models import RunResult
    run_result = RunResult(**data)

    output_dir = args.output or args.input.parent
    reporter = ReportGenerator(output_dir)
    paths = reporter.generate(run_result, formats=["html"])

    for p in paths:
        logger.info("生成: %s", p)

    return 0


async def _cmd_replay(args: argparse.Namespace) -> int:
    """回放单个失败用例，与旧 trace 做三维 diff。

    流程：
    1. 从 traces/<run_id>/<case_id>.json 加载旧 trace
    2. 定位 suite 中对应用例，重新执行一次（走完整评分管线）
    3. 构建新 trace 并跑三维 diff（回答/工具/耗时）
    4. 输出终端彩色 diff + HTML 报告
    """
    from agent_harness.core.trace_store import (
        build_trace_diff,
        load_trace,
        save_run_traces,
    )

    output_dir = args.output or Path("harness_output")
    traces_dir = output_dir / "traces"

    # 1. 加载旧 trace
    old_trace = load_trace(traces_dir, args.run_id, args.case_id)
    if old_trace is None:
        print(f"Error: 未找到 trace（run={args.run_id}, case={args.case_id}）", file=sys.stderr)
        print(f"       目录: {traces_dir}", file=sys.stderr)
        return 1

    # 2. 定位用例
    config = load_config(args.config)
    suite = load_suite(args.suite)
    case = next((c for c in suite.cases if c.id == args.case_id), None)
    if case is None:
        print(f"Error: 用例 {args.case_id} 不在用例集 {args.suite} 中", file=sys.stderr)
        return 1

    logger.info("回放用例 %s（参考 run=%s）", args.case_id, args.run_id)

    # 3. 重新执行 + 评分
    runner = BatchRunner(config)
    replay_result = await runner.run([case], concurrency=1)
    pipeline = EvalPipeline(config)
    replay_result = await pipeline.evaluate(replay_result)

    # 4. 持久化新 trace 并构建 diff
    save_run_traces(replay_result, traces_dir)
    new_trace = load_trace(traces_dir, replay_result.run_id, args.case_id)
    if new_trace is None:
        print("Error: 新 trace 持久化失败", file=sys.stderr)
        return 1

    diff = build_trace_diff(old_trace, new_trace)

    # 5. 终端彩色 diff
    _print_terminal_diff(diff, args.run_id, replay_result.run_id, args.case_id)

    # 6. HTML 报告
    reporter = ReportGenerator(output_dir)
    html_path = reporter.generate_diff_html(
        replay_result.run_id, args.case_id, diff,
        args.run_id, replay_result.run_id,
    )
    print(f"\nDiff 报告: {html_path}")

    return 0


def _print_terminal_diff(
    diff: dict,
    old_run_id: str,
    new_run_id: str,
    case_id: str,
) -> None:
    """终端彩色打印三维 diff。"""
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    print(f"\n{'='*60}")
    print(f"Replay Diff: {case_id}  ({old_run_id} → {new_run_id})")
    print(f"{'='*60}")

    sc = diff["score_change"]
    print(f"\n[评分] overall {sc.get('old_overall')} → {sc.get('new_overall')}"
          f" | passed {sc.get('old_passed')} → {sc.get('new_passed')}")

    print("\n[回答文本]")
    if diff["answer_diff"]:
        for line in diff["answer_diff"].splitlines():
            if line.startswith("+"):
                print(f"  {GREEN}{line}{RESET}")
            elif line.startswith("-"):
                print(f"  {RED}{line}{RESET}")
            elif line.startswith("@@"):
                print(f"  {BLUE}{line}{RESET}")
            else:
                print(f"  {line}")
    else:
        print("  （无变化）")

    print("\n[工具调用序列]")
    if diff["tool_opcodes"]:
        for op in diff["tool_opcodes"]:
            tag, old, new = op["tag"], op["old"], op["new"]
            if tag == "equal":
                for n in old:
                    print(f"  {n}")
            elif tag in ("delete",):
                for n in old:
                    print(f"  {RED}- {n}{RESET}")
            elif tag in ("insert",):
                for n in new:
                    print(f"  {GREEN}+ {n}{RESET}")
            elif tag in ("replace",):
                for n in old:
                    print(f"  {RED}- {n}{RESET}")
                for n in new:
                    print(f"  {GREEN}+ {n}{RESET}")
    else:
        print("  （无变化）")

    if diff["tool_arg_diffs"]:
        print("\n[工具参数差异]")
        for ad in diff["tool_arg_diffs"]:
            print(f"  {ad['tool']}:")
            print(f"    {RED}- {ad['old_arguments']}{RESET}")
            print(f"    {GREEN}+ {ad['new_arguments']}{RESET}")

    print("\n[耗时对比 (ms)]")
    for item in diff["latency_diff"]:
        old_ms = f"{item['old_ms']:.2f}" if item["old_ms"] is not None else "-"
        new_ms = f"{item['new_ms']:.2f}" if item["new_ms"] is not None else "-"
        marker = ""
        if item["old_ms"] is not None and item["new_ms"] is not None:
            if item["new_ms"] > item["old_ms"] * 1.2:
                marker = f" {RED}↑{RESET}"
            elif item["new_ms"] < item["old_ms"] * 0.8:
                marker = f" {GREEN}↓{RESET}"
        print(f"  {item['stage']:<20} {old_ms:>10} → {new_ms:>10}{marker}")


def _cmd_list(args: argparse.Namespace) -> int:
    """列出已保存的基线名称与 trace run（只读）。"""
    output_dir = args.output or Path("harness_output")

    # 基线
    from agent_harness.core.baseline import BaselineStore
    baselines = BaselineStore(output_dir / "baselines").list_baselines()

    # trace runs
    from agent_harness.core.trace_store import list_run_ids
    run_ids = list_run_ids(output_dir / "traces")

    print(f"\n{'='*50}")
    print("Baselines（基线）:")
    if baselines:
        for b in baselines:
            print(f"  - {b}")
    else:
        print("  （无）")

    print(f"\nTraces（回放记录, {len(run_ids)} 个 run）:")
    if run_ids:
        for r in run_ids:
            print(f"  - {r}")
    else:
        print("  （无）")
    print(f"{'='*50}\n")

    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    """清理旧 trace，保留最近 N 个 run。

    默认 dry-run：只列出将被删除的目录，不实际删除；
    加 --yes 才真正执行。仅动 traces 目录，不碰 baselines。
    """
    import json as _json

    from agent_harness.core.trace_store import list_run_ids

    output_dir = args.output or Path("harness_output")
    traces_dir = output_dir / "traces"
    run_ids = list_run_ids(traces_dir)

    if len(run_ids) <= args.keep:
        print(f"\n共 {len(run_ids)} 个 run，未超过保留上限 {args.keep}，无需清理。\n")
        return 0

    # 按 _run.json 的 saved_at 排序，保留最新 N 个
    def _saved_at(run_id: str) -> str:
        meta_path = traces_dir / run_id / "_run.json"
        if not meta_path.exists():
            return ""
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = _json.load(f)
            return str(meta.get("saved_at") or meta.get("started_at") or "")
        except (OSError, _json.JSONDecodeError):
            return ""

    ordered = sorted(run_ids, key=_saved_at, reverse=True)
    keep = ordered[: args.keep]
    to_delete = ordered[args.keep :]

    print(f"\n保留最近 {args.keep} 个 run:")
    for r in keep:
        print(f"  保留  {r}")
    print(f"\n将删除 {len(to_delete)} 个旧 run:")
    for r in to_delete:
        print(f"  删除  {r}")

    if not args.yes:
        print("\n[dry-run] 未实际删除。加 --yes 确认执行。\n")
        return 0

    import shutil
    for r in to_delete:
        target = traces_dir / r
        if target.is_dir():
            shutil.rmtree(target)
            print(f"已删除 {r}")

    print(f"\n完成，剩余 {args.keep} 个 run。\n")
    return 0


async def _cmd_synthesize(args: argparse.Namespace) -> int:
    """基于种子用例合成变体评测集（promptfoo 思路）。"""
    logger.info("加载种子用例集: %s", args.seeds)
    suite = load_suite(args.seeds)
    seeds = suite.cases
    if not seeds:
        print("Error: 种子用例集为空", file=sys.stderr)
        return 1

    # 复用 --config 的 judge.LLM 配置；未给 config 则用默认（api_key 为空 → 走规则变体器）
    judge = load_config(args.config).eval.judge if args.config else load_config(None).eval.judge

    # --per-seed 与 --count 互斥
    if args.per_seed is not None and args.count is not None:
        print("Error: --per-seed 与 --count 互斥，请只指定其一", file=sys.stderr)
        return 1

    if args.per_seed is not None:
        target_desc = f"每种子 {args.per_seed} 条"
    elif args.count is not None:
        target_desc = f"总量约 {args.count} 条"
    else:
        target_desc = f"每种子 {len(args.strategies)} 条（默认）"

    logger.info(
        "开始合成: 种子 %d 条，策略 %s，目标 %s",
        len(seeds), "、".join(args.strategies), target_desc,
    )
    variants = await synthesize(
        seeds,
        judge,
        strategies=args.strategies,
        count=args.count,
        per_seed=args.per_seed,
    )

    text = build_suite_yaml(seeds, variants)
    write_suite_yaml(args.out, text)

    print(f"\n{'='*50}")
    print(f"合成完成: {len(variants)} 条变体（来自 {len(seeds)} 条种子，{target_desc}）")
    print(f"输出: {args.out}")
    print(f"\nNOTE: 请人工审核后合并入主用例集——"
          f"自动扩充只是起点，人工审核保证数据质量。")
    print(f"{'='*50}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""评测报告生成器。

支持 JSON / HTML 两种输出格式。
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_harness.models import RunResult


class ReportGenerator:
    """生成评测报告。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        run_result: RunResult,
        formats: list[str] | None = None,
    ) -> list[Path]:
        """生成报告文件，返回生成的文件路径列表。"""
        formats = formats or ["json", "html"]
        generated: list[Path] = []

        if "json" in formats:
            path = self._generate_json(run_result)
            generated.append(path)

        if "html" in formats:
            path = self._generate_html(run_result)
            generated.append(path)

        return generated

    def _generate_json(self, run_result: RunResult) -> Path:
        """生成 JSON 报告。"""
        path = self.output_dir / f"report_{run_result.run_id}.json"
        data = run_result.model_dump(mode="json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def generate_diff_html(
        self,
        run_id: str,
        case_id: str,
        diff: dict[str, Any],
        old_run_id: str,
        new_run_id: str,
    ) -> Path:
        """生成 replay 的三维 diff 报告 HTML。

        Args:
            run_id: 回放的新 run id
            case_id: 用例 id
            diff: build_trace_diff 的返回
            old_run_id: 旧 trace 的 run id
            new_run_id: 新 trace 的 run id
        """
        import html as _html

        path = self.output_dir / f"replay_{run_id}_{case_id}.html"

        def esc(s: Any) -> str:
            return _html.escape(str(s))

        # 回答 diff 行渲染（+/- 高亮）
        diff_html_lines: list[str] = []
        for line in diff["answer_diff"].splitlines():
            if line.startswith("+"):
                diff_html_lines.append(
                    f'<div class="add">+ {esc(line[1:])}</div>'
                )
            elif line.startswith("-"):
                diff_html_lines.append(
                    f'<div class="del">- {esc(line[1:])}</div>'
                )
            elif line.startswith("@@"):
                diff_html_lines.append(
                    f'<div class="meta">{esc(line)}</div>'
                )
            else:
                diff_html_lines.append(f'<div class="ctx">{esc(line)}</div>')

        # 工具调用 diff
        tool_rows = []
        for op in diff["tool_opcodes"]:
            tag, old, new = op["tag"], op["old"], op["new"]
            if tag == "equal":
                for n in old:
                    tool_rows.append(f'<div class="ctx">  {esc(n)}</div>')
            elif tag == "delete":
                for n in old:
                    tool_rows.append(f'<div class="del">- {esc(n)}</div>')
            elif tag == "insert":
                for n in new:
                    tool_rows.append(f'<div class="add">+ {esc(n)}</div>')
            elif tag == "replace":
                for n in old:
                    tool_rows.append(f'<div class="del">- {esc(n)}</div>')
                for n in new:
                    tool_rows.append(f'<div class="add">+ {esc(n)}</div>')

        arg_diff_html = ""
        if diff["tool_arg_diffs"]:
            arg_blocks = []
            for ad in diff["tool_arg_diffs"]:
                arg_blocks.append(
                    f'<div class="argdiff"><b>{esc(ad["tool"])}</b>'
                    f'<span class="del">- {esc(ad["old_arguments"])}</span>'
                    f'<span class="add">+ {esc(ad["new_arguments"])}</span></div>'
                )
            arg_diff_html = "".join(arg_blocks)

        # 耗时对比表
        latency_rows = "".join(
            f"<tr><td>{esc(item['stage'])}</td>"
            f"<td>{esc('%.2f' % item['old_ms'] if item['old_ms'] is not None else '-')}</td>"
            f"<td>{esc('%.2f' % item['new_ms'] if item['new_ms'] is not None else '-')}</td></tr>"
            for item in diff["latency_diff"]
        )

        sc = diff["score_change"]
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Replay Diff - {case_id}</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }}
.header {{ background: #1a1a2e; color: white; padding: 20px 30px; border-radius: 8px; margin-bottom: 20px; }}
.header h1 {{ margin: 0; font-size: 22px; }}
.header p {{ margin: 5px 0 0; opacity: 0.8; }}
h2 {{ color: #1a1a2e; border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 30px; }}
.box {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: auto; }}
.diff {{ font-family: 'SF Mono', Consolas, monospace; font-size: 13px; line-height: 1.5; }}
.add {{ background: #e6ffec; color: #116329; }}
.del {{ background: #ffebe9; color: #82071e; }}
.ctx {{ color: #666; }}
.meta {{ color: #0969da; }}
.argdiff {{ margin: 6px 0; padding: 6px; background: #fafafa; border-radius: 4px; }}
.argdiff .add, .argdiff .del {{ display: block; padding: 2px 6px; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ background: #1a1a2e; color: white; padding: 8px; text-align: left; font-size: 12px; }}
td {{ padding: 8px; border-bottom: 1px solid #eee; font-size: 13px; }}
.footer {{ margin-top: 20px; text-align: center; color: #666; font-size: 12px; }}
</style>
</head>
<body>
<div class="header">
    <h1>Replay Diff — {esc(case_id)}</h1>
    <p>old: {esc(old_run_id)} → new: {esc(new_run_id)}</p>
</div>

<h2>评分变化</h2>
<div class="box">
    overall: {esc(sc.get('old_overall'))} → {esc(sc.get('new_overall'))}
    | passed: {esc(sc.get('old_passed'))} → {esc(sc.get('new_passed'))}
</div>

<h2>回答文本 diff</h2>
<div class="box diff">
    {''.join(diff_html_lines) or '<div class="ctx">（无变化）</div>'}
</div>

<h2>工具调用序列 diff</h2>
<div class="box diff">
    {''.join(tool_rows) or '<div class="ctx">（无变化）</div>'}
</div>
{('<h2>工具参数差异</h2><div class="box">' + arg_diff_html + '</div>') if arg_diff_html else ''}

<h2>耗时对比（ms）</h2>
<div class="box">
    <table>
        <thead><tr><th>Stage</th><th>old</th><th>new</th></tr></thead>
        <tbody>{latency_rows}</tbody>
    </table>
</div>

<div class="footer">Generated by Agent Harness replay at {datetime.now().isoformat()}</div>
</body>
</html>"""

        with path.open("w", encoding="utf-8") as f:
            f.write(html)
        return path

    def _generate_html(self, run_result: RunResult) -> Path:
        """生成 HTML 报告。"""
        path = self.output_dir / f"report_{run_result.run_id}.html"

        summary = run_result.summary
        cases = run_result.case_results

        has_repeats = any(cr.repeat_count > 1 for cr in cases)

        rows = []
        for cr in cases:
            case = cr.case
            eval_score = cr.eval_score
            status = "PASS" if cr.success and (eval_score and eval_score.overall_passed) else "FAIL"
            status_class = "pass" if status == "PASS" else "fail"

            dims = ""
            if eval_score:
                for d in eval_score.dimensions:
                    dims += f'<span class="dim">{d.scorer.value}: {d.score:.2f}</span> '

            latency_str = ""
            if cr.latency.request_total_ms:
                latency_str = f"{cr.latency.request_total_ms:.0f}ms"

            extra = ""
            if has_repeats and cr.flaky:
                extra = '<span class="badge flaky">flaky</span> '
            if has_repeats and cr.repeat_count > 1:
                extra += f'<span class="badge repeat">μ={cr.mean_score:.2f} σ={cr.std_score:.2f}</span> '

            rows.append(f"""
            <tr class="{status_class}">
                <td>{_html.escape(case.id)}</td>
                <td>{_html.escape(case.question[:60])}</td>
                <td>{status}</td>
                <td>{extra}{dims}</td>
                <td>{latency_str}</td>
                <td>{_html.escape(cr.error[:50]) if cr.error else ""}</td>
            </tr>
            """)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Agent Harness Report - {_html.escape(run_result.run_id)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .header {{ background: #1a1a2e; color: white; padding: 20px 30px; border-radius: 8px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 5px 0 0; opacity: 0.8; }}
        .summary {{ display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex: 1; min-width: 150px; }}
        .card h3 {{ margin: 0 0 10px; font-size: 14px; color: #666; text-transform: uppercase; }}
        .card .value {{ font-size: 32px; font-weight: bold; color: #1a1a2e; }}
        .card .value.pass {{ color: #22c55e; }}
        .card .value.fail {{ color: #ef4444; }}
        .card .value.sub {{ font-size: 14px; color: #666; font-weight: normal; }}
        table {{ width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-collapse: collapse; }}
        th {{ background: #1a1a2e; color: white; padding: 12px; text-align: left; font-size: 12px; text-transform: uppercase; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; font-size: 13px; }}
        tr.pass {{ background: #f0fdf4; }}
        tr.fail {{ background: #fef2f2; }}
        .dim {{ display: inline-block; background: #e0e7ff; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 4px; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 4px; font-weight: bold; }}
        .badge.flaky {{ background: #fef3c7; color: #92400e; }}
        .badge.repeat {{ background: #dbeafe; color: #1e40af; }}
        .footer {{ margin-top: 20px; text-align: center; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Agent Harness Report</h1>
        <p>Run ID: {_html.escape(run_result.run_id)} | Mode: {_html.escape(run_result.mode)} | {_html.escape(run_result.started_at)}</p>
    </div>

    <div class="summary">
        <div class="card">
            <h3>Total Cases</h3>
            <div class="value">{summary.get('total', len(cases))}</div>
        </div>
        <div class="card">
            <h3>Passed</h3>
            <div class="value pass">{summary.get('eval_passed', summary.get('succeeded', 0))}</div>
        </div>
        <div class="card">
            <h3>Failed</h3>
            <div class="value fail">{summary.get('eval_failed', summary.get('failed', 0))}</div>
        </div>
        <div class="card">
            <h3>Pass Rate</h3>
            <div class="value">{summary.get('eval_pass_rate', summary.get('success_rate', 0)):.1%}</div>
            {"<div class='value sub'>Wilson CI: [{:.1%}–{:.1%}]</div>".format(*summary.get('eval_pass_rate_ci', [0, 0])) if 'eval_pass_rate_ci' in summary else ""}
        </div>
        <div class="card">
            <h3>Avg Score</h3>
            <div class="value">{summary.get('eval_avg_score', 0):.2f}</div>
        </div>
        {"<div class='card'><h3>Flaky Cases</h3><div class='value" + (" fail'>{}</div></div>".format(summary.get('flaky_count', 0)) if summary.get('flaky_count', 0) > 0 else " pass'>0</div></div>") if 'flaky_count' in summary else ""}
    </div>

    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Question</th>
                <th>Status</th>
                <th>Scores</th>
                <th>Latency</th>
                <th>Error</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>

    <div class="footer">
        Generated by Agent Harness at {datetime.now().isoformat()}
    </div>
</body>
</html>"""

        with path.open("w", encoding="utf-8") as f:
            f.write(html)
        return path

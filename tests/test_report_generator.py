from agent_harness.models import CaseResult, RunResult
from agent_harness.models import TestCase as HarnessTestCase
from agent_harness.report.generator import ReportGenerator


def test_html_report_escapes_untrusted_case_content(tmp_path):
    case = HarnessTestCase(id='<case id="x">', question="<script>alert(1)</script>")
    result = CaseResult(case=case, success=False, error="<img src=x onerror=alert(1)>")
    run = RunResult(run_id="run1", mode="eval", case_results=[result])

    path = ReportGenerator(tmp_path).generate(run, formats=["html"])[0]
    rendered = path.read_text(encoding="utf-8")

    assert "<script>" not in rendered
    assert "<img src=x" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img src=x" in rendered

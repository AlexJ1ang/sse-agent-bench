"""统计工具单元测试：Wilson 区间、McNemar、paired t-test。"""

from __future__ import annotations

from agent_harness.core.stats import (
    mcnemar_p_value,
    paired_t_test,
    wilson_interval,
)


def test_wilson_interval_empty():
    center, lower, upper = wilson_interval(0, 0)
    assert (center, lower, upper) == (0.0, 0.0, 1.0)


def test_wilson_interval_bounds():
    center, lower, upper = wilson_interval(50, 100)
    assert 0.0 <= lower <= center <= upper <= 1.0
    assert abs(center - 0.5) < 0.02


def test_wilson_interval_perfect():
    center, lower, upper = wilson_interval(10, 10)
    assert center > 0.8
    assert upper == 1.0


def test_mcnemar_no_change():
    assert mcnemar_p_value(0, 0) == 1.0


def test_mcnemar_strong_change():
    # 10 个回归、0 个改进 → 明显显著
    p = mcnemar_p_value(10, 0)
    assert p < 0.01


def test_paired_t_test_insufficient_sample():
    t, p = paired_t_test([1.0], [1.0])
    assert (t, p) == (0.0, 1.0)


def test_paired_t_test_identical():
    t, p = paired_t_test([0.8, 0.9, 0.85], [0.8, 0.9, 0.85])
    assert p == 1.0
    assert t == 0.0


def test_paired_t_test_clear_improvement():
    t, p = paired_t_test(
        [0.5, 0.6, 0.55, 0.6], [0.9, 0.85, 0.9, 0.88]
    )
    assert t > 0
    assert p < 0.05

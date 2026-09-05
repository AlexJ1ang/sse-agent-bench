"""统计工具。

为 LLM Agent 评测提供统计严谨性支撑：
- Wilson score interval：通过率置信区间
- McNemar 检验：配对 pass/fail 显著性检验
- paired t-test：配对连续分数显著性检验

仅依赖 Python 标准库，不引入 scipy。
"""

from __future__ import annotations

import math
from statistics import mean, stdev

# 标准正态分布 z 值（95% 置信度）
Z_95 = 1.96


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = Z_95,
) -> tuple[float, float, float]:
    """Wilson score interval。

    对二项分布通过率给出比正态近似更稳健的置信区间，
    小样本下表现尤其好。

    Args:
        successes: 通过次数
        total: 总次数
        confidence: z 值，默认 1.96（95%）

    Returns:
        (center, lower, upper)：中心估计、区间下界、区间上界
    """
    if total == 0:
        return 0.0, 0.0, 1.0

    p = successes / total
    z = confidence
    z2 = z * z

    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)
        / denominator
    )

    return center, max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_p_value(b: int, c: int) -> float:
    """McNemar 检验（精确二项式版本）。

    用于判断两次评测中 pass/fail 结果的变化是否统计显著。
    b = baseline 通过但 current 失败的用例数
    c = baseline 失败但 current 通过的用例数

    当 b + c < 25 时用精确二项式检验；否则用卡方近似。
    返回 p 值（双侧）。
    """
    n = b + c
    if n == 0:
        return 1.0

    if n < 25:
        # 精确二项式检验：P(X <= min(b,c)) * 2，X ~ Binomial(n, 0.5)
        k = min(b, c)
        prob = 0.0
        for i in range(k + 1):
            prob += math.comb(n, i) * (0.5 ** n)
        return min(1.0, prob * 2)

    # 卡方近似（带连续性校正）
    chi2 = (abs(b - c) - 1) ** 2 / n
    # 卡方分布 df=1 的 p 值：p = erfc(sqrt(chi2/2))
    return math.erfc(math.sqrt(chi2 / 2))


def paired_t_test(
    baseline_scores: list[float],
    current_scores: list[float],
) -> tuple[float, float]:
    """配对 t 检验。

    对同一批用例的两次评测分数做配对检验。

    Args:
        baseline_scores: baseline 各用例得分
        current_scores: 当前各用例得分（与 baseline 一一对应）

    Returns:
        (t_statistic, p_value)：t 统计量与双侧 p 值
        样本量 < 2 或方差为 0 时返回 (0.0, 1.0)
    """
    n = len(baseline_scores)
    if n < 2 or n != len(current_scores):
        return 0.0, 1.0

    diffs = [c - b for b, c in zip(baseline_scores, current_scores)]
    mean_diff = mean(diffs)

    try:
        std_diff = stdev(diffs)
    except Exception:
        return 0.0, 1.0

    if std_diff == 0:
        # 所有差值相同：完全相关，无法检验
        if mean_diff == 0:
            return 0.0, 1.0
        return float("inf"), 0.0

    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # t 分布 p 值用近似计算（Abramowitz & Stegun 26.7.1）
    # 对 df >= 1 足够精确
    p_value = _t_dist_two_tailed_p(abs(t_stat), n - 1)
    return t_stat, p_value


def _t_dist_two_tailed_p(t: float, df: int) -> float:
    """t 分布双侧 p 值近似。

    使用 Abramowitz & Stegun 公式 26.7.1 的数值逼近。
    df >= 30 时退化为标准正态。
    """
    if df >= 30:
        # 正态近似
        return math.erfc(t / math.sqrt(2))

    x = df / (df + t * t)
    # 不完全贝塔函数 I_x(df/2, 1/2) 的连分数逼近
    a = df / 2
    b = 0.5
    ib = _incomplete_beta(x, a, b)
    return min(1.0, ib)


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """正则化不完全贝塔函数 I_x(a,b) 的连分数逼近。

    Numerical Recipes betacf 算法的 Python 实现。
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(x, a, b) / a
    return 1.0 - front * _beta_cf(1.0 - x, b, a) / b


def _beta_cf(x: float, a: float, b: float) -> float:
    """不完全贝塔函数的连分数部分（Lentz 算法）。"""
    max_iter = 200
    eps = 3e-14
    fpmin = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return h


def describe_change(p_value: float, alpha: float = 0.05) -> str:
    """将 p 值翻译为人类可读的结论标签。"""
    if p_value < alpha:
        return "significant"
    return "not_significant"

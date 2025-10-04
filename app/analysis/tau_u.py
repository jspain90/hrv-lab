"""
Tau-U Analysis for Single-Case Experimental Designs

Tau-U is a non-parametric effect size statistic for single-case research that:
1. Measures non-overlap between baseline and intervention phases
2. Controls for baseline trend
3. Provides effect size interpretation

References:
- Parker, R. I., Vannest, K. J., Davis, J. L., & Sauber, S. B. (2011).
  Combining nonoverlap and trend for single-case research: Tau-U.
  Behavior Therapy, 42(2), 284-299.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Tuple, Optional


def calculate_kendall_s(data: list[float]) -> float:
    """
    Calculate Kendall's S statistic for a single phase (trend analysis).

    S = sum of concordant pairs - sum of discordant pairs

    Args:
        data: Time series data for one phase

    Returns:
        Kendall's S statistic
    """
    if len(data) < 2:
        return 0.0

    n = len(data)
    s = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if data[j] > data[i]:
                s += 1  # Concordant pair (increasing trend)
            elif data[j] < data[i]:
                s -= 1  # Discordant pair (decreasing trend)
            # Ties contribute 0

    return float(s)


def calculate_phase_comparison_s(baseline: list[float], intervention: list[float]) -> float:
    """
    Calculate Kendall's S for comparing baseline vs intervention phases.

    For each baseline value, count how many intervention values are greater.

    Args:
        baseline: Baseline phase data
        intervention: Intervention phase data

    Returns:
        S statistic for phase comparison
    """
    if not baseline or not intervention:
        return 0.0

    s = 0
    for b_val in baseline:
        for i_val in intervention:
            if i_val > b_val:
                s += 1
            elif i_val < b_val:
                s -= 1
            # Ties contribute 0

    return float(s)


def calculate_tau_u(
    baseline: list[float],
    intervention: list[float],
    control_baseline_trend: bool = True
) -> Tuple[float, float, str]:
    """
    Calculate Tau-U effect size for single-case design.

    Tau-U = (S_AB - S_A) / (n_A × n_B)

    Where:
    - S_AB: Kendall's S for baseline vs intervention comparison
    - S_A: Kendall's S for baseline trend (if controlling for trend)
    - n_A: number of baseline observations
    - n_B: number of intervention observations

    Args:
        baseline: Baseline phase data points
        intervention: Intervention phase data points
        control_baseline_trend: Whether to correct for baseline trend

    Returns:
        Tuple of (tau_u, p_value, effect_size_label)
    """
    n_baseline = len(baseline)
    n_intervention = len(intervention)

    # Check for sufficient data
    if n_baseline < 3 or n_intervention < 3:
        return 0.0, 1.0, "insufficient_data"

    # Step 1: Calculate phase comparison S (non-overlap)
    s_ab = calculate_phase_comparison_s(baseline, intervention)

    # Step 2: Calculate baseline trend S (if controlling)
    s_a = 0.0
    if control_baseline_trend:
        s_a = calculate_kendall_s(baseline)

    # Step 3: Calculate Tau-U
    denominator = n_baseline * n_intervention
    tau_u = (s_ab - s_a) / denominator if denominator > 0 else 0.0

    # Calculate variance for p-value
    # Variance formula from Parker et al. (2011)
    n_a = n_baseline
    n_b = n_intervention

    # Simplified variance calculation for Tau-U
    var_s = (n_a * n_b * (n_a + n_b + 1)) / 3.0

    # Add variance component for baseline trend control
    if control_baseline_trend:
        var_s += (n_a * (n_a - 1) * (2 * n_a + 5)) / 18.0

    # Z-score calculation
    se_tau_u = np.sqrt(var_s) / denominator if denominator > 0 else 1.0
    z_score = tau_u / se_tau_u if se_tau_u > 0 else 0.0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Effect size interpretation based on Parker et al. (2011)
    abs_tau = abs(tau_u)
    if abs_tau < 0.20:
        effect_label = "negligible"
    elif abs_tau < 0.60:
        effect_label = "small_to_medium"
    elif abs_tau < 0.80:
        effect_label = "medium_to_large"
    else:
        effect_label = "large"

    return tau_u, p_value, effect_label


def interpret_tau_u(tau_u: float, p_value: float, effect_label: str) -> str:
    """
    Provide human-readable interpretation of Tau-U results.

    Args:
        tau_u: Tau-U effect size
        p_value: Statistical significance
        effect_label: Effect size category

    Returns:
        Human-readable interpretation string
    """
    direction = "positive" if tau_u > 0 else "negative" if tau_u < 0 else "no"
    significance = "significant" if p_value < 0.05 else "non-significant"

    effect_names = {
        "negligible": "negligible",
        "small_to_medium": "small to medium",
        "medium_to_large": "medium to large",
        "large": "large",
        "insufficient_data": "insufficient data for analysis"
    }

    effect_name = effect_names.get(effect_label, effect_label)

    if effect_label == "insufficient_data":
        return "Insufficient data for analysis (need at least 3 observations in each phase)"

    return f"{significance.capitalize()} {direction} effect ({effect_name})"

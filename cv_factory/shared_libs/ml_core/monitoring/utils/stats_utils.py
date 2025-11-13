# shared_libs/ml_core/monitoring/utils/stats_utils.py

import numpy as np
import logging
from scipy import stats
from typing import Union


logger = logging.getLogger(__name__)

def calculate_psi(expected_dist: Union[np.ndarray, list], actual_dist: Union[np.ndarray, list]) -> float:
    """
    Calculates the Population Stability Index (PSI).
    """
    expected = np.array(expected_dist)
    actual = np.array(actual_dist)
    
    # Avoid division by zero
    np.place(expected, expected == 0, 1e-10)
    np.place(actual, actual == 0, 1e-10)
    
    psi_score = np.sum((expected - actual) * np.log(expected / actual))
    return psi_score

def calculate_ks_test(expected_data: Union[np.ndarray, list], actual_data: Union[np.ndarray, list]) -> tuple:
    """
    Performs the Kolmogorov-Smirnov (KS) test to compare two distributions.
    """
    ks_test_result = stats.ks_2samp(expected_data, actual_data)
    return ks_test_result.statistic, ks_test_result.pvalue

def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculates the Kullback-Leibler (KL) Divergence D(P || Q).
    """
    # Xử lý các trường hợp log(0)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Chỉ tính khi cả p và q đều > 0 (điểm cần lưu ý về mặt toán học)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def calculate_jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculates the Jensen-Shannon Divergence (JSD) between two probability distributions.
    """
    m = 0.5 * (p + q)
    jsd_score = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
    return jsd_score
"""
Unit tests for the constraint checkers.
Run with:  pytest tests/ -v
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module4.constraints import (
    c2_release_bounds, c4_mass_conservation, num_active_constraints
)
from module4.config import Config


def test_c2_within_bounds():
    R = np.array([[100.0, 200.0], [50.0, 75.0]])
    R_max = np.array([500.0, 500.0])
    assert c2_release_bounds(R, R_max) == 0.0


def test_c2_exceeds_capacity():
    R = np.array([[600.0, 200.0]])     # 600 > 500 by 100
    R_max = np.array([500.0])
    assert abs(c2_release_bounds(R, R_max) - 100.0) < 1e-9


def test_c2_negative_release():
    R = np.array([[-30.0, 200.0]])     # negative release of 30
    R_max = np.array([500.0])
    assert abs(c2_release_bounds(R, R_max) - 30.0) < 1e-9


def test_c4_feasible():
    R = np.array([[100.0], [100.0]])
    S_current = np.array([1000.0, 1000.0])
    Q_expected = np.array([[0.0], [0.0]])
    # total release 200 <= total available 2000 -> no violation
    assert c4_mass_conservation(R, S_current, Q_expected) == 0.0


def test_c4_infeasible():
    R = np.array([[1500.0], [1500.0]])     # release 3000
    S_current = np.array([1000.0, 1000.0]) # available 2000
    Q_expected = np.array([[0.0], [0.0]])
    # violation = 3000 - 2000 = 1000
    assert abs(c4_mass_conservation(R, S_current, Q_expected) - 1000.0) < 1e-9


def test_active_constraint_count_default():
    cfg = Config()
    # default: C1,C2,C3,C4,C5 on, C6 off -> 5
    assert num_active_constraints(cfg) == 5


def test_active_constraint_count_with_mahaweli():
    cfg = Config()
    cfg.ENABLE_C6_MAHAWELI = True
    assert num_active_constraints(cfg) == 6

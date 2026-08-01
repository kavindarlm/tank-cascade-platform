"""
Unit tests for the four objective functions, including the f3 zero-demand fix.
Run with:  pytest tests/ -v
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module4.objectives import f1_shortage, f2_overflow, f3_equity, f4_water_loss


def test_f1_shortage():
    R = np.array([[4000.0, 5500.0], [3000.0, 1000.0]])
    D = np.array([[5000.0, 5000.0], [3000.0, 3000.0]])
    # shortfalls: 1000, 0, 0, 2000 -> 3000
    assert abs(f1_shortage(R, D) - 3000.0) < 1e-9


def test_f4_water_loss():
    R = np.array([[4000.0, 5500.0], [3000.0, 1000.0]])
    D = np.array([[5000.0, 5000.0], [3000.0, 3000.0]])
    # over-release: 0, 500, 0, 0 -> 500
    assert abs(f4_water_loss(R, D) - 500.0) < 1e-9


def test_f1_f4_zero_at_perfect_delivery():
    D = np.array([[5000.0, 3000.0], [2000.0, 4000.0]])
    R = D.copy()
    assert f1_shortage(R, D) == 0.0
    assert f4_water_loss(R, D) == 0.0


def test_f3_zero_when_equal_ratios():
    """All tanks at the same satisfaction ratio -> zero inequity."""
    D = np.array([[1000.0], [2000.0], [4000.0]])
    R = 0.5 * D                        # every tank at rho = 0.5
    assert abs(f3_equity(R, D)) < 1e-9


def test_f3_excludes_zero_demand_tanks():
    """
    A zero-demand tank must NOT contribute artificial inequity (Bug 6 fix).
    Two active tanks at equal ratios + one zero-demand tank -> still zero.
    """
    D = np.array([[1000.0], [1000.0], [0.0]])   # tank 2 needs nothing
    R = np.array([[800.0], [800.0], [0.0]])     # active tanks both at 0.8
    # Only active tanks counted; both at 0.8 -> zero inequity.
    assert abs(f3_equity(R, D)) < 1e-9


def test_f3_detects_inequity():
    """Unequal ratios among active tanks -> positive inequity."""
    D = np.array([[1000.0], [1000.0]])
    R = np.array([[1000.0], [500.0]])           # ratios 1.0 and 0.5, mean 0.75
    # |1.0-0.75| + |0.5-0.75| = 0.25 + 0.25 = 0.5
    assert abs(f3_equity(R, D) - 0.5) < 1e-9


def test_f2_overflow_from_spill():
    spill = np.array([[0.0, 150.0], [0.0, 0.0]])
    assert abs(f2_overflow(spill) - 150.0) < 1e-9

"""
Unit tests for the state transition engine and the Bug 1/Bug 3 fixes.
Run with:  pytest tests/ -v
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module4.state_transition import simulate_cascade


def test_mass_balance_no_transfer():
    """A single tank, no upstream transfer, hand-checked one step."""
    S_init = np.array([1000.0])
    R = np.array([[100.0]])            # release 100
    Q = np.array([[50.0]])            # inflow 50
    alpha = np.array([[0.0]])
    E = np.array([[10.0]])            # evaporation 10
    S_min = np.array([0.0])
    S_max = np.array([10000.0])

    res = simulate_cascade(S_init, R, Q, alpha, E, S_min, S_max)
    # S_next = 1000 + 50 + 0 - 100 - 10 = 940
    assert abs(res["S_clipped"][0, 1] - 940.0) < 1e-9
    assert res["spill"][0, 0] == 0.0
    assert res["deficit"][0, 0] == 0.0


def test_overflow_is_detected():
    """Storage pushed above capacity must produce non-zero spill (Bug 1 fix)."""
    S_init = np.array([9950.0])
    R = np.array([[0.0]])
    Q = np.array([[200.0]])
    alpha = np.array([[0.0]])
    E = np.array([[0.0]])
    S_min = np.array([0.0])
    S_max = np.array([10000.0])

    res = simulate_cascade(S_init, R, Q, alpha, E, S_min, S_max)
    # S_next = 9950 + 200 = 10150 -> spill of 150
    assert abs(res["spill"][0, 0] - 150.0) < 1e-9
    # clipped storage capped at capacity
    assert abs(res["S_clipped"][0, 1] - 10000.0) < 1e-9


def test_mass_conservation_on_overrelease():
    """
    Requesting more release than available must be capped at what exists,
    and the shortfall recorded (Bug 3 fix). No water invented.
    """
    S_init = np.array([100.0])
    R = np.array([[500.0]])           # ask to release 500 but only ~100 available
    Q = np.array([[0.0]])
    alpha = np.array([[0.0]])
    E = np.array([[0.0]])
    S_min = np.array([0.0])
    S_max = np.array([10000.0])

    res = simulate_cascade(S_init, R, Q, alpha, E, S_min, S_max)
    # available = 100 + 0 + 0 - 0 - 0 = 100, so effective release = 100
    assert abs(res["R_effective"][0, 0] - 100.0) < 1e-9
    assert abs(res["R_shortfall"][0, 0] - 400.0) < 1e-9
    # storage cannot go negative
    assert res["S_clipped"][0, 1] >= 0.0


def test_upstream_transfer():
    """Two tanks: tank 0 releases, fraction arrives at tank 1."""
    S_init = np.array([1000.0, 500.0])
    R = np.array([[200.0], [0.0]])    # tank 0 releases 200, tank 1 releases 0
    Q = np.array([[0.0], [0.0]])
    # alpha[1,0] = 0.75 -> 75% of tank 0's release reaches tank 1
    alpha = np.array([[0.0, 0.0],
                      [0.75, 0.0]])
    E = np.array([[0.0], [0.0]])
    S_min = np.array([0.0, 0.0])
    S_max = np.array([10000.0, 10000.0])

    res = simulate_cascade(S_init, R, Q, alpha, E, S_min, S_max)
    # tank 0: 1000 - 200 = 800
    assert abs(res["S_clipped"][0, 1] - 800.0) < 1e-9
    # tank 1: 500 + 0.75*200 - 0 = 650
    assert abs(res["S_clipped"][1, 1] - 650.0) < 1e-9

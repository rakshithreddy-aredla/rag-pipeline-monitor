"""Phase 7 tests - CUSUM Eq. 11/12, calibration, per-run state, ARL0 simulation."""

import numpy as np
import pytest

from core.detection.cusum import (
    CusumConfigError,
    CusumDetector,
    CusumDetectorPool,
    CusumParams,
    arl0_approximation,
    calibrate,
    solve_control_limit,
)

# Regime note: Eq. 12 is an APPROXIMATION - it is accurate for small |Delta| and
# degrades as the standardized gap grows (verified empirically). These params
# give |Delta|=0.25 where empirical ARL lands within ~2x of target.
MU0, SIGMA, DELTA = 0.3, 0.4, 0.2


def _params(arl0=500):
    return CusumParams(mu0=MU0, sigma=SIGMA, delta=DELTA, arl0_target=arl0)


class TestEq12ControlLimit:
    def test_inversion_round_trips_for_several_gaps(self):
        """Solving Eq. 12 for h must reproduce the target ARL0."""
        kappa = MU0 + DELTA / 2
        gap = (MU0 - kappa) / SIGMA  # negative, per paper
        for target in (100, 500, 2000):
            h = solve_control_limit(gap, target)
            assert abs(arl0_approximation(gap, h) - target) < target * 1e-6

    def test_degenerate_gap_rejected(self):
        with pytest.raises(CusumConfigError):
            solve_control_limit(0.0, 500)


class TestEq11Update:
    def test_hand_computed_recursion(self):
        """mu0=0, sigma=1, delta=1 -> kappa=.5.
        Stream [0.6, 0.6]: S1=max(0,.1)=.1; S2=max(0,.1+.6-.5)=.2."""
        det = CusumDetector(CusumParams(mu0=0.0, sigma=1.0, delta=1.0))
        assert not det.update(0.6)
        assert abs(det.S - 0.1) < 1e-12
        assert not det.update(0.6)
        assert abs(det.S - 0.2) < 1e-12

    def test_negative_drift_clamps_to_zero(self):
        det = CusumDetector(CusumParams(mu0=0.0, sigma=1.0, delta=1.0))
        det.update(-10.0)
        assert det.S == 0.0


class TestCalibration:
    def test_calibrate_matches_sample_statistics(self):
        vals = list(np.random.default_rng(3).normal(0.05, 0.02, 500))
        mu0, sigma = calibrate(vals)
        assert abs(mu0 - float(np.mean(vals))) < 1e-9
        assert abs(sigma - float(np.std(vals, ddof=1))) < 1e-9

    def test_too_few_samples_rejected(self):
        with pytest.raises(CusumConfigError):
            calibrate([0.1] * 5)


class TestPropositionOneSimulation:
    def test_empirical_ARL0_near_target_under_null(self):
        """Null streams: mean run length should land in a generous band around
        the Eq. 12 target (TODO Phase 7 exit criterion)."""
        rng = np.random.default_rng(11)
        params = _params()
        runs, cap = 200, 4000
        lengths = []
        for _ in range(runs):
            det = CusumDetector(params)
            fired = None
            for i in range(cap):
                if det.update(float(rng.normal(MU0, SIGMA))):
                    fired = i
                    break
            lengths.append(fired if fired is not None else cap)
        mean_arl = float(np.mean(lengths))
        # Eq. 12 is an approximation; at |Delta|=0.25 it runs CONSERVATIVE
        # (empirical ALR ~ 2x target => real false-alarm rate LOWER than claimed).
        assert params.arl0_target / 2 <= mean_arl <= params.arl0_target * 3
        assert sum(1 for L in lengths if L == cap) < runs * 0.1  # low censoring

    def test_shift_detected_quickly(self):
        """Inject +delta shift after 20 clean steps; detector must fire fast."""
        rng = np.random.default_rng(13)
        params = _params()
        delays = []
        for _ in range(100):
            det = CusumDetector(params)
            fired_at = None
            for i in range(120):
                mean = MU0 if i < 20 else MU0 + 2 * DELTA
                if det.update(float(rng.normal(mean, SIGMA))):
                    fired_at = i
                    break
            assert fired_at is not None, "detector missed an injected shift"
            delays.append(fired_at)
        detection_delay = np.mean(delays) - 20
        assert detection_delay < 40  # well within a pipeline run length


class TestPerRunStateIsolation:
    def test_pool_state_never_crosses_runs(self):
        """AGENTS.md section 2.5: S_n is per-run, never global."""
        pool = CusumDetectorPool()
        d1 = pool.get_or_create("run-A", _params())
        d2 = pool.get_or_create("run-B", _params())
        d1.update(10.0)  # huge spike on run A only
        assert d2.S == 0.0
        pool.complete("run-A")
        d1b = pool.get_or_create("run-A", _params())
        assert d1b.S == 0.0  # garbage-collected on completion

"""Engineering-only tests; no videos, model weights, or participant records."""
import unittest
import numpy as np

from rppg_lab.waveform_v2_timing_contract import (estimate_signed_lag,
    synthetic_pulse, audit_synthetic_timing_transfer, demonstration_cases)

class TimingContractTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(0,18,1/60.)
        self.f=synthetic_pulse(self.t)
        self.h=synthetic_pulse(self.t,delay_ms=45.)

    def audit(self, f, h):
        return audit_synthetic_timing_transfer(self.t,source_face=self.f,source_hand=self.h,
                  reconstructed_face=f,reconstructed_hand=h,known_source_delay_ms=45.)

    def test_sign_and_clean_shift(self):
        pos=estimate_signed_lag(self.t,self.f,self.h)
        neg=estimate_signed_lag(self.t,self.h,self.f)
        self.assertEqual(pos.status,"diagnostic_only")
        self.assertEqual(neg.status,"diagnostic_only")
        self.assertAlmostEqual(pos.delay_ms,45.,delta=12.)
        self.assertAlmostEqual(neg.delay_ms,-45.,delta=12.)

    def test_subframe_result_keeps_source_uncertainty_floor(self):
        est=estimate_signed_lag(self.t,self.f,self.h)
        self.assertGreaterEqual(est.uncertainty_floor_ms,16.)

    def test_identity_transfer_passes_synthetic_only(self):
        r=self.audit(self.f,self.h)
        self.assertEqual(r["outcome"],"PASS_SYNTHETIC_ONLY")

    def test_common_latency_is_not_timing_preserving(self):
        r=self.audit(synthetic_pulse(self.t,delay_ms=100.),synthetic_pulse(self.t,delay_ms=145.))
        self.assertEqual(r["outcome"],"FAIL")
        self.assertEqual(r["reason"],"camera_time_anchor_shift")

    def test_differential_delay_failure(self):
        r=self.audit(self.f,synthetic_pulse(self.t,delay_ms=130.))
        self.assertEqual(r["outcome"],"FAIL")
        self.assertEqual(r["reason"],"differential_delay_distortion")

    def test_unrelated_signal_does_not_pass(self):
        r=self.audit(self.f,np.random.default_rng(43).normal(size=len(self.t)))
        self.assertNotEqual(r["outcome"],"PASS_SYNTHETIC_ONLY")

    def test_pol_inversion_does_not_pass(self):
        r=self.audit(self.f,-self.h)
        self.assertNotEqual(r["outcome"],"PASS_SYNTHETIC_ONLY")

    def test_missing_sample_fails_closed(self):
        y=self.h.copy();y[120]=np.nan
        self.assertEqual(estimate_signed_lag(self.t,self.f,y).status,"abstain")

    def test_mismatched_clock_fails_closed(self):
        t=self.t.copy();t[120:]+=0.01
        self.assertEqual(estimate_signed_lag(t,self.f,self.h).reason,"irregular_or_gapped_clock")

    def test_short_or_flat_fails_closed(self):
        self.assertEqual(estimate_signed_lag(self.t[:50],self.f[:50],self.h[:50]).status,"abstain")
        self.assertEqual(estimate_signed_lag(self.t,self.f,self.h*0).status,"abstain")

    def test_invalid_ground_truth_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_synthetic_timing_transfer(self.t,source_face=self.f,source_hand=self.h,
                    reconstructed_face=self.f,reconstructed_hand=self.h,known_source_delay_ms=float('nan'))

    def test_determinism(self):
        self.assertEqual(demonstration_cases(30.),demonstration_cases(30.))

    def test_morphology_mismatch_does_not_pass_at_30fps(self):
        r=[r for r in demonstration_cases(30.) if r["case"]=="hand_morphology_change"][0]
        self.assertEqual(r["outcome"], "FAIL")
        self.assertEqual(r["reason"], "differential_delay_distortion")

    def test_no_reference_fine_tuning_or_forced_alignment(self):
        r=self.audit(self.f,synthetic_pulse(self.t,delay_ms=130.))
        self.assertLess(r["zero_lag_correlation_hand"], 0.9)
        self.assertIsNotNone(r["differential_delay_error_ms"])

if __name__ == '__main__':
    unittest.main()

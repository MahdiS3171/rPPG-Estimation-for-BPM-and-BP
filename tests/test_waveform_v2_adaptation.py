"""CPU synthetic checks only: no training, original checkpoint or participant data."""
from dataclasses import replace
import io
import unittest

import numpy as np
import torch

from rppg_lab.models import MultiROIPriorResidualWaveformNet
from rppg_lab.waveform_v2_adaptation import (CalibrationSpec, SessionContext,
    SubjectContextPriorAdapter, summarize_calibration)
from rppg_lab.waveform_v2_timing_contract import synthetic_pulse, audit_synthetic_timing_transfer


class SubjectContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)
        self.spec = CalibrationSpec(duration_s=2.)
        self.base = MultiROIPriorResidualWaveformNet(6, ("GREEN", "CHROM", "POS_WIN"),
            base_channels=4, num_blocks=0, dropout=0).eval()
        # Exercise compatibility with nonzero V1 gates/residuals, without training.
        with torch.no_grad():
            self.base.gate_head[-1].weight.normal_(0, .03)
            self.base.residual_head.weight.normal_(0, .002)
        self.adapter = SubjectContextPriorAdapter(self.base, calibration=self.spec,
                                                 enabled=True, sample_rate_hz=60.)
        t = torch.arange(150, dtype=torch.float64) / 30
        rgb = torch.tensor([120., 100., 80.])[None, None, None, :].expand(2, 3, 150, 3).clone()
        rgb += torch.sin(t).float()[None, None, :, None]
        self.rgb, self.times = rgb, t
        self.valid = torch.ones(2, 3, 150, dtype=torch.bool)
        self.context = self.summarize()
        self.x = torch.randn(2, 3, 6, 600)
        pulse = torch.tensor(synthetic_pulse(np.arange(600) / 60), dtype=torch.float32)
        self.x[:, :, 3:] = pulse[None, None, None, :] * torch.tensor([1., 1.2, .9])[None, None, :, None]
        self.starts = torch.tensor([2., 3.])

    def summarize(self, rgb=None, valid=None, times=None):
        return summarize_calibration(self.rgb if rgb is None else rgb,
            self.times if times is None else times, self.valid if valid is None else valid,
            roi_names=self.base.roi_names, calibration=self.spec)

    def forward(self, **kw):
        options = dict(context=self.context, target_start_s=self.starts, return_dict=True)
        options.update(kw)
        return self.adapter(self.x, **options)

    def assert_outputs_equal(self, one, two):
        self.assertEqual(set(one), set(two))
        for k in one:
            self.assertTrue(torch.equal(one[k], two[k]), k)

    def test_zero_initialization_is_exact_v1_with_nonzero_window_gate(self):
        self.assertEqual(self.adapter.context_head.weight.count_nonzero(), 0)
        self.assert_outputs_equal(self.base(self.x, return_dict=True), self.forward())

    def test_disabled_ignores_context_and_requires_no_target_clock(self):
        self.adapter.enabled = False
        with torch.no_grad():
            self.adapter.context_head.weight.fill_(100.)
        bad = replace(self.context, features=torch.full_like(self.context.features, torch.nan))
        self.assert_outputs_equal(self.base(self.x, return_dict=True),
                                  self.forward(context=bad, target_start_s=None))

    def test_missing_context_is_exact_v1_even_with_nonzero_head(self):
        with torch.no_grad():
            self.adapter.context_head.weight.fill_(100.)
        self.assert_outputs_equal(self.base(self.x, return_dict=True),
                                  self.forward(context=None, target_start_s=None))

    def test_zero_context_and_missing_rois_do_not_create_adjustments(self):
        with torch.no_grad():
            self.adapter.context_head.weight.fill_(100.)
        empty = replace(self.context, features=torch.zeros_like(self.context.features))
        self.assert_outputs_equal(self.base(self.x, return_dict=True), self.forward(context=empty))
        absent = self.summarize(valid=torch.zeros_like(self.valid))
        self.assertFalse(absent.usable.any())
        self.assertTrue(torch.isfinite(absent.features).all())
        self.assert_outputs_equal(self.base(self.x, return_dict=True), self.forward(context=absent))

    def test_context_uses_only_predeclared_past_segment(self):
        changed = self.rgb.clone()
        changed[:, :, self.times >= self.spec.end_s] = torch.nan
        self.assertTrue(torch.equal(self.context.features, self.summarize(rgb=changed).features))
        self.assertFalse(self.context.features.requires_grad)

    def test_context_is_snapshot_and_deterministic(self):
        second = self.summarize()
        self.rgb[:] = 0
        self.assertTrue(torch.equal(self.context.features, second.features))
        self.assertTrue(torch.equal(self.context.usable, second.usable))
        self.assert_outputs_equal(self.forward(), self.forward())

    def test_target_overlap_missing_clock_and_invalid_clocks_rejected(self):
        for start in (None, [1.99, 4.], [float('nan'), 4.], [2.], [float('inf'), 3.]):
            with self.subTest(start=start), self.assertRaises(ValueError):
                self.forward(target_start_s=start)
        self.forward(target_start_s=[2., 2.])  # half-open boundary is allowed

    def test_malformed_context_shape_order_interval_and_nonfinite_rejected(self):
        bad_contexts = [
            replace(self.context, features=self.context.features[:1]),
            replace(self.context, roi_names=self.base.roi_names[::-1]),
            replace(self.context, calibration=CalibrationSpec(duration_s=1.)),
            replace(self.context, usable=self.context.usable.float()),
            replace(self.context, features=self.context.features.long()),
        ]
        for value in (float('nan'), float('inf'), -1., 1.01):
            bad_contexts.append(replace(self.context, features=torch.full_like(self.context.features, value)))
        for context in bad_contexts:
            with self.subTest(context=context), self.assertRaises(ValueError):
                self.forward(context=context)

    def test_calibration_shape_support_and_gap_validation(self):
        for spec in (dict(duration_s=0), dict(start_s=-1), dict(duration_s=float('nan'))):
            with self.assertRaises(ValueError):
                CalibrationSpec(**spec)
        with self.assertRaises(ValueError):
            self.summarize(rgb=self.rgb[:, :, :2])
        with self.assertRaises(ValueError):
            self.summarize(valid=self.valid.float())
        gapped = self.times.clone()
        gapped[30:] += .2
        for index, times in enumerate((torch.zeros_like(self.times), self.times + 10, gapped)):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.summarize(times=times)

    def test_missing_and_extreme_optical_values_stay_finite(self):
        rgb = self.rgb.double()
        rgb[:, 0, :60] = torch.nan
        rgb[:, 1, :60] = 1e300
        rgb[:, 2, :10] = -1
        context = self.summarize(rgb=rgb)
        self.assertFalse(context.usable[:, 0].any())
        self.assertTrue(torch.isfinite(context.features).all())
        self.assertTrue((context.features >= 0).all() and (context.features <= 1).all())
        self.assertLess(context.features[0, 2, 6], 1)

    def test_bounded_offsets_and_nonfinite_parameter_rejection(self):
        with torch.no_grad():
            self.adapter.context_head.weight.fill_(1e30)
        offsets = self.adapter.context_offsets(self.context, 2)
        self.assertTrue(torch.isfinite(offsets).all())
        self.assertLessEqual(float(offsets.abs().max().detach()), .25)
        with torch.no_grad():
            self.adapter.context_head.weight.fill_(torch.nan)
        with self.assertRaises(ValueError):
            self.forward()

    def test_nonzero_offset_matches_bounded_logit_correction(self):
        with torch.no_grad():
            self.adapter.context_head.weight[0].fill_(.8)
            self.adapter.context_head.weight[1].fill_(-.4)
        original = self.base(self.x, return_dict=True)
        actual, audit = self.forward(return_audit=True)
        offsets = self.adapter.context_offsets(self.context, 2)
        expected = (original['prior_weights'].double().log() + offsets.double()).softmax(-1).float()
        self.assertEqual(audit['adapted_windows'], 2)
        torch.testing.assert_close(actual['prior_weights'], expected)
        for key in ('residuals', 'roi_attention', 'quality', 'features'):
            self.assertTrue(torch.equal(actual[key], original[key]), key)

    def test_active_masks_nan_channels_and_normalization(self):
        with torch.no_grad():
            self.adapter.context_head.weight[0].fill_(.4)
        roi = torch.tensor([[True, False, True], [False, True, True]])
        prior = torch.ones(2, 3, 3, dtype=torch.bool)
        prior[:, :, 1] = False
        self.x[:, :, 4] = torch.nan
        self.x[~roi] = torch.nan
        output = self.forward(roi_valid=roi, prior_valid=prior)
        self.assertTrue(torch.isfinite(output['ppg']).all())
        self.assertEqual(output['prior_weights'][~prior].count_nonzero(), 0)
        self.assertEqual(output['prior_weights'][~roi].count_nonzero(), 0)
        torch.testing.assert_close(output['prior_weights'].sum(-1), roi.float())
        torch.testing.assert_close(output['roi_attention'].sum(-1), torch.ones(2))
        self.assertEqual(output['roi_attention'][~roi].count_nonzero(), 0)
        self.assertEqual(output['roi_ppg'][~roi].count_nonzero(), 0)

    def test_invalid_roi_prior_combinations_remain_v1_errors(self):
        with self.assertRaises(ValueError):
            self.forward(roi_valid=torch.zeros(2, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.forward(prior_valid=torch.zeros(2, 3, 3, dtype=torch.bool))

    def test_underflowed_valid_weights_remain_finite_and_masked(self):
        with torch.no_grad():
            self.base.static_prior_logits.copy_(torch.tensor([1000., -1000., 0.]))
            self.adapter.context_head.weight[0].fill_(.8)
        out = self.forward()
        self.assertTrue(torch.isfinite(out['ppg']).all())
        torch.testing.assert_close(out['prior_weights'].sum(-1), torch.ones(2, 3))

    def test_checkpoint_roundtrip_and_original_state_unchanged(self):
        before = {k: v.clone() for k, v in self.base.state_dict().items()}
        buffer = io.BytesIO()
        torch.save(before, buffer)
        buffer.seek(0)
        copy = MultiROIPriorResidualWaveformNet(**self.base.model_config).eval()
        loaded = copy.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        wrapped = SubjectContextPriorAdapter(copy, calibration=self.spec, enabled=True, sample_rate_hz=60.)
        self.assert_outputs_equal(self.forward(), wrapped(self.x, return_dict=True,
            context=self.context, target_start_s=self.starts))
        self.assertEqual(set(self.adapter.state_dict()) - {'v1.' + k for k in before},
                         {'context_head.weight'})
        for key, value in self.base.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertFalse(self.base.gate_head._forward_hooks)

    def test_eval_only_and_constructor_validation(self):
        for kwargs in (dict(max_logit_offset=0), dict(max_logit_offset=1.1),
                       dict(sample_rate_hz=float('nan')), dict(sample_rate_hz=0)):
            with self.assertRaises(ValueError):
                SubjectContextPriorAdapter(self.base, **kwargs)
        self.adapter.train()
        with self.assertRaises(ValueError):
            self.forward()
        with self.assertRaises(ValueError):
            SubjectContextPriorAdapter(self.base)

    def test_coverage_includes_all_missing_context_windows(self):
        out, audit = self.forward(context=None, return_audit=True)
        self.assertEqual(audit['intended_windows'], 2)
        self.assertEqual(audit['adapted_windows'], 0)
        self.assertEqual(audit['fallback_or_identity_windows'], 2)
        self.assertEqual(audit['status_counts'], {'missing_context': 2})
        self.assertTrue(torch.equal(out['ppg'], self.adapter(self.x, context=None)))

    def test_identity_preserves_injected_signed_delay_and_camera_anchors(self):
        with torch.no_grad():
            self.base.residual_head.weight.zero_()
            self.base.residual_head.bias.zero_()
        t = np.arange(600) / 60
        face = synthetic_pulse(t)
        hand = synthetic_pulse(t, delay_ms=45.)
        self.x[0, :, 3:] = torch.tensor(face, dtype=torch.float32)
        self.x[1, :, 3:] = torch.tensor(hand, dtype=torch.float32)
        result = self.forward()['ppg'].numpy()
        report = audit_synthetic_timing_transfer(t, source_face=face, source_hand=hand,
            reconstructed_face=result[0], reconstructed_hand=result[1], known_source_delay_ms=45.)
        self.assertEqual(report['outcome'], 'PASS_SYNTHETIC_ONLY')

    def test_equal_hr_phase_changing_weights_fall_back_to_v1(self):
        base = MultiROIPriorResidualWaveformNet(5, ('GREEN', 'CHROM'), roi_names=('face',),
            base_channels=4, num_blocks=0, dropout=0).eval()
        with torch.no_grad():
            base.static_prior_logits.zero_()
        adapter = SubjectContextPriorAdapter(base, calibration=self.spec, enabled=True,
                                             max_logit_offset=1., sample_rate_hz=60.)
        with torch.no_grad():
            adapter.context_head.weight[0].fill_(-100)
            adapter.context_head.weight[1].fill_(100)
        t = np.arange(600) / 60
        early, late = synthetic_pulse(t), synthetic_pulse(t, delay_ms=130.)
        x = torch.zeros(1, 1, 5, 600)
        x[0, 0, 3] = torch.tensor(early)
        x[0, 0, 4] = torch.tensor(late)
        context = SessionContext(self.context.features[:1, :1], self.context.usable[:1, :1],
                                 ('face',), self.spec)
        result, audit = adapter(x, context=context, target_start_s=[2.], return_dict=True, return_audit=True)
        self.assert_outputs_equal(base(x, return_dict=True), result)
        self.assertEqual(audit['adapted_windows'], 0)
        self.assertEqual(audit['fallback_or_identity_windows'], 1)
        self.assertTrue(next(iter(audit['status_counts'])).startswith('abstain_'))

    def test_too_short_phase_control_falls_back_without_dropping_denominator(self):
        with torch.no_grad():
            self.adapter.context_head.weight[0].fill_(1.)
        self.x = self.x[..., :120]
        result, audit = self.forward(return_audit=True)
        self.assert_outputs_equal(self.base(self.x, return_dict=True), result)
        self.assertEqual(audit['status_counts'], {'abstain_too_short': 2})
        self.assertEqual(audit['intended_windows'], 2)


if __name__ == '__main__':
    unittest.main()

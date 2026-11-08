"""Rate, mask, context and gradient contracts of the new screen models."""
import unittest
import numpy as np
import torch
from rppg_lab.waveform_v2_screen_models import AdaptiveWaveformNet
from rppg_lab.waveform_v2_screen_losses import training_loss
from rppg_lab.waveform_v2_screen_features import window_descriptors


class WaveformScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def batch(self,fs=30):
        t=torch.arange(10*fs)/fs
        x=torch.stack([torch.sin(2*torch.pi*(1.1+.01*c)*t+.1*c) for c in range(27)]).reshape(1,3,9,-1)
        return (x,torch.ones(1,3,dtype=torch.bool),torch.ones(1,3,6,dtype=torch.bool),
                torch.zeros(1,75),torch.zeros(1,75),torch.zeros(1,dtype=torch.bool))
    def test_physical_kernel_span_at_30_and_60(self):
        for kind in ('gate','gate_residual','raw_temporal','dual_band'):
            torch.manual_seed(7);model=AdaptiveWaveformNet(kind).eval()
            with torch.inference_mode():
                low=model(*self.batch(30),fs=30)['ppg'];high=model(*self.batch(60),fs=60)['ppg']
            torch.testing.assert_close(high[:,::2],low,rtol=2e-4,atol=2e-5)
    def test_context_identity_at_initialization(self):
        torch.manual_seed(9);plain=AdaptiveWaveformNet('gate_residual')
        torch.manual_seed(9);context=AdaptiveWaveformNet('gate_residual',context=True)
        batch=list(self.batch());batch[4]=torch.randn(1,75);batch[5][:]=True
        torch.testing.assert_close(plain(*batch)['ppg'],context(*batch)['ppg'],rtol=0,atol=0)
    def test_all_losses_have_finite_nonzero_training_gradients(self):
        for kind in ('gate','gate_residual','raw_temporal','dual_band'):
            for loss in ('native','morphology','derivative_spectral','staged'):
                model=AdaptiveWaveformNet(kind);batch=self.batch();target=torch.cos(torch.arange(300)[None]*.2)
                value=training_loss(model(*batch),target,loss,90,120);value.backward()
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum()>0 for g in gradients))
    def test_missing_all_inputs_abstains_instead_of_generating_a_pulse(self):
        for kind in ('gate','gate_residual','raw_temporal','dual_band'):
            batch=list(self.batch());batch[0][:]=float('nan');batch[1][:]=False;batch[2][:]=False
            output=AdaptiveWaveformNet(kind)(*batch)
            self.assertFalse(bool(output['usable'][0]));self.assertEqual(float(output['ppg'].detach().abs().sum()),0)
    def test_flat_rgb_abstains_even_when_geometry_is_valid(self):
        for kind in ('gate','gate_residual','raw_temporal','dual_band'):
            batch=list(self.batch());batch[0][:,:,:3,:]=0
            output=AdaptiveWaveformNet(kind)(*batch)
            self.assertFalse(bool(output['usable'][0]))
            self.assertEqual(float(output['ppg'].detach().abs().sum()),0)
    def test_descriptors_are_finite_for_flat_and_inverted_inputs(self):
        for x in (np.zeros((3,9,300)),self.batch()[0][0].numpy(),-self.batch()[0][0].numpy()):
            result=window_descriptors(x)
            self.assertEqual(result.shape,(75,));self.assertTrue(np.isfinite(result).all())


if __name__=='__main__':unittest.main()

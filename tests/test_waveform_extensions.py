import unittest
import torch
from rppg_lab.waveform_v2_extended_models import make_extended_model
from rppg_lab.waveform_v2_screen_losses import training_loss


class ExtensionTests(unittest.TestCase):
    def test_native_rates_shapes_and_finite_gradients(self):
        torch.set_num_threads(1)
        for architecture in ('long_tcn','physical_attention','spectral','temporal_gate'):
            model=make_extended_model(architecture)
            for fs in (30,60):
                x=torch.randn(1,3,9,10*fs);r=torch.ones(1,3,dtype=torch.bool);p=torch.ones(1,3,6,dtype=torch.bool)
                q=torch.zeros(1,75);v=torch.zeros(1,dtype=torch.bool)
                out=model(x,r,p,q,q,v,fs)
                self.assertEqual(tuple(out['ppg'].shape),(1,10*fs));self.assertTrue(torch.isfinite(out['ppg']).all())
                value=training_loss(out,torch.randn(1,10*fs),'morphology',0,120,fs)
                value.backward();self.assertTrue(all(a.grad is None or torch.isfinite(a.grad).all() for a in model.parameters()))
                model.zero_grad(set_to_none=True)


if __name__=='__main__':unittest.main()

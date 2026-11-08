import unittest
import numpy as np
import torch
from rppg_lab.waveform_feature_specialists import pulse_features,ThreePartFeatureModel,matched_widths


class SpecialistTests(unittest.TestCase):
    def test_known_pulse_feature_units_and_amplitude_offset_invariance(self):
        values=[]
        for fs in (30,60):
            t=np.arange(10*fs)/fs;wave=np.sin(2*np.pi*t)
            a,valid=pulse_features(wave,fs);b,other=pulse_features(3*wave+17,fs)
            self.assertTrue(valid.all());np.testing.assert_array_equal(valid,other)
            np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-7)
            self.assertAlmostEqual(a[0],60,delta=.5);self.assertAlmostEqual(a[1],1,delta=.03)
            self.assertAlmostEqual(a[3],.5,delta=.04);self.assertAlmostEqual(a[4],.5,delta=.04)
            self.assertAlmostEqual(a[5],1/3,delta=.04);self.assertAlmostEqual(a[6],.5,delta=.04)
            self.assertAlmostEqual(a[7],np.pi,delta=.2);values.append(a)
        np.testing.assert_allclose(values[0][3:],values[1][3:],rtol=.08,atol=.04)
        self.assertFalse(pulse_features(np.ones(300))[1].any())

    def test_independent_experts_receive_no_other_tasks_gradient(self):
        torch.set_num_threads(1);model=ThreePartFeatureModel('independent')
        x=torch.randn(2,3,9,300);roi=torch.ones(2,3,dtype=torch.bool);prior=torch.ones(2,3,6,dtype=torch.bool)
        model(x,roi,prior,torch.zeros(2,75))['features'][:,:3].sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.experts[0].parameters() if p.grad is not None),0)
        for expert in model.experts[1:]:self.assertEqual(sum(float(p.grad.abs().sum()) for p in expert.parameters() if p.grad is not None),0)
        groups=[{id(p) for p in expert.parameters()} for expert in model.experts]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

    def test_parameter_matching_native_shapes_and_flat_abstention(self):
        torch.set_num_threads(1);widths=matched_widths();counts=[]
        for mode,width in widths.items():
            model=ThreePartFeatureModel(mode,width);counts.append(sum(p.numel() for p in model.parameters()))
            for fs in (30,60):
                roi=torch.ones(1,3,dtype=torch.bool);prior=torch.ones(1,3,6,dtype=torch.bool);q=torch.zeros(1,75)
                output=model(torch.randn(1,3,9,10*fs),roi,prior,q,fs)
                self.assertEqual(tuple(output['features'].shape),(1,10));self.assertTrue(torch.isfinite(output['features']).all())
                self.assertFalse(model(torch.zeros(1,3,9,10*fs),roi,prior,q,fs)['usable'].any())
        self.assertLess(max(counts)/min(counts),1.10)


if __name__=='__main__':unittest.main()

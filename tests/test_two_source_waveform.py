"""Scientific contracts of the small source-balanced development baseline."""
import unittest
import numpy as np

from scripts.study_two_source_waveform import fit_ridge


class SourceBalancedRidgeTests(unittest.TestCase):
    def test_fixed_signed_model_learns_inversion_without_per_window_oracle(self):
        y=np.sin(np.arange(300)*.13)
        rows=[dict(id='training_subject',x=(-y)[None],y=y)]
        weight=fit_ridge([rows],np.array([0]))
        self.assertLess(float(weight[0]),0)
        held=np.cos(np.arange(300)*.17)
        self.assertGreater(np.corrcoef(weight@(-held)[None],held)[0,1],.999999)

    def test_repeating_one_sources_windows_does_not_change_its_total_weight(self):
        t=np.arange(300)
        a=dict(id='a',x=np.sin(t*.1)[None],y=np.cos(t*.1))
        b=dict(id='b',x=np.cos(t*.13)[None],y=np.cos(t*.13))
        once=fit_ridge([[a],[b]],np.array([0]))
        repeated=fit_ridge([[a]*11,[b]],np.array([0]))
        np.testing.assert_allclose(once,repeated,atol=1e-12,rtol=1e-12)


if __name__=='__main__': unittest.main()

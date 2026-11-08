import unittest
import numpy as np
from rppg_lab.types import RegionObservation
from rppg_lab.roi import FACE_ROIS
from rppg_lab.region_masks import get_face_masks
from rppg_lab.pilot_frozen_extraction import measure_native_masks
from rppg_lab.pilot11_acquisition import face_pixels


class NativePixelsTests(unittest.TestCase):
    def test_bounded_face_masks_preserve_exact_native_pixels_including_clipped_edges(self):
        rng=np.random.default_rng(20261108);frame=rng.integers(0,256,(180,120,3),dtype=np.uint8)
        for offset in (0.,-40.,100.):
            points=rng.uniform([5,5],[115,175],(468,2))+offset
            observation=RegionObservation(0,0.,landmarks=points,valid=True)
            names=tuple(FACE_ROIS);full=measure_native_masks(frame,get_face_masks(frame,observation,names,False,(133,173),(77,127)),50)
            bounded=face_pixels(frame,observation,names,50)
            for name in names:
                np.testing.assert_array_equal(full[name][0],bounded[name][0])
                for key in ('mask_pixel_count','skin_pixel_count','mask_coverage','clipping_fraction','brightness','extraction_valid'):
                    np.testing.assert_allclose(full[name][1][key],bounded[name][1][key],equal_nan=True)


if __name__=='__main__':unittest.main()

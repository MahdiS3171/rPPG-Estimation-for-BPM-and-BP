"""Controlled optical truth/ambiguity and timing fixtures, never real skin labels."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from rppg_lab.waveform_v2_optical_quality import (OpticalQualityConfig,optical_measurement,temporal_descriptors,METRICS)
from rppg_lab.waveform_v2_micro_rois import (partition_polygon,evaluate_candidate,frozen_patch_mix,
    IndependentSkinEvidence,weight_change_diagnostics)
from rppg_lab.waveform_v2_quality_timing import timing_cases
from rppg_lab.waveform_v2_optical_audit import collect_optical_evidence
from rppg_lab.waveform_v2_scamps import ScampsRecording,ScampsLayout,source_sha256
from rppg_lab.waveform_v2_scamps_roi import extract_scamps_roi_traces,ScampsROIConfig,ROI_NAMES
from rppg_lab.roi import FACE_ROIS
from tests.test_waveform_v2_scamps_inputs import hdf_fixture,FakeDetector
from scripts.audit_waveform_v2_optical_quality import review_template,clean


def patches_from_image(frame,n=12):
    mask=np.ones(frame.shape[:2],dtype=bool)
    rgb=np.full((1,6,n,3),np.nan);valid=np.zeros((1,6,n),dtype=bool)
    pm={k:np.full((1,6,n),np.nan) for k in METRICS}
    rm={k:np.full((1,n),np.nan) for k in METRICS}
    _,_,a=optical_measurement(frame,mask,50)
    for k in METRICS:rm[k][:]=a[k]
    for p,cell in enumerate(partition_polygon(mask)):
        value,available,metrics=optical_measurement(frame,cell,16)
        rgb[0,p]=value;valid[0,p]=available
        for k in METRICS:pm[k][0,p]=metrics[k]
    return rgb,valid,pm,rm


def uniform_skin():
    return np.broadcast_to(np.array([180.,130.,110.]),(32,32,3)).copy()


def evidence(blocks=2,state='verified_skin'):
    return IndependentSkinEvidence(np.full((1,blocks),state,dtype='U40'),
        'controlled_synthetic_ground_truth','fixture-truth-not-a-human-annotation')


class WideFixtureDetector(FakeDetector):
    def detect(self,*args):
        observation=super().detect(*args)
        if observation.valid:
            for name in ROI_NAMES:
                ids=list(FACE_ROIS[name]);center=observation.landmarks[ids].mean(0)
                observation.landmarks[ids]=center+1.5*(observation.landmarks[ids]-center)
        return observation


class OpticalDescriptorTests(unittest.TestCase):
    def test_original_values_unchanged_and_named_uncertainty(self):
        frame=uniform_skin();before=frame.copy()
        rgb,valid,q=optical_measurement(frame,np.ones((32,32),dtype=bool),16)
        np.testing.assert_array_equal(frame,before)
        np.testing.assert_array_equal(rgb,[180,130,110])
        self.assertTrue(valid)
        self.assertEqual(q['chroma_compatible_fraction'],1.)
        self.assertNotIn('physiological_confidence',q)

    def test_dark_hair_like_occlusion_reduces_proxy_not_confirmed_hair_label(self):
        frame=uniform_skin();frame[:,:24]=[20,20,20]
        _,valid,q=optical_measurement(frame,np.ones((32,32),dtype=bool),16)
        self.assertTrue(valid)
        self.assertEqual(q['chroma_compatible_fraction'],.25)
        self.assertNotIn('hair_fraction_confirmed',q)

    def test_skin_colored_hair_is_an_explicit_proxy_counterexample(self):
        frame=np.broadcast_to(np.array([150.,110.,80.]),(32,32,3)).copy()
        frame[:,::2]*=.95  # hair-like stripes with compatible color
        _,valid,q=optical_measurement(frame,np.ones((32,32),dtype=bool),16)
        self.assertTrue(valid)
        self.assertEqual(q['chroma_compatible_fraction'],1.)

    def test_saturation_clipping_both_endpoints(self):
        for color in (0.,255.):
            frame=np.full((8,8,3),color)
            _,valid,q=optical_measurement(frame,np.ones((8,8),dtype=bool),16)
            self.assertTrue(valid)
            self.assertEqual(q['clipped_fraction'],1.)

    def test_nan_out_of_range_and_empty_masks_remain_missing(self):
        for frame,mask in ((np.full((8,8,3),np.nan),np.ones((8,8),dtype=bool)),
                           (np.full((8,8,3),1e308),np.ones((8,8),dtype=bool)),
                           (np.ones((8,8,3)),np.zeros((8,8),dtype=bool))):
            rgb,valid,q=optical_measurement(frame,mask,16)
            self.assertFalse(valid)
            self.assertTrue(np.isnan(rgb).all())
            self.assertEqual(q['pixel_count'],0)
            self.assertEqual(q['finite_fraction'],0)

    def test_malformed_mask_frame_types_and_config_refused(self):
        with self.assertRaises(ValueError):optical_measurement(uniform_skin(),np.ones((32,32)),16)
        with self.assertRaises(ValueError):optical_measurement(uniform_skin().astype('u1'),np.ones((32,32),dtype=bool),16)
        for kw in (dict(score_floor=.2),dict(min_eligible_patches=1),dict(support_fraction=np.nan),dict(patch_min_pixels=0)):
            with self.assertRaises(ValueError):OpticalQualityConfig(**kw)

    def test_color_dispersion_is_descriptive_and_brightness_is_not_skin_truth(self):
        frame=uniform_skin();mask=np.ones((32,32),dtype=bool)
        _,_,a=optical_measurement(frame,mask,16)
        frame[:16]=[120.,90.,70.]
        _,_,b=optical_measurement(frame,mask,16)
        self.assertGreater(b['spatial_chroma_mad'],a['spatial_chroma_mad'])
        self.assertNotIn('poor_pulsatility',b)

    def test_flicker_and_motion_are_separate_nonphysiological_descriptors(self):
        rgb=np.repeat([[100.,120.,80.]],12,axis=0)
        rgb[::2]+=10
        centers=np.column_stack((np.linspace(0,1,12),np.zeros(12)))
        q=temporal_descriptors(rgb,centers)
        self.assertGreater(q['median_second_difference_flicker_proxy'],0)
        self.assertGreater(q['median_normalized_centroid_step'],0)
        self.assertEqual(q['valid_adjacent_pairs'],11)
        self.assertIn('not_physiological',q['interpretation'])


class WholeROIAndPatchTests(unittest.TestCase):
    def setUp(self):
        self.data=patches_from_image(uniform_skin())
        self.config=OpticalQualityConfig(block_samples=6)

    def run_candidate(self,data=None,**kw):
        return evaluate_candidate(*(self.data if data is None else data),config=self.config,**kw)

    def test_partition_is_disjoint_exact_union_and_stable_order(self):
        mask=np.zeros((32,32),dtype=bool);mask[3:27,6:24]=True
        cells=partition_polygon(mask)
        np.testing.assert_array_equal(np.sum(cells,axis=0),mask.astype(int))
        for a,b in zip(cells,partition_polygon(mask)):np.testing.assert_array_equal(a,b)

    def test_moving_patch_boundaries_keep_union_but_can_change_optical_means(self):
        frame=uniform_skin();frame[:,:,0]=np.arange(32)[None,:]*4+60
        means=[]
        for shift in (0,3):
            mask=np.zeros((32,32),dtype=bool);mask[4:24,4+shift:24+shift]=True
            cells=partition_polygon(mask)
            np.testing.assert_array_equal(np.logical_or.reduce(cells),mask)
            colors=[optical_measurement(frame,c,16)[0] for c in cells]
            means.append(np.mean(colors,axis=0))
        self.assertGreater(np.max(np.abs(means[1]-means[0])),0)

    def test_geometrically_valid_mostly_hair_rejects_whole_roi_even_with_good_patches(self):
        frame=uniform_skin();frame[:,:24]=[20,20,20]
        data=patches_from_image(frame)
        # Two patches are locally compatible, yet the whole ROI is 75% hair-like.
        self.assertGreaterEqual(int((data[2]['chroma_compatible_fraction'][0,:,0]>=.5).sum()),2)
        self.assertTrue(data[1].all())
        result=self.run_candidate(data)
        self.assertFalse(result['roi_valid'].any() or result['provisional_valid'].any())
        self.assertEqual(np.count_nonzero(result['planned_weights']),0)
        self.assertTrue((result['whole_roi_reason']=='ABSTAIN_WHOLE_ROI_LOW_SKIN_PROXY').all())

    def test_unknown_skin_abstains_entire_roi_despite_all_patch_scores_passing(self):
        result=self.run_candidate()
        self.assertTrue(result['provisional_valid'].all())
        self.assertFalse(result['roi_valid'].any())
        self.assertTrue(np.isnan(result['rgb']).all())
        self.assertEqual(np.count_nonzero(result['applied_weights']),0)
        self.assertTrue((result['whole_roi_reason']=='ABSTAIN_SKIN_UNVERIFIED').all())

    def test_skin_colored_hair_cannot_gain_trusted_admission_from_color_alone(self):
        frame=np.broadcast_to(np.array([150.,110.,80.]),(32,32,3)).copy()
        data=patches_from_image(frame)
        self.assertEqual(data[3]['chroma_compatible_fraction'][0,0],1.)
        result=self.run_candidate(data)
        self.assertTrue(result['provisional_valid'].all())
        self.assertFalse(result['roi_valid'].any())
        self.assertEqual(result['anatomical_skin_confidence'],'UNVERIFIED')

    def test_synthetic_positive_truth_allows_engineering_mix_without_training(self):
        result=self.run_candidate(independent_evidence=evidence())
        self.assertTrue(result['roi_valid'].all())
        np.testing.assert_allclose(result['rgb'][0],np.tile([180,130,110],(12,1)))
        np.testing.assert_allclose(result['applied_weights'].sum(1),1)
        self.assertFalse(result['allow_training'])
        self.assertEqual(result['physiological_pulse_quality'],'UNVERIFIED')

    def test_independent_occlusion_rejects_all_patches_at_whole_roi_level(self):
        result=self.run_candidate(independent_evidence=evidence(state='verified_non_skin_or_occluded'))
        self.assertFalse(result['provisional_valid'].any() or result['roi_valid'].any())
        self.assertTrue((result['whole_roi_reason']=='ABSTAIN_WHOLE_ROI_INDEPENDENT_OCCLUSION').all())

    def test_weights_are_frozen_within_block_and_nonzero_ratio_bounded(self):
        rgb,valid,pm,rm=self.data
        pm['chroma_compatible_fraction'][0,:,]=np.linspace(.5,1,6)[:,None]
        result=self.run_candidate(independent_evidence=evidence())
        w=result['planned_weights']
        self.assertLessEqual(w.max()/w[w>0].min(),2)
        for block in (slice(0,6),slice(6,12)):
            self.assertTrue(np.all(np.diff(result['applied_weights'][:,:,block],axis=-1)==0))

    def test_missing_weighted_patch_abstains_instead_of_dynamic_renormalization(self):
        rgb,valid,pm,rm=patches_from_image(uniform_skin(),n=300)
        valid[0,0,100]=False;rgb[0,0,100]=np.nan
        result=evaluate_candidate(rgb,valid,pm,rm,independent_evidence=IndependentSkinEvidence(
            np.array([['verified_skin']]),'controlled_synthetic_ground_truth','fixture'))
        self.assertFalse(result['roi_valid'][0,100])
        self.assertTrue(np.isnan(result['rgb'][0,100]).all())
        self.assertEqual(np.count_nonzero(result['applied_weights'][0,:,100]),0)
        self.assertTrue(result['roi_valid'][0,99] and result['roi_valid'][0,101])

    def test_transient_saturation_abstains_without_selecting_other_patches(self):
        rgb,valid,pm,rm=patches_from_image(uniform_skin(),n=300)
        pm['clipped_fraction'][0,0,100]=1
        result=evaluate_candidate(rgb,valid,pm,rm)
        self.assertFalse(result['provisional_valid'][0,100])
        self.assertTrue(result['provisional_valid'][0,99])

    def test_homogeneous_color_changes_or_spatial_mad_do_not_directly_drive_selection(self):
        one=self.run_candidate()
        self.data[2]['brightness'][:]=.01
        self.data[2]['spatial_chroma_mad'][:]=.9
        two=self.run_candidate()
        np.testing.assert_array_equal(one['planned_weights'],two['planned_weights'])

    def test_all_missing_and_too_few_patches_retain_full_denominator(self):
        rgb,valid,pm,rm=self.data
        valid[:]=False;rgb[:]=np.nan
        result=self.run_candidate()
        self.assertEqual(result['roi_valid'].shape,(1,12))
        self.assertFalse(result['roi_valid'].any())
        self.assertTrue((result['whole_roi_reason']=='ABSTAIN_TOO_FEW_ELIGIBLE_PATCHES').all())

    def test_malformed_evidence_fractions_and_weights_rejected(self):
        bad=IndependentSkinEvidence(np.array([['verified_skin']]),'algorithmic_color_proxy','not-independent')
        with self.assertRaises(ValueError):self.run_candidate(independent_evidence=bad)
        self.data[3]['chroma_compatible_fraction'][:]=1.1
        with self.assertRaises(ValueError):self.run_candidate()
        rgb,valid,_,_=patches_from_image(uniform_skin())
        with self.assertRaises(ValueError):frozen_patch_mix(rgb,valid,np.ones((1,2,6)),6)

    def test_numerical_determinism_and_source_arrays_not_mutated(self):
        rgb=self.data[0].copy()
        one=self.run_candidate();two=self.run_candidate()
        np.testing.assert_array_equal(one['provisional_rgb'],two['provisional_rgb'])
        np.testing.assert_array_equal(rgb,self.data[0])

    def test_weight_induced_step_is_reported_separately(self):
        patches=np.ones((1,2,12,3))*np.array([100.,140.])[None,:,None,None]
        weights=np.zeros((1,2,12));weights[0,0,:6]=2/3;weights[0,0,6:]=1/3;weights[0,1]=1-weights[0,0]
        d=weight_change_diagnostics(patches,weights)
        self.assertAlmostEqual(d['weight_l1_step'][0,5],2/3)
        self.assertAlmostEqual(d['weight_change_rgb_term'][0,5,0],40/3)


class ReferenceFreeCollectionTests(unittest.TestCase):
    def test_only_xsub_values_are_read_and_baseline_a_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=hdf_fixture(root)
            layout=ScampsLayout(frames=6,raw_width=40,crop_width=32,height=32)
            reader=ScampsRecording(root,path.name,layout=layout,max_window_frames=3)
            original=h5py.Dataset.__getitem__
            def no_reference(ds,key):
                self.assertEqual(ds.name,'/Xsub')
                return original(ds,key)
            before=source_sha256(path)
            with patch.object(h5py.Dataset,'__getitem__',new=no_reference):
                result=collect_optical_evidence(reader,WideFixtureDetector(),chunk_frames=3,sample_indices=(0,3,5))
            baseline=extract_scamps_roi_traces(reader,WideFixtureDetector(),config=ScampsROIConfig(chunk_frames=3))
            self.assertTrue(baseline.roi_valid.all())
            np.testing.assert_array_equal(result['baseline'].rgb,baseline.rgb)
            np.testing.assert_array_equal(result['baseline'].roi_valid,baseline.roi_valid)
            self.assertTrue(np.isnan(result['baseline'].raw_d_ppg).all())
            self.assertFalse(result['baseline'].provenance['reference_values_decoded'])
            self.assertEqual(source_sha256(path),before)

    def test_dropped_landmarks_keep_all_frames_and_missing_optical_indicators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=hdf_fixture(root)
            reader=ScampsRecording(root,path.name,layout=ScampsLayout(frames=6,raw_width=40,crop_width=32,height=32),max_window_frames=3)
            result=collect_optical_evidence(reader,FakeDetector(missing=(2,)),chunk_frames=3,sample_indices=(2,))
            self.assertEqual(result['baseline'].rgb.shape,(3,6,3))
            self.assertTrue(np.isnan(result['patch_rgb'][:,:,2]).all())
            self.assertIsNone(result['contacts'][2]['masks'])

    def test_annotation_template_cannot_be_mistaken_for_human_assessment(self):
        row=review_template('opaque_id',299,'forehead')
        self.assertEqual(row['status'],'PENDING_HUMAN_REVIEW')
        self.assertIsNone(row['reviewer_1']['reviewer_id'])
        self.assertIsNone(row['reviewer_2']['skin_fraction_bin'])
        self.assertIsNone(row['adjudication'])
        self.assertIsNone(clean(float('nan')))


class QualityTimingTests(unittest.TestCase):
    def test_identical_phase_frozen_mixing_passes_synthetic_only(self):
        for fs in (30.,60.):
            row=timing_cases(fs)[0]
            self.assertEqual(row['outcome'],'PASS_SYNTHETIC_ONLY')

    def test_bounded_unequal_spatial_phases_fail_camera_anchoring(self):
        for fs in (30.,60.):
            rows={r['case']:r for r in timing_cases(fs)}
            self.assertEqual(rows['symmetric_phase_uniform_control']['outcome'],'PASS_SYNTHETIC_ONLY')
            self.assertEqual(rows['heterogeneous_phase_bounded_unequal_weights']['reason'],'camera_time_anchor_shift')

    def test_switching_and_boundaries_do_not_earn_timing_acceptance(self):
        for row in timing_cases(30.):
            if row['case'] in ('unsafe_rapid_switching_negative_control','block_boundary_weight_change'):
                self.assertNotEqual(row['outcome'],'PASS_SYNTHETIC_ONLY')
                self.assertGreater(row['maximum_weight_induced_rgb_step'],10.)

    def test_missing_patch_is_not_filled_for_timing_and_runs_are_deterministic(self):
        rows=timing_cases(30.)
        self.assertEqual(rows[-1]['outcome'],'ABSTAIN_UNVERIFIED')
        self.assertEqual(rows,timing_cases(30.))


if __name__=='__main__':unittest.main()

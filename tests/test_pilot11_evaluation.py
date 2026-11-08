import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from rppg_lab.pilot11_inference import FACE_NAMES,prepared_windows,without_pressure_values
from rppg_lab.config import PipelineConfig
from rppg_lab.pilot11_bp import fit_ridge,predict,session_reference,score_locked_predictions
from rppg_lab.artifacts import file_sha256


def check_primary_and_native_rate_keep_same_physical_windows_and_large_gap():
    t=np.arange(2400)/60;signal=np.sin(2*np.pi*1.2*t)
    rgb=100+np.stack((signal,.7*signal,.3*signal),1);valid=np.ones(len(t),bool);valid[(t>=12)&(t<13)]=False
    rgb[~valid]=np.nan;native={'timestamps':t}
    for name in FACE_NAMES:native.update({name+'_rgb':rgb,name+'_extraction_valid':valid})
    config=PipelineConfig(regions=('face',),face_rois=FACE_NAMES,face_roi='forehead',max_gap_sec=.050001)
    a30,r30,_=prepared_windows(native,30,config);a60,r60,_=prepared_windows(native,60,config)
    assert [r['start_sec'] for r in r30]==[r['start_sec'] for r in r60]
    assert a30['roi'][:,0].tolist()==[True,False,True]
    assert a60['roi'][:,0].tolist()==[True,False,True]
    assert a30['x'].shape==(3,3,9,300) and a60['x'].shape==(3,3,9,600)
    assert np.all(a30['x'][1]==0) and np.all(a60['x'][1]==0)


def check_pressure_and_demographics_cannot_reach_signal_analysis():
    s={'raw':{'SBP1_mmHg':121,'DBP1_mmHg':80,'CuffHR1_bpm':70,'BP1_Start_s':15,'Event_Time_Source':'Legacy'},'subject_raw':{'Age':35}}
    changed={'raw':dict(s['raw'],SBP1_mmHg=190,DBP1_mmHg=100,CuffHR1_bpm=120),'subject_raw':{'Age':80}}
    assert without_pressure_values(s)==without_pressure_values(changed)
    assert without_pressure_values(s)['raw']['BP1_Start_s']==15
    assert s['subject_raw']=={'Age':35}  # Input remains intact.


def check_bp_transform_is_training_only_and_mean_baseline_is_exact():
    x=np.array([[1,np.nan],[2,4],[3,8]],float);y=np.array([[100,60],[110,70],[120,80]],float)
    fit=fit_ridge(x,y);before={k:v.copy() for k,v in fit.items() if isinstance(v,np.ndarray)}
    first=predict(fit,[[np.nan,np.nan]]);predict(fit,[[1e9,-1e9]])
    for key,value in before.items():np.testing.assert_array_equal(fit[key],value)
    np.testing.assert_array_equal(fit['median'],[2,6]);np.testing.assert_allclose(first,[[110,70]])
    mean=fit_ridge(np.empty((3,0)),y);np.testing.assert_allclose(predict(mean,np.empty((2,0))),[[110,70],[110,70]])
    # Fold preprocessing must change when an extreme person is held out.
    fold=fit_ridge(np.array([[1],[2]],float),y[:2]);np.testing.assert_array_equal(fold['median'],[1.5])


def check_bp_exact_session_reference_uses_one_record_not_repeated_windows():
    s={'raw':{'SBP1_mmHg':100,'DBP1_mmHg':60,'SBP2_mmHg':120,'DBP2_mmHg':80,'SBP3_mmHg':None,'DBP3_mmHg':None}}
    mean,count=session_reference(s);np.testing.assert_array_equal(mean,[110,70]);assert count==2
    with unittest.TestCase().assertRaises(ValueError):session_reference({'raw':{'SBP1_mmHg':60,'DBP1_mmHg':100}})


def check_heldout_scores_require_a_preexisting_prediction_lock(tmp_path):
    with unittest.TestCase().assertRaises(FileNotFoundError):score_locked_predictions(tmp_path/'missing.json',[],tmp_path)
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'models':{}}))
    result=score_locked_predictions(lock,[],tmp_path)
    assert result['event_level_bp_available'] is False and result['timing_increment_available'] is False


class PilotEvaluationTests(unittest.TestCase):
    def test_rate_and_gap_support(self):check_primary_and_native_rate_keep_same_physical_windows_and_large_gap()
    def test_pressure_firewall(self):check_pressure_and_demographics_cannot_reach_signal_analysis()
    def test_train_only_preprocessing(self):check_bp_transform_is_training_only_and_mean_baseline_is_exact()
    def test_session_reference(self):check_bp_exact_session_reference_uses_one_record_not_repeated_windows()
    def test_prediction_lock(self):
        with tempfile.TemporaryDirectory() as folder:check_heldout_scores_require_a_preexisting_prediction_lock(Path(folder))
    def test_scoring_keeps_frozen_predictions_when_reference_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);path=folder/'locked.json'
            entry=dict(evaluation_subjects=['A','B','C'],predictions_mmHg=[[105,65]]*3,
                mean_baseline_mmHg=[[110,70]]*3,hr_baseline_mmHg=[[108,68]]*3)
            path.write_text(json.dumps({'models':{'fixed':entry}}));digest=file_sha256(path)
            sessions=[{'subject_id':s,'raw':{'SBP1_mmHg':100,'DBP1_mmHg':60}} for s in ('A','B','C')]
            score=score_locked_predictions(path,sessions,folder)['models']['fixed']
            np.testing.assert_array_equal(score['mae_mmHg'],[5,5])
            np.testing.assert_array_equal(score['improvement_over_mean_baseline_mmHg']['mae_reduction_mmHg'],[5,5])
            for s in sessions:s['raw'].update(SBP1_mmHg=120,DBP1_mmHg=80)
            changed=score_locked_predictions(path,sessions,folder)['models']['fixed']
            np.testing.assert_array_equal(changed['mae_mmHg'],[15,15]);assert file_sha256(path)==digest


if __name__=='__main__':unittest.main()

"""Read-only saved-result replay, with an independent sklearn ridge pipeline."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from rppg_lab.artifacts import file_sha256,write_json
from rppg_lab.pilot11_bp import session_predictors,session_reference


def independent_prediction(x,y,test):
    if x.shape[1]==0:return np.tile(y.mean(0),(len(test),1))
    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=10.,solver='svd'))
    x=np.where(np.isfinite(x),x,np.nan);test=np.where(np.isfinite(test),test,np.nan)
    return model.fit(x,y).predict(test)


def verify(root):
    root=Path(root);scope_path=Path('docs/waveform_research/pilot11_model_and_scope_lock_20261108.json');scope=json.loads(scope_path.read_text())
    for row in scope['checkpoints']:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Frozen evaluation model changed')
    expected={'checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt':'d6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff',
        'checkpoints/waveform_v1_official_ubfc/waveform_v1_freeze_manifest.json':'da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc',
        'outputs/waveform_screen/development_candidate_20261108_01/waveform_v2_development_candidate.pt':'2851cc36fea693113a27359ef5fb18741595eea26fcf2852e35dd768464e00a9'}
    for path,digest in expected.items():
        if file_sha256(path)!=digest:raise ValueError('Preserved prior artifact changed')
    sessions=json.loads((root/'private_metadata/matched_sessions.json').read_text());by_id={s['subject_id']:s for s in sessions}
    lock_path=root/'bp/bp_prediction_lock.json';lock=json.loads(lock_path.read_text());report=json.loads((root/'bp/bp_summary.json').read_text())
    if report['prediction_lock_sha256']!=file_sha256(lock_path):raise ValueError('Prediction lock changed')
    if lock['matched_session_metadata_sha256']!=file_sha256(root/'private_metadata/matched_sessions.json'):raise ValueError('Matched reference metadata changed')
    if lock['metadata_snapshot_sha256']!=file_sha256(root/'private_metadata/metadata_snapshot.xlsx'):raise ValueError('Workbook snapshot changed')
    if lock['source_sha256']!=file_sha256('rppg_lab/pilot11_bp.py'):raise ValueError('BP implementation changed since predictions')
    features={};artifact_count=0
    for subject in scope['subjects']:
        folder=root/'evaluation'/by_id[subject]['session_id'];receipt=folder/'evaluation_complete.json'
        if file_sha256(receipt)!=lock['input_receipt_sha256'][subject]:raise ValueError('Input receipt changed')
        completed=json.loads(receipt.read_text())
        for name,digest in completed['artifact_sha256'].items():
            if file_sha256(folder/name)!=digest:raise ValueError('Saved inference/timing artifact changed')
            artifact_count+=1
        features[subject]=session_predictors(folder/'inference')[0]
    comparisons=[]
    def same(a,b):
        a,b=np.asarray(a,float),np.asarray(b,float);np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-9)
        comparisons.extend(abs(a-b).reshape(-1).tolist())
    for name,model in lock['models'].items():
        train=model['training_subjects'];test=model['evaluation_subjects']
        if set(train)&set(test) or not set(train)<=set(scope['bp_development_subjects']) or not set(test)<=set(scope['bp_evaluation_subjects']):raise ValueError('BP subject leakage')
        x=np.stack([features[s][name] for s in train]);xt=np.stack([features[s][name] for s in test]);y=np.stack([session_reference(by_id[s])[0] for s in train])
        prediction=independent_prediction(x,y,xt);same(prediction,model['predictions_mmHg'])
        same(np.tile(y.mean(0),(len(test),1)),model['mean_baseline_mmHg'])
        same(independent_prediction(np.stack([features[s]['HR_only'] for s in train]),y,np.stack([features[s]['HR_only'] for s in test])),model['hr_baseline_mmHg'])
        target=np.stack([session_reference(by_id[s])[0] for s in test]);error=prediction-target
        for key,value in (('mae_mmHg',np.mean(abs(error),axis=0)),('bias_mmHg',error.mean(0)),('rmse_mmHg',np.sqrt(np.mean(error**2,axis=0)))):same(value,report['models'][name][key])
        loo=[]
        for i in range(len(train)):
            keep=np.arange(len(train))!=i;loo.append(independent_prediction(x[keep],y[keep],x[i:i+1])[0])
        error=np.asarray(loo)-y
        for key,value in (('mae_mmHg',np.mean(abs(error),axis=0)),('bias_mmHg',error.mean(0)),('rmse_mmHg',np.sqrt(np.mean(error**2,axis=0)))):same(value,report['development'][name]['leave_one_subject_out'][key])
    result=dict(checked_numeric_values=len(comparisons),maximum_absolute_difference=max(comparisons) if comparisons else None,
        verified_evaluation_artifacts=artifact_count,retained_models_verified=len(scope['checkpoints']),prior_artifacts_verified=len(expected),
        prediction_lock_sha256=file_sha256(lock_path),independent_fitter='sklearn median imputer + StandardScaler + SVD Ridge alpha10',
        neural_inference_rerun=False,waveform_fitting=False,source_sha256=file_sha256(__file__))
    write_json(root/'bp/numerical_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    print(json.dumps(verify(Path(args.run))))

"""Small, fixed recording-level BP feasibility study; no waveform training."""
import json
from pathlib import Path
import warnings
import numpy as np
from .artifacts import file_sha256,json_safe,write_json

MODELS=('train_mean','HR_only','HR_compact_geometry','HR_TCN_geometry','HR_joint_geometry','HR_independent_geometry')
GEOMETRY=(3,4,6)


def median_finite(values,axis=0):
    values=np.asarray(values,float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmedian(values,axis=axis)


def session_predictors(folder):
    """Pressure-free aggregation; seed averaging precedes window medians."""
    folder=Path(folder)
    with np.load(folder/'features.npz',allow_pickle=False) as z,np.load(folder/'feature_validity.npz',allow_pickle=False) as valid:
        eligible=valid['30Hz__candidate_cuff_free']&valid['30Hz__face_quality_good']
        hr=np.where(valid['30Hz__face_GREEN'][:,0]&eligible,z['30Hz__face_GREEN'][:,0],np.nan)
        hr_value=float(median_finite(hr))
        features={'train_mean':np.empty(0),'HR_only':np.asarray([hr_value])};counts={}
        scope=json.loads(Path('docs/waveform_research/pilot11_model_and_scope_lock_20261108.json').read_text())
        tcn_names={c['identity']['configuration']['name'] for c in scope['checkpoints'] if c['kind']=='waveform' and c['identity']['configuration']['architecture']=='long_tcn'}
        if len(tcn_names)!=1:raise ValueError('One fixed TCN family required')
        families={'compact':'morph_real_only__seed','TCN':next(iter(tcn_names))+'__seed','joint':'joint__seed','independent':'independent__seed'}
        for family,prefix in families.items():
            names=[n for n in z.files if n.startswith('30Hz__'+prefix)]
            if len(names)!=3:raise ValueError('Exactly three predeclared seeds required for '+family)
            values=np.stack([z[n][:,GEOMETRY] for n in sorted(names)])
            support=np.stack([valid[n][:,GEOMETRY] for n in sorted(names)])
            support&=np.isfinite(values)&(values>0)&(values<=1)
            keep=support.all(0)&eligible[:,None]
            averaged=np.where(keep,values.mean(0),np.nan)
            geometry=median_finite(averaged)
            features['HR_'+family+'_geometry']=np.r_[hr_value,geometry]
            counts[family]=dict(valid_windows_per_geometry=keep.sum(0).tolist(),windows_with_any_geometry=int(keep.any(1).sum()),
                windows_with_all_geometry=int(keep.all(1).sum()),seed_members=sorted(names))
        return features,dict(attempted_windows=len(eligible),eligible_windows=int(eligible.sum()),valid_hr_windows=int(np.isfinite(hr).sum()),geometry=counts)


def fit_ridge(x,y,alpha=10.):
    """All imputation/scaling parameters depend only on this training matrix."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    med=median_finite(x);med=np.where(np.isfinite(med),med,0.)
    complete=np.where(np.isfinite(x),x,med);center=complete.mean(0);scale=complete.std(0)
    scale=np.where(scale>1e-8,scale,1.);standard=(complete-center)/scale
    intercept=y.mean(0)
    coef=np.linalg.solve(standard.T@standard+alpha*np.eye(x.shape[1]),standard.T@(y-intercept))
    return dict(median=med,center=center,scale=scale,coef=coef,intercept=intercept,alpha=alpha,training_rows=len(x))


def predict(fit,x):
    x=np.asarray(x,float);complete=np.where(np.isfinite(x),x,fit['median'])
    return ((complete-fit['center'])/fit['scale'])@fit['coef']+fit['intercept']


def session_reference(session):
    pairs=[]
    for i in (1,2,3):
        a,b=(session['raw'].get(f'{p}{i}_mmHg') for p in ('SBP','DBP'))
        if a is None or b is None:continue
        if isinstance(a,bool) or isinstance(b,bool):raise ValueError('Boolean BP is invalid')
        a,b=float(a),float(b)
        if not np.isfinite([a,b]).all() or not 0<b<a:raise ValueError('Invalid BP pair')
        pairs.append([a,b])
    if not pairs:raise ValueError('No exact-session numeric cuff reference')
    return np.mean(pairs,axis=0),len(pairs)


def metrics(error):
    error=np.asarray(error,float)
    return dict(mae_mmHg=np.mean(abs(error),axis=0),bias_mmHg=error.mean(0),rmse_mmHg=np.sqrt(np.mean(error**2,axis=0)))


def score_locked_predictions(lock_path,evaluation_sessions,out):
    """Called only after the held-out prediction lock exists on disk."""
    lock_path=Path(lock_path)
    lock=json.loads(lock_path.read_text());references={s['subject_id']:session_reference(s) for s in evaluation_sessions}
    scores={};private={};rng=np.random.default_rng(20261108)
    for name,row in lock['models'].items():
        ids=row['evaluation_subjects'];y=np.stack([references[s][0] for s in ids]);p=np.asarray(row['predictions_mmHg'])
        error=p-y;entry=dict(subjects=len(ids),cuff_readings=sum(references[s][1] for s in ids),**metrics(error))
        for baseline in ('mean_baseline_mmHg','hr_baseline_mmHg'):
            base=np.asarray(row[baseline]);gain=abs(base-y)-abs(error)
            boots=np.mean(gain[rng.integers(0,len(ids),(10000,len(ids)))],axis=1)
            entry['improvement_over_'+baseline]=dict(mae_reduction_mmHg=gain.mean(0),descriptive_95pct_subject_bootstrap_mmHg=np.percentile(boots,[2.5,97.5],axis=0))
        entry['evaluation_reference_range_mmHg']=np.ptp(y,axis=0);entry['evaluation_reference_sd_mmHg']=y.std(0)
        scores[name]=entry;private[name]=dict(subject_ids=ids,reference_mmHg=y,prediction_mmHg=p,error_mmHg=error)
    write_json(Path(out)/'private_bp_errors.json',private)
    return dict(prediction_lock_sha256=file_sha256(lock_path),targets=['SBP','DBP'],unit='mmHg',models=scores,
        event_level_bp_available=False,timing_increment_available=False,clinical_validation=False,
        interpretation='Exploratory session-mean transfer; legacy cuff timing, tiny fixed evaluation cohort, no waveform/timing ground truth')


def run_bp_study(root,scope_path):
    root=Path(root);out=root/'bp';out.mkdir(exist_ok=True)
    scope=json.loads(Path(scope_path).read_text());sessions=json.loads((root/'private_metadata/matched_sessions.json').read_text())
    by_id={s['subject_id']:s for s in sessions};features={};coverage={}
    if file_sha256(root/'private_metadata/metadata_snapshot.xlsx')!='785cf7f989fd9473ba8dd6bafcc17c0f0e9abd7149416905228970584ca38930':raise ValueError('Metadata snapshot changed')
    input_hashes={}
    for subject in scope['subjects']:
        session=by_id[subject]
        folder=root/'evaluation'/session['session_id'];receipt=folder/'evaluation_complete.json'
        completed=json.loads(receipt.read_text())
        if completed['identity']['scope_sha256']!=file_sha256(scope_path):raise ValueError('Evaluation scope changed')
        for name,digest in completed['artifact_sha256'].items():
            if file_sha256(folder/name)!=digest:raise ValueError('Frozen evaluation artifact changed')
        input_hashes[subject]=file_sha256(receipt)
        features[subject],coverage[subject]=session_predictors(folder/'inference')
    write_json(out/'private_feature_coverage.json',coverage)
    train_ids=scope['bp_development_subjects'];test_ids=scope['bp_evaluation_subjects']
    if set(train_ids)&set(test_ids):raise ValueError('Subject leakage')
    # Only development pressures are read while fitting and creating predictions.
    y_train={s:session_reference(by_id[s])[0] for s in train_ids}
    locked=dict(scope_sha256=file_sha256(scope_path),source_sha256=file_sha256(Path(__file__)),input_receipt_sha256=input_hashes,
        alpha=10.,feature_source='30-Hz fixed frozen-model session medians',
        no_bp_based_feature_or_model_selection=True,models={});dev={}
    for name in MODELS:
        def usable(s):
            x=features[s][name]
            return np.isfinite(features[s]['HR_only']).all() and (name in ('train_mean','HR_only') or np.isfinite(x[1:]).any())
        train=[s for s in train_ids if usable(s)];test=[s for s in test_ids if usable(s)]
        if len(train)<4 or len(test)<3:
            dev[name]=dict(status='unavailable',development_subjects=len(train),evaluation_subjects=len(test));continue
        x=np.stack([features[s][name] for s in train]);xt=np.stack([features[s][name] for s in test]);y=np.stack([y_train[s] for s in train])
        fit=fit_ridge(x,y);p=predict(fit,xt);loo=[]
        for i in range(len(train)):
            keep=np.arange(len(train))!=i;loo.append(predict(fit_ridge(x[keep],y[keep]),x[i:i+1])[0])
        hr=fit_ridge(np.stack([features[s]['HR_only'] for s in train]),y)
        locked['models'][name]=dict(training_subjects=train,evaluation_subjects=test,predictions_mmHg=p,
            mean_baseline_mmHg=np.tile(y.mean(0),(len(test),1)),hr_baseline_mmHg=predict(hr,np.stack([features[s]['HR_only'] for s in test])),
            fit=fit,evaluation_missing_features=(~np.isfinite(xt)).sum(0),training_missing_features=(~np.isfinite(x)).sum(0))
        dev[name]=dict(status='available',subjects=len(train),leave_one_subject_out=metrics(np.asarray(loo)-y))
    lock_path=out/'bp_prediction_lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text())!=json_safe(locked):raise ValueError('Predictions differ from existing held-out lock')
    else:
        with lock_path.open('x',encoding='utf-8') as stream:json.dump(json_safe(locked),stream,indent=2,allow_nan=False)
    write_json(out/'development_loso.json',dev)
    # Evaluation labels enter scoring only after the immutable prediction file.
    report=score_locked_predictions(lock_path,[by_id[s] for s in test_ids],out)
    report['development']=dev
    report['coverage']=dict(subjects=len(features),eligible_face_windows=sum(c['eligible_windows'] for c in coverage.values()),
        subjects_with_hr=sum(np.isfinite(f['HR_only']).all() for f in features.values()))
    write_json(out/'bp_summary.json',report)
    return report

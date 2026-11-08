"""Run the explicitly approved, consistent full-recording pilot acquisition."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
from rppg_lab.artifacts import file_sha256,write_json
from rppg_lab.pilot11_acquisition import acquire_session


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--resume',action='store_true');args=p.parse_args()
    allowed=(Path.cwd()/'outputs/pilot11_new_models').resolve()
    if allowed not in args.run.resolve().parents or not 1<=args.workers<=3:p.error('Use the authorized pilot namespace and <=3 workers')
    scope=json.loads(Path('docs/waveform_research/pilot11_model_and_scope_lock_20261108.json').read_text())
    sessions=json.loads((args.run/'private_metadata/matched_sessions.json').read_text());ids=[s['subject_id'] for s in sessions]
    if sorted(ids)!=scope['subjects']:raise ValueError('Cohort differs from the authorized eleven')
    for s in sessions:
        if str(s['subject_raw'].get('Video_Consent_Confirmed')).casefold()!='yes':raise ValueError('Video consent not confirmed')
        if str(s['raw'].get('Manual_Exclude','')).casefold() in ('yes','true','1'):raise ValueError('Explicitly excluded participant')
    manifest=Path('checkpoints/waveform_v1_official_ubfc/waveform_v1_freeze_manifest.json')
    if file_sha256(manifest)!='da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc':raise ValueError('Frozen manifest changed')
    config=json.loads(manifest.read_text())['extraction_config'];output=args.run/'consistent_acquisition';output.mkdir(exist_ok=True)
    receipts=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(acquire_session,s['video'],output/s['session_id'],config,args.resume):s['session_id'] for s in sessions}
        for future in as_completed(jobs):
            try:receipt=future.result()
            except Exception as error:
                receipt=dict(session_id=jobs[future],status='failed',error_type=type(error).__name__,detail=str(error))
            receipts.append(receipt);write_json(args.run/'acquisition_stage.json',dict(completed=len(receipts),intended=len(sessions),records=receipts))
            print(json.dumps(dict(session_id=receipt['session_id'],status=receipt['status'],completed=len(receipts),intended=len(sessions))),flush=True)
    if any(r['status']=='failed' for r in receipts):raise RuntimeError('At least one acquisition failed; inspect local receipt')


if __name__=='__main__':main()

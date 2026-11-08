"""Consume only the authorized consistent acquisitions as they complete."""
import argparse
import json
from pathlib import Path
import time
import torch
from rppg_lab.artifacts import file_sha256,write_json
from rppg_lab.pilot11_inference import load_models
from rppg_lab.pilot11_evaluation import evaluate_session


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();root=Path(args.run)
    scope_path=Path('docs/waveform_research/pilot11_model_and_scope_lock_20261108.json');scope=json.loads(scope_path.read_text())
    sessions=json.loads((root/'private_metadata/matched_sessions.json').read_text())
    if isinstance(sessions,dict):sessions=sessions['sessions']
    if sorted(s['subject_id'] for s in sessions)!=sorted(scope['subjects']):raise ValueError('Exact eleven-subject scope required')
    snapshot=root/'private_metadata/metadata_snapshot.xlsx'
    if file_sha256(snapshot)!='785cf7f989fd9473ba8dd6bafcc17c0f0e9abd7149416905228970584ca38930':raise ValueError('Authoritative metadata snapshot changed')
    config=json.loads(Path('configs/pilot6_delay_v1.json').read_text())
    torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
    models=load_models(scope_path,device);completed={}
    while True:
        for session in sessions:
            sid=session['session_id'];path=root/'consistent_acquisition'/sid/'native_rgb.npz'
            if sid in completed or not path.exists():continue
            print(json.dumps(dict(stage='evaluation_start',session_id=sid,device=device)),flush=True)
            completed[sid]=evaluate_session(path,session,models,device,config,root/'evaluation'/sid,scope_path)
            write_json(root/'evaluation_stage.json',dict(completed=list(completed),remaining=[s['session_id'] for s in sessions if s['session_id'] not in completed]))
            print(json.dumps(dict(stage='evaluation_complete',session_id=sid)),flush=True)
        if len(completed)==len(sessions) or not args.watch:break
        time.sleep(15)
    print(json.dumps(dict(stage='label_free_evaluation_finished',completed=len(completed))),flush=True)


if __name__=='__main__':main()

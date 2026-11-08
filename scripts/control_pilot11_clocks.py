"""Known synthetic delays sampled on actual MP4 clocks; no physiological truth."""
import argparse
import json
from pathlib import Path
from rppg_lab.artifacts import file_sha256,write_json
from rppg_lab.pilot_video import mp4_presentation_times
from rppg_lab.pilot6_delay_validation import run_synthetic_controls


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();root=Path(args.run)
    scope=json.loads(Path('docs/waveform_research/pilot11_model_and_scope_lock_20261108.json').read_text())
    audit=json.loads((root/'private_metadata/video_clock_audit.json').read_text());clocks={}
    for row in audit:
        if row['session_id'].split('_')[0] not in scope['subjects']:raise ValueError('Unauthorized clock')
        if file_sha256(row['video_path'])!=row['video_sha256']:raise ValueError('Video identity changed')
        clocks[row['session_id']]=mp4_presentation_times(row['video_path'])[0]
    out=root/'clock_controls';out.mkdir(exist_ok=True);config=json.loads(Path('configs/pilot6_delay_v1.json').read_text())
    trial,summary=run_synthetic_controls(out,config,clocks)
    write_json(out/'scope.json',dict(actual_clocks=len(clocks),attempted=len(trial),physiological_validation=False,
        acquisition_calibration=False,scope='Synthetic known lag, original camera PTS, inherited fixed protocol'))
    print('Clock controls complete:',len(trial),'synthetic attempts; no biological timing validation')

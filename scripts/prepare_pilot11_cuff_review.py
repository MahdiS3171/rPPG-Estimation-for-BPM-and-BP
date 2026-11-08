"""Local BP-blinded review queue; media metadata never verifies cuff events."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from rppg_lab.artifacts import file_sha256,write_json


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();root=Path(args.run)
    scope_path=Path('docs/waveform_research/pilot11_model_and_scope_lock_20261108.json');scope=json.loads(scope_path.read_text())
    metadata=root/'private_metadata/matched_sessions.json';sessions=json.loads(metadata.read_text())
    lock=json.loads((root/'bp/bp_prediction_lock.json').read_text())
    if file_sha256(metadata)!=lock['matched_session_metadata_sha256']:raise ValueError('Reference metadata changed')
    if sorted(s['subject_id'] for s in sessions)!=sorted(scope['subjects']):raise ValueError('Scope differs')
    clocks={r['session_id']:r for r in json.loads((root/'private_metadata/video_clock_audit.json').read_text())}
    probe=shutil.which('ffprobe');audit=[];queue=[]
    for s in sessions:
        clock=clocks[s['session_id']]
        if file_sha256(s['video'])!=clock['video_sha256']:raise ValueError('Video changed')
        audio=None
        if probe:
            command=[probe,'-v','error','-show_entries','stream=codec_type,codec_name,sample_rate,channels,start_time,duration','-of','json',s['video']]
            result=subprocess.run(command,check=True,capture_output=True,text=True)
            audio=[r for r in json.loads(result.stdout)['streams'] if r['codec_type']=='audio']
        audit.append(dict(session_id=s['session_id'],video_sha256=clock['video_sha256'],audio_streams=audio))
        for index in (1,2,3):
            a=s['raw'].get(f'BP{index}_Start_s');b=s['raw'].get(f'BP{index}_End_s')
            queue.append(dict(session_id=s['session_id'],event_id=f'{s["session_id"]}_BP{index}',video=s['video'],
                video_sha256=clock['video_sha256'],recorded_start_sec=a,recorded_end_sec=b,
                declared_source=s['raw'].get('Event_Time_Source'),
                suggested_review_start_sec=max(0,float(a)-10) if a is not None else 0.,
                suggested_review_end_sec=min(clock['duration_seconds'],float(b)+10) if b is not None else clock['duration_seconds'],
                actual_start_sec=None,actual_end_sec=None,status='pending_actual_evidence_review',
                timing_verified=False,available_audio_tracks=len(audio) if audio is not None else None))
    out=root/'cuff_review';out.mkdir(exist_ok=True)
    queue_path=out/'private_blinded_review_queue.json'
    if queue_path.exists() and json.loads(queue_path.read_text())!=queue:
        raise FileExistsError('Preserve the existing review/annotations; use a new namespace rather than replacing them')
    write_json(out/'private_media_stream_audit.json',audit)
    if not queue_path.exists():write_json(queue_path,queue)
    summary=dict(events=len(queue),recordings=len(audit),recordings_with_audio=sum(bool(r['audio_streams']) for r in audit),
        audio_probe_available=probe is not None,verified_events=0,bp_values_in_review_queue=False,
        metadata_sha256=file_sha256(metadata),source_sha256=file_sha256(__file__),
        interpretation='Audio-track presence supplies a possible review channel, not verified cuff sounds or physiological timing')
    write_json(out/'aggregate_summary.json',summary)
    print(json.dumps(summary))

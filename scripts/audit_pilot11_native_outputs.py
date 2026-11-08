"""Reference-free native-60 bandwidth audit; high-frequency power is not detail."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.signal import periodogram
from rppg_lab.artifacts import file_sha256,write_json


def audit(root):
    root=Path(root);groups=defaultdict(list);private=[];receipts=[]
    for receipt in sorted((root/'evaluation').glob('*/evaluation_complete.json')):
        receipts.append(file_sha256(receipt))
        folder=receipt.parent/'inference'
        with np.load(folder/'face_waveforms.npz',allow_pickle=False) as waves,np.load(folder/'feature_validity.npz',allow_pickle=False) as valid:
            eligible=valid['60Hz__candidate_cuff_free']&valid['60Hz__face_quality_good']
            for key in waves.files:
                if not key.startswith('60Hz__'):continue
                values=waves[key];usable=eligible&np.isfinite(values).all(1)&(np.std(values,axis=1)>1e-8)
                rows=[]
                for i in np.flatnonzero(usable):
                    frequency,power=periodogram(values[i],60,window='hann',detrend='linear')
                    total=power[(frequency>=.5)&(frequency<=29.5)].sum()
                    middle=power[(frequency>=8)&(frequency<15)].sum()/total
                    high=power[(frequency>=15)&(frequency<=29.5)].sum()/total
                    rows.append([middle,high])
                private.append(dict(session_id=receipt.parent.name,model=key,attempted=len(values),eligible=int(eligible.sum()),
                    usable=int(usable.sum()),fractions_8_to_15_and_15_to_29_5Hz=rows))
                if rows:groups[key].append(np.median(rows,axis=0))
    summary=[]
    for model,person in sorted(groups.items()):
        summary.append(dict(model=model,subjects=len(person),subject_median_band_power_fraction=np.median(person,axis=0),
            subject_IQR_band_power_fraction=np.percentile(person,[25,75],axis=0)))
    result=dict(scope='Label-free complete native60 processing path; no matched high-rate contact target',
        frequency_bands_hz=[[8,15],[15,29.5]],normalization_band_hz=[.5,29.5],window='Hann with linear detrending',
        model_summaries=summary,higher_power_is_not_improved_waveform_accuracy=True,
        source_sha256=file_sha256(__file__),input_receipt_sha256=sorted(receipts),completed_recordings=len(receipts))
    out=root/'native_output_audit';out.mkdir(exist_ok=True)
    write_json(out/'private_rows.json',private);write_json(out/'aggregate.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    print('Native bandwidth audit complete:',len(audit(Path(args.run))['model_summaries']),'fixed waveform groups')

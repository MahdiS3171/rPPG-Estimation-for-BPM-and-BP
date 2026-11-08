# Native 60-fps waveform direction

Updated 2026-11-08 after the user's explicit steering: use additional available
online waveform data and develop a model that can retain the 60-fps recordings'
information. Do not assume camera angle is the principal cause of failure.
Repeat acquisition only when a specific comparison can resolve a known issue.

## What the existing code actually does

Frozen V1 expects 300 samples at 30 Hz, covering 10 seconds. Its reviewed external
adapter regularizes native timestamps at 60 Hz, applies an antialias FIR (8-Hz
passband, 15-Hz stopband), then supplies the unchanged 30-Hz API. This is a
deliberate compatibility path, not a native-60 waveform model.

The six classical prior channels apply their original approximately 0.7–4-Hz
processing before per-window normalization. Raw RGB channels also reach the net,
so the entire V1 input is not restricted to 4 Hz. Nevertheless, increasing FPS
alone while keeping the same filtered priors does not restore discarded harmonics.
Changing a sample-rate label or returning an interpolated 60-Hz output is also
insufficient; learned temporal receptive fields and targets need the correct clock.

## Executed transfer audit

Command: `python -m scripts.audit_waveform_sample_rate --out
outputs/waveform_sample_rate/20261108_01`. Seven unit-amplitude sinusoidal inputs
were generated on a regular 60-Hz clock. The existing adapter was used unchanged;
no checkpoint or participant data was used. Gains were fitted before normalization
on retained seconds [5,25). A complete receipt is in
[sample_rate_results_20261108.json](project_audit/sample_rate_results_20261108.json).

| Input frequency | 30-Hz adapter RGB amplitude gain | Default GREEN amplitude gain even at 60 Hz |
|---|---:|---:|
| 1.2 Hz | 1.0001 | 0.9983 |
| 3.5 Hz | 1.0001 | 0.7738 |
| 8 Hz | 1.0001 | 0.0048 |
| 12 Hz | 0.3427 | 0.00022 |
| 18 Hz | 0.000043 at alias frequency | 0.0000045 |

Thus the clean low-frequency signal is preserved by the adapter, while some
higher-frequency content is intentionally lost. This supports investigating
native-60 processing, but does **not** show those frequencies contain recoverable
physiology in our phone videos. Higher FPS trades exposure/photon budget against
sample density. Numerical Nyquist bandwidth and real pulse-detail SNR differ.

## Candidate and decisive comparison

Keep V1 unchanged. The next V2 candidate should take and return 600 samples for
the same 10-second duration at 60 Hz, preserving original PTS provenance. Start
with a compact temporal model and raw RGB plus explicitly characterized optional
priors; avoid making a 4-Hz channel bank the only information source. Express
filter cutoffs and receptive fields in seconds/Hz, not unexamined sample counts.
At the time of this initial strategy, no candidate had been trained. The later
[development campaign](WAVEFORM_DEVELOPMENT_CAMPAIGN_REPORT.md) now supplies a
30-Hz-trained TCN candidate with 60-Hz input support and equivalent-grid checks.
It does not yet supply native-60 training/reference evidence.

Use real native-60-or-higher RGB with a sufficiently sampled waveform reference.
Lower-FPS datasets can contribute domain diversity and rate augmentation; their
upsampled labels cannot validate newly invented high-frequency detail. Keep
uncertainty about reference filtering and anatomical site explicit. Do not use
a broad lag-tolerant loss alone as proof of temporal fidelity.

The key experiment is paired: take the **same native-60 recordings**, compare
native-60 input/output against properly antialiased 30-Hz input/output at matched
duration, ROIs, participants and reference. Include fixed classical baselines and
frozen V1; distinguish sampling benefit from architecture and training-set size.
Evaluate native waveform shape/derivatives, beat precision and failure coverage
as well as HR. A bigger model trained on different subjects is not a clean FPS
ablation. Determine the justified signal band from development reference quality,
then lock it before held-out evaluation.

## Data work already begun

Following the user's online-data instruction, one small official
[MPU-rPPG sample](https://figshare.com/articles/dataset/MPU-rPPG_Sample_Dataset/29377835)
video and a candidate CSV were downloaded under the posted CC BY 4.0 licence
and research-use notice. The 106.82-MB MP4 contains 34,496 samples on a uniform
container 60-Hz grid (~575 s); the CSV has the same row count and Count=0..34495,
with PPG/HR/SpO2 columns. The repeated source filenames provide no explicit
pairing manifest; sensor timing, identity and reference processing remain
unverified. Matching counts are structural evidence, not alignment proof.
All 34,496 frames were subsequently decoded for a hash-only duplicate screen:
zero exact adjacent duplicates. This excludes simple exact frame repetition in
that file, not all interpolation or unknown camera processing. Container 60 fps
is still not certified native sensor timing. These files are development-only and cannot support independent efficacy or
generalization claims. Receipts record file IDs, byte counts and verified hashes.

[Tricam's official project page](https://tricam-rppg.github.io/) describes 60-fps
RGB, 125-Hz fingertip PPG, and face/raised-hand tasks. It requires an executed
data-use licence; distribution size and exact timing must be confirmed.
[VicarPPG-2's original paper](https://doi.org/10.3390/app10238630) is another
native-60 lead with physiological references; its
[official licence](https://d.sentient.nl/download/files/VV-DatasetRequests/VicarPPG2_LicenseAgreement.pdf)
requires research use and supervisor involvement. Prefer small RGB/reference
subsets rather than entire multimodal archives. PURE remains useful for real
domain diversity even though it does not solve native-60 training by itself.

Next: obtain explicit sample correspondence/clock documentation, then admit a
development subset and fix subject grouping. Training follows that evidence and
a bounded compute proposal. No new pilot videos, unblinded reserved sessions or
large downloads are needed to complete these admission steps.

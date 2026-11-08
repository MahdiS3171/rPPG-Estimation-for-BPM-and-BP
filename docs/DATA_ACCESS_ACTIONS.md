# Focused data access actions

Prepared 2026-11-08. Drafts for the researcher/supervisor; **not sent**. Codex
has not signed an agreement or contacted another person. Existing PURE request
is already pending; do not send a duplicate by default.

## Immediate development: MPU sample clarification

The public sample is already downloaded within the user's online-data scope;
no additional application was required for the posted local research use.
Before training, obtain documented video/CSV correspondence, reference sampling/
filtering and participant identity. Equal counts do not resolve these points.
The linked code repository currently exposes README, LICENSE and a plan, rather
than the acquisition/training implementation described in the paper's code
availability statement. Do not claim that code was reproduced.

Draft to the dataset contact listed in the
[official Figshare record](https://figshare.com/articles/dataset/MPU-rPPG_Sample_Dataset/29377835):

> We are studying camera-based waveform reconstruction using 60-fps recordings.
> We obtained your public MPU-rPPG sample, citing the dataset and associated paper.
> Could you confirm whether file 55553429 (Output.mp4) corresponds to file 55553426
> (Output.csv), and provide the participant/session mapping? Both contain 34,496
> samples. Is Count a frame index or a reference-device counter? Please clarify
> the original PPG rate, any resampling/filtering, capture-to-reference clock
> offset/latency, and where the acquisition code or synchronization documentation
> can be obtained. We seek a small, documented development subset rather than
> the full archive.

## Recommended additional native-60 application: Tricam

Its [official page](https://tricam-rppg.github.io/) describes the relevant RGB/
raised-hand/reference modalities and requires a DUL. Access and storage remain
unconfirmed. Recommendation: ask for the licence and the smallest useful RGB+
PPG subset before considering NIR or large archives. This is an institutional
data-access decision; no spending or new hardware is assumed.

Draft to the two contacts on that official page:

> Our research project investigates temporally faithful RGB rPPG waveform
> reconstruction and exploratory simultaneous face–hand optical timing. We are
> interested in a small 60-fps RGB subset with the corresponding PPG, particularly
> resting and raised-hand tasks. Could you provide the Data Use License, available
> subset sizes, participant/session metadata, original frame/reference timestamp
> format, synchronization/latency documentation, reference filtering information
> and permitted processing/publication scope? We would complete the institutional
> approval process before accessing controlled data.

Alternative: [VicarPPG-2](https://d.sentient.nl/download/files/VV-DatasetRequests/VicarPPG2_LicenseAgreement.pdf)
for native-60 facial waveform work. MSPM is a later multisite-reference option;
its broader institutional signing requirements should be weighed against the
immediate waveform priority. Do not open multiple large access/download campaigns
without a specific information need and resource estimate.

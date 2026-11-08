# SCAMPS source attribution and snapshot provenance

SCAMPS: Synthetics for Camera Measurement of Physiological Signals.
Daniel McDuff, Miah Wander, Xin Liu, Brian L. Hill, Javier Hernandez,
Jonathan Lester, and Tadas Baltrusaitis (2022).

- [Official repository](https://github.com/danmcduff/scampsdataset)
- [Paper](https://arxiv.org/abs/2206.04197)
- [Pinned documentation revision](https://github.com/danmcduff/scampsdataset/tree/0d3e1fd9915ea52a66af87caa1508b292555f765)
- [Official datasheet](https://github.com/danmcduff/scampsdataset/blob/0d3e1fd9915ea52a66af87caa1508b292555f765/Datasheet%20-%20Synthetics%20for%20Camera%20Measurement%20of%20Physiological%20Signals.pdf)

The SCAMPS dataset and the vendored public split metadata are subject to the
accompanying [Research Use of Data Agreement v1.0](LICENSE_SCAMPS.txt).
Recipients must follow those terms, including non-commercial computational
research use and redistribution attribution/terms. The license restricts use of
Data and Results in commercial offerings. This is not a replacement project-wide
license for independently authored adapter code. No SCAMPS frames, videos, raw
MATLAB files or derived image tensors are distributed in this Git repository.

`official_split.csv` is the public 2,800-record split artifact linked from the
official README; it contains recording IDs, source URLs and Train/Val/Test names.
Its malformed two-field header is retained byte-for-byte. The parser verifies
the pinned SHA-256 and handles its three-field rows explicitly. The URLs in that
CSV are provenance only; no data is fetched by parsing it.

`official_sources.json` records source URLs, file sizes and SHA-256 checksums for
the public documentation/license/split snapshots inspected on 2026-11-05. The
README and PDF themselves remain in local ignored diagnostic output; only their
provenance is stored here. The local `.mat` source checksums identify the inspected
user-supplied example files; no publisher checksum manifest was available to
independently certify those bytes as a specific dataset release.

The admission JSON files contain metadata, finite/range summaries and engineering
observations only. They are not model performance reports. All ten sample files,
including the official Val/Test examples, are marked engineering-inspected and
must not be promoted to an independent generalization benchmark.

```bibtex
@article{mcduff2022scamps,
  title={SCAMPS: Synthetics for Camera Measurement of Physiological Signals},
  author={McDuff, Daniel and Wander, Miah and Liu, Xin and Hill, Brian L and
          Hernandez, Javier and Lester, Jonathan and Baltrusaitis, Tadas},
  journal={arXiv preprint arXiv:2206.04197},
  year={2022}
}
```

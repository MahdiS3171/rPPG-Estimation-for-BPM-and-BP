"""Reviewed rPPG/BP research codebase.

Main modules
------------
- ``classical``: transparent traditional rPPG baselines and priors.
- ``roi``: backwards-compatible facial ROI extraction.
- ``pipeline`` / ``detection`` / ``region_masks``: synchronized face/hand extraction.
- ``types`` / ``config`` / ``video`` / ``processing`` / ``quality``: explicit clocks and quality.
- ``bp`` / ``study`` / ``splits`` / ``bp_models``: pilot features, provenance and baselines.
- ``signals``: filtering, HR estimation, inter-site delay, and waveform features.
- ``datasets``: UBFC and future session-level dataset helpers.
- ``models``: waveform, HR/quality, video, multi-ROI, and BP baselines.
- ``losses`` / ``metrics``: reproducible training and evaluation helpers.
"""

__version__ = "0.2.0"

"""Clean rPPG/BP research codebase.

Main modules:
- classical: traditional rPPG methods (GREEN, CHROM, POS, PBV, LGI, OMIT, ICA, PCA)
- roi: MediaPipe-based multi-ROI extraction from facial/hand videos
- signals: filtering, HR estimation, PTT/phase features, waveform utilities
- datasets: UBFC loaders and PyTorch datasets
- models: waveform-first HR models and BP feature baselines
- losses/metrics: reproducible training and evaluation helpers
"""

__version__ = "0.1.0"

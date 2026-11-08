"""Predetermined native-axis Gate 09 plots; display scaling never changes scoring."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .signals import standardize_1d
from .waveform_v2_classical_study import METHODS, CONDITIONS


def display(x):
    return standardize_1d(x) if np.isfinite(x).all() and np.std(x) > 1e-10 else x


def make_plots(root, records, outputs, references, rows, summaries, controls, control_outputs):
    root = Path(root); root.mkdir(exist_ok=False)
    for filename in ('P000001.mat', 'P000003.mat'):
        record = next(r for r in records if r['filename'] == filename and r['start'] == 0)
        key = (filename, 0); t = record['indices']/30
        fig, axes = plt.subplots(6, 4, figsize=(18, 14), sharex=True, sharey=True)
        for k, method in enumerate(METHODS):
            for r, roi in enumerate(CONDITIONS):
                ax = axes[k, r]
                ax.plot(t, display(references[key]), color='black', lw=1, alpha=.65, label='raw PPG (display z)')
                ax.plot(t, display(record['classical'][r, k]), color='#dc7a22', lw=.8, label='fixed classical (display z)')
                ax.plot(t, display(outputs[key]['waveform']), color='#2355b5', lw=.9, label='frozen V1 (display z)')
                ax.set_title(f'{roi} / {method}', fontsize=9)
                ax.set_ylim(-4, 4); ax.grid(alpha=.15)
        fig.suptitle(f'{filename} first original window: native polarity, zero shift; display-only scaling', y=.997)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .976), ncol=3)
        fig.supxlabel('Original nominal index / 30 (s)'); fig.supylabel('Window z score for display')
        fig.tight_layout(rect=(.02, .02, 1, .95)); fig.savefig(root/f'fixed_baselines_{Path(filename).stem}.png', dpi=125)
        plt.close(fig)
    for filename in dict.fromkeys(r['filename'] for r in records):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        for ax, record in zip(axes, [r for r in records if r['filename'] == filename]):
            key = (filename, record['start']); t = record['indices']/30
            ax.plot(t, display(references[key]), color='black', lw=1, label='raw PPG (display z)')
            ax.plot(t, display(outputs[key]['waveform']), color='#2355b5', lw=1, label='V1 (display z)')
            metric = next(r['metrics'] for r in rows if (r['filename'], r['start'], r['condition']) == (*key, 'frozen_v1'))
            ax.set_title(f"{record['split']} [{record['start']},{record['stop']}): native r={metric['zero_lag_r']:.3f}; "
                f"HR error={metric['hr_error_bpm']:+.1f} bpm; diagnostic lag={metric['selected_lag_ms']:.0f} ms")
            ax.grid(alpha=.2); ax.legend(loc='upper right')
        fig.suptitle(f'{filename}: independent windows, native polarity and original axes')
        axes[-1].set_xlabel('Original nominal index / 30 (s)')
        fig.tight_layout(); fig.savefig(root/f'native_{Path(filename).stem}.png', dpi=130); plt.close(fig)
    neural = [r for r in rows if r['condition'] == 'frozen_v1']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for split, color, marker in (('Train', '#2355b5', 'o'), ('Val', '#db8126', 's'), ('Test', '#b8335b', '^')):
        group = [r for r in neural if r['split'] == split]
        for ax, xkey, ykey in ((axes[0, 0], 'reference_hr_bpm', 'pred_hr_bpm'),
                (axes[0, 1], 'zero_lag_r', 'lag_aligned_r'),
                (axes[1, 0], 'reference_hr_bpm', 'hr_error_bpm'),
                (axes[1, 1], 'zero_lag_r', 'selected_lag_ms')):
            ax.scatter([r['metrics'][xkey] for r in group], [r['metrics'][ykey] for r in group],
                       color=color, marker=marker, label=f'{split} ({len(group)} windows)')
            ax.set_xlabel(xkey); ax.set_ylabel(ykey); ax.grid(alpha=.2)
    axes[0, 0].plot([42, 210], [42, 210], 'k--', lw=.8)
    axes[0, 1].plot([-1, 1], [-1, 1], 'k--', lw=.8)
    axes[1, 0].axhline(0, color='black', lw=.8)
    axes[0, 0].legend(); fig.suptitle('All 20 intended windows; correlated within ten exposed recordings')
    fig.tight_layout(); fig.savefig(root/'hr_native_lag_diagnostics.png', dpi=140); plt.close(fig)
    train = [r for r in summaries if r['split'] == 'Train']
    fig, axes = plt.subplots(1, 3, figsize=(14, 10), sharey=True)
    labels = [r['condition'] for r in train]; y = np.arange(len(train))
    for ax, metric, title in zip(axes, ('hr_abs_error_bpm', 'zero_lag_r', 'lag_aligned_r'),
                               ('HR MAE (bpm)', 'Native signed r', 'Lag-optimized diagnostic r')):
        ax.barh(y, [r['recording_balanced'][metric] for r in train], color=['#2355b5']+['#8097a7']*(len(train)-1))
        ax.set_title(title); ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(y, labels, fontsize=8); axes[0].invert_yaxis()
    fig.suptitle('Train only: eight recording-balanced means, every fixed condition')
    fig.tight_layout(); fig.savefig(root/'complete_train_comparison.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for ax, (name, predictions) in zip(axes.flat, control_outputs.items()):
        for w, signal in enumerate(predictions):
            ax.plot(np.arange(w*300, (w+1)*300)/30, display(signal), lw=.8)
        group = [r for r in controls if r['filename'] == name]
        ax.set_title(name+' | eligible '+str(sum(r['pulse_estimate_eligible'] for r in group))+'/2', fontsize=10)
        ax.grid(alpha=.2)
    fig.suptitle('Manufactured controls: display z scores; periodic output is not physiological validation')
    fig.supxlabel('Original nominal time (s)'); fig.tight_layout()
    fig.savefig(root/'negative_controls.png', dpi=135); plt.close(fig)

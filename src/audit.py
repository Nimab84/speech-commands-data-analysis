"""Phase 1: inspect every WAV, register speaker splits, and export measured EDA.

No resampling, normalization, filtering, model fitting, or augmentation is done.
Full-corpus metadata/quality are descriptive. Relationship plots use train only.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
import wave
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'tmp/matplotlib'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#218c91', '#e2a14b', '#697ec4', '#d37780', '#57a773', '#aa85b3', '#9e7e61', '#597990']


def assign_split(speaker, validation=10, test=10):
    """TensorFlow which_set rule, independently expressed for speaker/session ID."""
    maximum = 2**27 - 1
    h = int(hashlib.sha1(speaker.encode('utf-8')).hexdigest(), 16)
    percent = (h % (maximum + 1)) * (100.0 / maximum)
    return 'validation' if percent < validation else 'test' if percent < validation + test else 'train'


def load_wav(path):
    with wave.open(str(path), 'rb') as f:
        params = f.getparams()
        raw = f.readframes(params.nframes)
    if params.sampwidth != 2 or params.comptype != 'NONE':
        raise ValueError('Only uncompressed 16-bit PCM is supported; flagged for review.')
    if len(raw) != params.nframes * params.nchannels * params.sampwidth:
        raise ValueError('Truncated PCM payload')
    pcm = np.frombuffer(raw, dtype='<i2')
    if params.nchannels < 1 or pcm.size == 0 or params.framerate <= 0:
        raise ValueError('Empty signal or invalid WAV header')
    return params, raw, pcm, pcm.astype(np.float64).reshape(-1, params.nchannels).mean(axis=1) / 32768.0


def inspect(path, data_dir, cfg):
    row = {'path': path.relative_to(data_dir).as_posix(), 'label': path.parent.name,
           'file_bytes': path.stat().st_size,
           'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    match = re.fullmatch(r'([0-9a-f]{8})_nohash_(\d+)\.wav', path.name)
    row.update(speaker_id=match[1] if match else None,
               utterance_index=int(match[2]) if match else None,
               split=assign_split(match[1], cfg['validation_percent'], cfg['test_percent']) if match else 'unassigned')
    try:
        p, raw, pcm, x = load_wav(path)
        rms = float(np.sqrt(np.mean(x*x)))
        n = max(1, int(p.framerate * cfg['frame_ms'] / 1000))
        frames = np.array_split(x, np.arange(n, len(x), n))
        frame_rms = np.array([np.sqrt(np.mean(f*f)) for f in frames])
        row.update(status='ok', error='', sample_rate_hz=p.framerate, channels=p.nchannels,
                   sample_width_bytes=p.sampwidth, n_frames=p.nframes,
                   duration_s=p.nframes / p.framerate,
                   pcm_sha256=hashlib.sha256(raw).hexdigest(),
                   signal_min=float(x.min()), signal_max=float(x.max()),
                   peak_abs=float(np.max(np.abs(x))), rms=rms,
                   rms_dbfs=float(20*np.log10(max(rms, 1e-12))),
                   dc_offset=float(x.mean()), zero_fraction=float(np.mean(pcm == 0)),
                   saturated_sample_count=int(np.count_nonzero((pcm == -32768) | (pcm == 32767))),
                   saturation_fraction=float(np.mean((pcm == -32768) | (pcm == 32767))),
                   low_energy_frame_fraction=float(np.mean(20*np.log10(np.maximum(frame_rms, 1e-12)) < cfg['frame_low_energy_dbfs'])),
                   zero_crossing_rate=float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if x.size > 1 else 0.0,
                   exact_silence=bool(np.all(pcm == 0)),
                   low_energy=bool(20*np.log10(max(rms, 1e-12)) < cfg['low_energy_dbfs']),
                   very_short=bool(p.nframes / p.framerate < cfg['short_seconds']),
                   shorter_than_one_s=bool(p.nframes < p.framerate),
                   longer_than_one_s=bool(p.nframes > p.framerate),
                   unexpected_format=bool(p.framerate != cfg['sample_rate_hz'] or p.nchannels != 1),
                   high_dc=bool(abs(x.mean()) > cfg['dc_offset_abs']))
    except (wave.Error, EOFError, ValueError, OSError) as error:
        row.update(status='error', error=f'{type(error).__name__}: {error}')
    return row


def savefig(fig, name, figdir):
    fig.savefig(figdir / f'{name}.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_class_counts(counts, figdir):
    with plt.rc_context({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.titleweight':'bold'}):
        ax = counts.plot.bar(stacked=True, figsize=(9.0,3.4),
                             color=['#218c91','#e2a14b','#697ec4'], rot=0)
        ax.set(ylabel='Number of recordings', xlabel='Command')
        ax.set_title('Class counts by speaker split', pad=34)
        ax.legend(frameon=False, ncol=3, loc='lower center',
                  bbox_to_anchor=(0.5,1.02), borderaxespad=0,
                  fontsize=9, handlelength=1.8, columnspacing=1.8)
        savefig(ax.figure,'class_split_counts',figdir)


def plot_class_relationships(train, labels, out, figdir):
    """Show short-clip rates explicitly; one-second medians hide this variation."""
    short = (train.n_frames < train.sample_rate_hz).groupby(train.label).agg(['sum', 'size']).reindex(labels)
    short.columns = ['short_clips', 'train_clips']
    short['percent_short'] = 100 * short.short_clips / short.train_clips
    short.to_csv(out / 'train_short_clip_rates.csv', index_label='label')
    with plt.rc_context({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.titleweight':'bold'}):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
        bars = axes[0].bar(labels, short.percent_short, color=COLORS, alpha=.8)
        axes[0].bar_label(bars, labels=[f'{value:.1f}%' for value in short.percent_short],
                          padding=3, fontsize=8)
        ceiling = max(5, np.ceil(short.percent_short.max()/5)*5+5)
        axes[0].set(title='Clips shorter than 1 s', ylabel='Within-class training clips (%)',
                    xlabel='Command', ylim=(0, ceiling))
        for ax, column, title in zip(axes[1:], ['rms_dbfs', 'low_energy_frame_fraction'],
                                     ['RMS energy (dBFS)', 'Low-energy frame fraction']):
            boxes = ax.boxplot([train.loc[train.label == label, column] for label in labels],
                               tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, color in zip(boxes['boxes'], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(.75)
            ax.set(title=title, xlabel='Command')
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        savefig(fig, 'train_class_relationships', figdir)


def plots(df, cfg, out, figdir):
    labels = cfg['labels']
    train = df[(df.status == 'ok') & (df.split == 'train')]
    good = df[df.status == 'ok']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.dpi': 120})
    counts = pd.crosstab(df.label, df.split).reindex(labels)[['train', 'validation', 'test']]
    plot_class_counts(counts,figdir)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, column, title, xlabel in zip(axes,
            ['duration_s', 'rms_dbfs', 'peak_abs'],
            ['Duration', 'RMS energy', 'Peak amplitude'],
            ['Seconds', 'dBFS', 'Normalized absolute amplitude']):
        ax.hist(train[column], bins=35, color='#218c91', edgecolor='white', linewidth=0.4)
        ax.set(title=title, xlabel=xlabel, ylabel='Train recordings')
    savefig(fig, 'train_distributions', figdir)
    plot_class_relationships(train, labels, out, figdir)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    speakers = good.groupby('speaker_id').agg(recordings=('path','size'), labels=('label','nunique'))
    axes[0].hist(speakers.recordings, bins=30, color='#218c91', edgecolor='white')
    axes[0].set(xlabel='Recordings per speaker/session', ylabel='Number of IDs', title='Unequal speaker contribution: full corpus')
    vc = speakers.labels.value_counts().sort_index()
    axes[1].bar(vc.index, vc.values, color='#697ec4')
    axes[1].set(xlabel='Distinct commands per speaker/session', ylabel='Number of IDs', title='Class coverage per speaker/session')
    axes[1].set_xticks(range(1, 9))
    savefig(fig, 'speaker_coverage', figdir)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for label, color in zip(labels, COLORS):
        x = train[train.label == label]
        axes[0].scatter(x.duration_s, x.rms_dbfs, s=6, alpha=.3, label=label, color=color)
    axes[0].set(xlabel='Duration (s)', ylabel='RMS energy (dBFS)', title='Duration vs energy: train only')
    axes[0].legend(ncol=2, frameon=False, markerscale=2, fontsize=8)
    columns = ['duration_s', 'rms_dbfs', 'peak_abs', 'low_energy_frame_fraction', 'zero_crossing_rate']
    # Pearson correlation of average ranks equals Spearman, including ties.
    corr = train[columns].rank(method='average').corr(method='pearson')
    im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    short = ['duration', 'RMS dBFS', 'peak', 'quiet frames', 'ZCR']
    axes[1].set_xticks(range(5), short, rotation=40, ha='right')
    axes[1].set_yticks(range(5), short)
    axes[1].set_title('Spearman correlation: train only')
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8,
                         color='white' if abs(corr.iloc[i,j]) > .6 else '#243743')
    fig.colorbar(im, ax=axes[1], fraction=.045)
    savefig(fig, 'train_multivariate', figdir)
    corr.to_csv(out / 'train_spearman.csv')
    # Representatives are selected by a fixed hash on path, from train only.
    selected = []
    for label in labels:
        subset = train[train.label == label].copy()
        subset['selection_key'] = subset.path.map(lambda p: hashlib.sha256(f"{cfg['seed']}:{p}".encode()).hexdigest())
        subset = subset.sort_values('selection_key').head(cfg['spectrogram_examples_per_class'])
        fig, axes = plt.subplots(3, 2, figsize=(9.5, 6.7), constrained_layout=True,
                                 gridspec_kw={'width_ratios': [1, 1.25]})
        for (idx, row), pair in zip(subset.iterrows(), axes):
            p, _, _, x = load_wav(ROOT / 'data/raw/mini_speech_commands' / row.path)
            pair[0].plot(np.arange(len(x))/p.framerate, x, color='#218c91', lw=.65)
            pair[0].set(xlim=(0,1), ylim=(-1,1), ylabel='Amplitude', title=Path(row.path).name)
            window, hop, nfft = (cfg[k] for k in ['spectrogram_window','spectrogram_hop','spectrogram_nfft'])
            xp = np.pad(x, (0, max(0, window-len(x))))
            frames = np.lib.stride_tricks.sliding_window_view(xp, window)[::hop]
            power = np.abs(np.fft.rfft(frames*np.hanning(window), n=nfft, axis=1))**2 / np.hanning(window).sum()**2
            db = 10*np.log10(np.maximum(power, 1e-10))
            spectral_image = pair[1].imshow(db.T, origin='lower', aspect='auto', cmap='magma', vmin=-100, vmax=0,
                           extent=[window/(2*p.framerate), (window/2+hop*(len(frames)-1))/p.framerate, 0, p.framerate/2000])
            pair[1].set(xlim=(0,1), ylabel='Frequency (kHz)', title=f'{label} | STFT power')
            selected.append({'label':label, 'path':row.path, 'speaker_id':row.speaker_id,
                             'split':row.split, 'rms_dbfs':row.rms_dbfs})
        axes[-1,0].set_xlabel('Time (s)')
        axes[-1,1].set_xlabel('Time (s)')
        fig.colorbar(spectral_image, ax=axes[:,1].tolist(), label='Power (dB re. full scale)', fraction=.03, pad=.02)
        savefig(fig, f'examples_{label}', figdir)
    pd.DataFrame(selected).to_csv(out / 'example_selection.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/phase1.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['eda_split'] != 'train':
        raise ValueError('Comparative phase-1 EDA must use train only.')
    data_dir = ROOT / 'data/raw/mini_speech_commands'
    if not data_dir.is_dir():
        raise SystemExit('Dataset missing. Run src/download_data.py first.')
    out = ROOT / 'reports/phase1'
    figdir = ROOT / 'figures/phase1'
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    paths = sorted(data_dir.rglob('*.wav'))
    background_paths = [p for p in paths if p.parent.name == '_background_noise_']
    command_paths = [p for p in paths if p.parent.name in cfg['labels']]
    unexpected_labels = sorted({p.parent.name for p in paths} - set(cfg['labels']) - {'_background_noise_'})
    rows = []
    for i, path in enumerate(command_paths, 1):
        rows.append(inspect(path, data_dir, cfg))
        if i % 1000 == 0:
            print(f'Audited {i}/{len(command_paths)} WAV files', flush=True)
    if not rows:
        raise ValueError('No command WAV files found')
    df = pd.DataFrame(rows)
    df.to_csv(out / 'audio_inventory.csv', index=False)
    good = df[df.status == 'ok'].copy()
    train = good[good.split == 'train']
    assert not df.speaker_id.isna().any(), 'Unparseable speaker IDs; split cannot be trusted'
    assert not unexpected_labels, f'Unexpected labels: {unexpected_labels}'
    assert set(df.label) == set(cfg['labels'])
    assert good.groupby('speaker_id').split.nunique().max() == 1
    counts = pd.crosstab(df.label, df.split).reindex(cfg['labels'])[['train','validation','test']]
    assert (counts > 0).all().all(), 'A class is absent from a split'
    counts.to_csv(out / 'class_split_counts.csv')
    speaker_classes = pd.crosstab(good.speaker_id, good.label).reindex(columns=cfg['labels'], fill_value=0)
    speaker_classes.to_csv(out / 'speaker_class_counts.csv')
    speakers = good.groupby('speaker_id').agg(n_recordings=('path','size'), n_labels=('label','nunique'), split=('split','first'))
    speakers.to_csv(out / 'speakers.csv')
    duplicates = good[good.duplicated('pcm_sha256', keep=False)]
    duplicates.to_csv(out / 'duplicate_records.csv', index=False)
    cross = duplicates.groupby('pcm_sha256').split.nunique()
    cross_count = int((cross > 1).sum())
    if cross_count:
        raise ValueError('PCM duplicates cross splits. Review the exported records before modeling.')
    flagged_columns = ['very_short','low_energy','exact_silence','longer_than_one_s','unexpected_format','high_dc']
    flagged = good[good[flagged_columns].any(axis=1) | (good.saturated_sample_count > 0)]
    pd.concat([df[df.status != 'ok'], flagged]).to_csv(out / 'quality_review_queue.csv', index=False)
    quality_counts = {c:int(good[c].sum()) for c in flagged_columns + ['shorter_than_one_s']}
    quality_counts.update(corrupt_files=int((df.status != 'ok').sum()),
                          saturated_files=int((good.saturated_sample_count > 0).sum()),
                          duplicate_groups=int(duplicates.pcm_sha256.nunique()),
                          duplicate_extra_records=int(good.duplicated('pcm_sha256').sum()),
                          missing_speaker_id=int(df.speaker_id.isna().sum()),
                          missing_label=int(df.label.isna().sum()))
    quality_counts['duplicate_label_conflict_groups'] = int((duplicates.groupby('pcm_sha256').label.nunique() > 1).sum())
    quality_counts['duplicate_speaker_conflict_groups'] = int((duplicates.groupby('pcm_sha256').speaker_id.nunique() > 1).sum())
    numeric = ['duration_s','rms_dbfs','peak_abs','zero_fraction','low_energy_frame_fraction','dc_offset','zero_crossing_rate']
    stats = good[numeric].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    stats.to_csv(out / 'numeric_summary_all.csv')
    train[numeric].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T.to_csv(out / 'numeric_summary_train.csv')
    train.groupby('label')[numeric].agg(['count','mean','std','median','min','max']).to_csv(out / 'class_summary_train.csv')
    # Demonstrate why an ordinary shuffled file split leaks known speaker/session IDs.
    rng = np.random.default_rng(cfg['seed'])
    order = rng.permutation(len(good))
    n = len(good)
    naive_train = set(good.iloc[order[:int(.8*n)]].speaker_id)
    naive_val = good.iloc[order[int(.8*n):int(.9*n)]]
    naive_test = good.iloc[order[int(.9*n):]]
    naive = {'validation_seen_speaker_fraction':float(naive_val.speaker_id.isin(naive_train).mean()),
             'test_seen_speaker_fraction':float(naive_test.speaker_id.isin(naive_train).mean()),
             'validation_overlapping_ids':len(set(naive_val.speaker_id)&naive_train),
             'test_overlapping_ids':len(set(naive_test.speaker_id)&naive_train)}
    classes = cfg['labels']
    majority = sorted(train.label.value_counts().items(), key=lambda item:(-item[1],item[0]))[0][0]
    validation = good[good.split == 'validation']
    accuracy = float((validation.label == majority).mean())
    prevalence = float((validation.label == majority).mean())
    majority_macro_f1 = (2*prevalence/(1+prevalence))/len(classes)
    baseline = {'kind':'constant training-majority class; lexical tie-break', 'prediction':majority,
                'training_count':int((train.label == majority).sum()),
                'validation_accuracy':accuracy, 'validation_macro_f1':majority_macro_f1,
                'validation_balanced_accuracy':1/len(classes),
                'test_scores_computed':False, 'uniform_random_expected_accuracy':1/len(classes)}
    split_info = {s:{'recordings':int((good.split == s).sum()),
                    'speakers':int(good[good.split == s].speaker_id.nunique())} for s in ['train','validation','test']}
    summary = {'n_files':len(df), 'n_valid':len(good), 'n_labels':int(df.label.nunique()),
               'n_speakers':int(good.speaker_id.nunique()), 'labels':classes,
               'sample_rates_hz':{str(int(k)):int(v) for k,v in good.sample_rate_hz.value_counts().items()},
               'channels':{str(int(k)):int(v) for k,v in good.channels.value_counts().items()},
               'total_duration_minutes':float(good.duration_s.sum()/60),
               'quality':quality_counts, 'splits':split_info, 'speaker_overlap_count':0,
               'cross_split_duplicate_groups':cross_count,
               'background_noise_files':len(background_paths),
               'provided_silence_or_unknown_labels':False,
               'official_split_list_files':{p:(data_dir/p).exists() for p in ['validation_list.txt','testing_list.txt']},
               'naive_row_split':naive, 'baseline':baseline,
               'speaker_recording_quantiles':{str(k):float(v) for k,v in speakers.n_recordings.quantile([0,.25,.5,.75,.95,1]).items()},
               'speaker_label_counts':{str(k):int(v) for k,v in speakers.n_labels.value_counts().sort_index().items()},
               'train_class_medians':train.groupby('label')[numeric].median().round(6).to_dict(orient='index'),
               'thresholds':{k:cfg[k] for k in ['short_seconds','low_energy_dbfs','frame_low_energy_dbfs','frame_ms','dc_offset_abs']},
               'eda_split':cfg['eda_split'], 'seed':cfg['seed'],
               'dataset_sha256':json.loads((ROOT/'data/source_manifest.json').read_text())['archive_sha256'],
               'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest()}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    plots(df, cfg, out, figdir)
    deps = ['numpy','pandas','matplotlib','reportlab','Pillow','pypdf']
    environment = {'python':sys.version, 'platform':platform.platform(),
                   'packages':{p:version(p) for p in deps}}
    build = ROOT/'tmp/phase1_build'
    build.mkdir(parents=True,exist_ok=True)
    (build / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Publication figures from measured Phase 2 outputs; training data unless marked."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from audio_features import read_audio,prepare_signal,extract_features,augment_train
from run_phase2 import locate_raw,ordered_sample

FIG = ROOT/'figures'
TABLES = ROOT/'reports/tables'
COLORS = ['#218c91','#e2a14b','#697ec4','#d37780','#57a773','#aa85b3','#9e7e61','#597990',
          '#56a9b5','#b08458','#916db2','#516b8b']


def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9.5,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.labelcolor':'#263849','text.color':'#16364d'})
    cfg = json.loads((ROOT/'configs/phase2.json').read_text())
    summary = json.loads((ROOT/'reports/summary.json').read_text())
    retained = pd.read_csv(ROOT/'data/processed/retained_clips.csv')
    inventory = pd.read_csv(ROOT/'data/phase1_inventory.csv').set_index('path')
    feats = pd.read_csv(ROOT/'data/processed/features.csv').set_index('path')
    assignments = pd.read_csv(TABLES/'train_assignments.csv')
    domain = feats.loc[assignments.path]
    raw = locate_raw()
    # Two actual files make the padding/length distinction concrete.
    chosen = [inventory.loc['stop/82c6d220_nohash_1.wav'],inventory.loc['up/982babaf_nohash_1.wav']]
    fig,axes = plt.subplots(2,2,figsize=(10,4.8),constrained_layout=True)
    for row,(ax_before,ax_after) in zip(chosen,axes):
        path = row.name
        x,rate = read_audio(raw/path)
        y,ledger = prepare_signal(x,rate,cfg)
        ax_before.plot(np.arange(len(x))/rate,x,color=COLORS[0],lw=.7)
        ax_after.plot(np.arange(len(y))/cfg['sample_rate_hz'],y,color=COLORS[2],lw=.7)
        if ledger['padded_samples']:
            ax_after.axvspan(len(x)/rate,1,color='#e2a14b',alpha=.22,label='Added zeros')
            ax_after.legend(frameon=False,fontsize=8)
        for ax in [ax_before,ax_after]:
            ax.set(xlim=(0,1),ylim=(-1,1),xlabel='Time (s)',ylabel='Amplitude')
        ax_before.set_title(f'{path.split("/")[0]}: original {len(x)/rate:.3f} s')
        ax_after.set_title('DC centering + right padding to 1 s')
    save(fig,'signal_preparation')
    path = assignments.loc[assignments.label.eq('yes'),'path'].iloc[0]
    x,rate = read_audio(raw/path)
    y,_ = prepare_signal(x,rate,cfg)
    _,representations = extract_features(y,len(x)/rate,cfg)
    fig,axes = plt.subplots(1,2,figsize=(10,3.4),constrained_layout=True)
    for ax,key,title,ylabel in zip(axes,['logmel','mfcc'],['40-band log-mel power','MFCC detail: coefficients 1-12'],['Mel band index','Coefficient index']):
        values = representations[key].T if key=='logmel' else representations[key][:,1:].T
        start,end = (-.5,39.5) if key=='logmel' else (.5,12.5)
        limits = {} if key=='logmel' else {'vmin':-np.abs(values).max(),'vmax':np.abs(values).max()}
        image = ax.imshow(values,aspect='auto',origin='lower',extent=(.0075,.9875,start,end),cmap='magma' if key=='logmel' else 'RdBu_r',**limits)
        ax.set(title=title,xlabel='Time (s)',ylabel=ylabel)
        fig.colorbar(image,ax=ax,label='dB' if key=='logmel' else 'DCT coefficient',fraction=.045)
    save(fig,'audio_representations')
    pca = pd.read_csv(TABLES/'pca_variance.csv')
    fig,axes = plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    axes[0].plot(pca.component,pca.cumulative,marker='o',markersize=3,color=COLORS[0])
    axes[0].axhline(.95,color=COLORS[1],ls='--',label='95% target')
    axes[0].set(title='Retained variance: training fit',xlabel='Number of principal components',ylabel='Cumulative explained variance',ylim=(0,1.02))
    axes[0].legend(frameon=False)
    axes[1].bar(pca.component,pca.explained_variance_ratio,color=COLORS[2])
    axes[1].set(title='Variance per retained component',xlabel='Principal component',ylabel='Explained variance ratio')
    save(fig,'pca_variance')
    def scatter_panels(name):
        fig,axes = plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
        for label,color in zip(cfg['labels'],COLORS):
            a = assignments[assignments.label.eq(label)]
            axes[0].scatter(a.pc1,a.pc2,s=5,alpha=.28,color=color,label=label,rasterized=True)
        axes[0].set_title('Colored by command: post-hoc only')
        axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=4,frameon=False,markerscale=3)
        for cluster in sorted(assignments.cluster.unique()):
            a = assignments[assignments.cluster.eq(cluster)]
            axes[1].scatter(a.pc1,a.pc2,s=5,alpha=.28,color=COLORS[cluster],label=f'C{cluster}',rasterized=True)
        axes[1].set_title(f'k-means clusters: k={summary["selected_k"]}')
        axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=min(6,summary['selected_k']),frameon=False,markerscale=3)
        for ax in axes:
            ax.set(xlabel=f'PC1 ({summary["pc1_variance"]:.1%})',ylabel=f'PC2 ({summary["pc2_variance"]:.1%})')
        save(fig,name)
    scatter_panels('pca_commands_clusters')
    fig,axes = plt.subplots(1,2,figsize=(10,4.1),constrained_layout=True)
    sc = axes[0].scatter(assignments.pc1,assignments.pc2,c=domain.rms_dbfs,s=6,alpha=.6,cmap='viridis',rasterized=True)
    fig.colorbar(sc,ax=axes[0],label='Prepared RMS (dBFS)',fraction=.045)
    axes[0].set_title('Recording level across the PCA map')
    frequent = assignments.speaker_id.value_counts().head(5).index
    axes[1].scatter(assignments.pc1,assignments.pc2,s=4,color='#d7e1e8',alpha=.35,rasterized=True)
    for speaker,color in zip(frequent,COLORS):
        a = assignments[assignments.speaker_id.eq(speaker)]
        axes[1].scatter(a.pc1,a.pc2,s=18,color=color,label=speaker,edgecolors='white',linewidth=.25)
    axes[1].set_title('Five most represented training IDs')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=8,frameon=False)
    for ax in axes:
        ax.set(xlabel='PC1',ylabel='PC2')
    save(fig,'pca_energy_speakers')
    search = pd.read_csv(TABLES/'cluster_search_summary.csv')
    fig,axes = plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    axes[0].errorbar(search.k,search.silhouette_mean,yerr=search.silhouette_std,color=COLORS[0],marker='o',capsize=3)
    axes[0].axvline(summary['selected_k'],color=COLORS[1],ls='--',label='Selected k')
    axes[0].set(title='Unlabeled selection: mean +/- SD',xlabel='Number of clusters',ylabel='Training silhouette (fixed 1,200 clips)')
    axes[0].legend(frameon=False)
    axes[1].plot(search.k,search.inertia_mean,color=COLORS[2],marker='o')
    axes[1].set(title='Within-cluster sum of squares',xlabel='Number of clusters',ylabel='Mean training inertia')
    for ax in axes:
        ax.set_xticks(search.k)
    save(fig,'cluster_selection')
    stable = pd.read_csv(TABLES/'stability_checks.csv')
    subsamples = stable[stable.experiment.str.startswith('80%')].anchor_ari
    robust = stable[stable.experiment.str.startswith('Robust')].anchor_ari.iloc[0]
    fig,ax = plt.subplots(figsize=(8,3.3),constrained_layout=True)
    ax.bar(['Initialization seeds','80% speaker subsets','Robust scaling'],[summary['seed_ari_mean'],subsamples.mean(),robust],color=COLORS[:3],width=.55)
    ax.scatter(np.linspace(.85,1.15,len(subsamples)),subsamples,color='#16364d',s=20,zorder=3)
    ax.axhline(1,color='#657686',lw=.8,ls='--')
    ax.set(ylabel='Adjusted Rand index',ylim=(-.1,1.08),title='Stability of training-cluster assignments')
    save(fig,'cluster_stability')
    commands = pd.read_csv(TABLES/'cluster_commands.csv',index_col=0)
    percentages = commands.div(commands.sum(axis=1),axis=0)*100
    fig,ax = plt.subplots(figsize=(9,max(2.8,len(commands)*.5+1)),constrained_layout=True)
    im = ax.imshow(percentages,cmap='Blues',vmin=0,vmax=max(25,float(percentages.to_numpy().max())))
    ax.set_xticks(range(8),cfg['labels'])
    ax.set_yticks(range(len(commands)),[f'C{i} (n={commands.loc[i].sum():,})' for i in commands.index])
    ax.set(xlabel='Command',ylabel='Training cluster',title='Command composition within each cluster')
    for i in range(len(commands)):
        for j in range(8):
            ax.text(j,i,f'{percentages.iloc[i,j]:.1f}%',ha='center',va='center',fontsize=8,
                    color='white' if percentages.iloc[i,j]>percentages.to_numpy().max()*.65 else '#16364d')
    fig.colorbar(im,ax=ax,label='Within-cluster percentage',fraction=.035)
    save(fig,'cluster_command_composition')
    profile = pd.read_csv(TABLES/'cluster_profile_medians.csv',index_col=0)
    columns = ['rms_dbfs','quiet_fraction','centroid_hz','bandwidth_hz','spectral_flatness','zero_crossing_rate']
    effect = (profile[columns]-domain[columns].mean())/domain[columns].std(ddof=0)
    fig,ax = plt.subplots(figsize=(9,max(2.7,len(profile)*.5+1)),constrained_layout=True)
    limit = max(1,abs(effect.to_numpy()).max())
    im = ax.imshow(effect,cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax.set_xticks(range(len(columns)),['RMS','Quiet fraction','Centroid','Bandwidth','Flatness','Zero crossings'],rotation=25,ha='right')
    ax.set_yticks(range(len(profile)),[f'C{i}' for i in profile.index])
    ax.set(title='Cluster medians relative to the training distribution',ylabel='Cluster')
    for i in range(len(profile)):
        for j in range(len(columns)):
            ax.text(j,i,f'{effect.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=9,
                    color='white' if abs(effect.iloc[i,j])>limit*.65 else '#16364d')
    fig.colorbar(im,ax=ax,label='(Cluster median - train mean) / train SD',fraction=.035)
    save(fig,'cluster_profiles')
    compare = pd.read_csv(TABLES/'representation_comparison.csv')
    fig,axes = plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    tick_labels = ['MFCC\nstandard','MFCC\nrobust','Log-mel\nstandard','MFCC\nwithout c0']
    axes[0].bar(range(4),compare.silhouette,color=COLORS[:4])
    axes[0].set_xticks(range(4),tick_labels)
    axes[0].set(ylabel='Training silhouette',title='Fixed-k sensitivity to representation')
    pos = np.arange(4)
    axes[1].bar(pos-.18,compare.command_ami,width=.36,label='Command',color=COLORS[0])
    axes[1].bar(pos+.18,compare.energy_ami,width=.36,label='Energy quartile',color=COLORS[1])
    axes[1].set_xticks(pos,tick_labels)
    axes[1].set(ylabel='Adjusted mutual information',title='Post-hoc associations')
    axes[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.2),ncol=2)
    save(fig,'representation_comparison')
    augmentation = pd.read_csv(TABLES/'augmentation_ledger.csv')
    row = augmentation.iloc[np.abs(augmentation.shift_samples).argmax()]
    # Replay the recorded operation exactly for the illustrative waveform.
    x,rate = read_audio(raw/row.path)
    y,_ = prepare_signal(x,rate,cfg)
    shift = int(row.shift_samples)
    shifted = np.zeros_like(y)
    if shift > 0:
        shifted[shift:] = y[:-shift]
    elif shift < 0:
        shifted[:shift] = y[-shift:]
    else:
        shifted[:] = y
    altered = shifted*10**(row.effective_gain_db/20)
    fig,axes = plt.subplots(2,1,figsize=(9,3.7),sharex=True,sharey=True,constrained_layout=True)
    axes[0].plot(np.arange(len(y))/16000,y,color=COLORS[0],lw=.65)
    axes[1].plot(np.arange(len(y))/16000,altered,color=COLORS[2],lw=.65)
    axes[0].set_title(f'Training example: {row.path}')
    axes[1].set_title(f'Shift {shift/16:+.1f} ms; effective gain {row.effective_gain_db:+.2f} dB')
    for ax in axes:
        ax.set(ylabel='Amplitude',xlim=(0,1))
    axes[1].set_xlabel('Time (s)')
    save(fig,'controlled_augmentation')
    bundle = joblib.load(ROOT/'artifacts/mfcc_pca_kmeans.joblib')
    Z = bundle['preprocessor'].transform(feats.loc[assignments.path,bundle['feature_columns']].to_numpy())
    distances = np.linalg.norm(Z-bundle['clusterer'].cluster_centers_[assignments.cluster],axis=1)
    cases = assignments[['path','label','cluster']].copy()
    cases['distance_to_center'] = distances
    cases['rms_dbfs'] = domain.rms_dbfs.to_numpy()
    cases['quiet_fraction'] = domain.quiet_fraction.to_numpy()
    selected_cases = cases.sort_values(['cluster','distance_to_center','path']).groupby('cluster').head(3)
    selected_cases.to_csv(TABLES/'cluster_representatives.csv',index=False)
    displayed = selected_cases.groupby('cluster').head(1).head(2)
    fig,axes = plt.subplots(len(displayed),2,figsize=(10,5.5),constrained_layout=True,squeeze=False)
    for row,(ax1,ax2) in zip(displayed.itertuples(),axes):
        x,rate = read_audio(raw/row.path)
        y,_ = prepare_signal(x,rate,cfg)
        _,rep = extract_features(y,len(x)/rate,cfg)
        ax1.plot(np.arange(len(y))/16000,y,color=COLORS[row.cluster],lw=.7)
        ax1.set(xlim=(0,1),ylim=(-1,1),xlabel='Time (s)',ylabel='Amplitude',
                title=f'C{row.cluster}: {row.label}; RMS {row.rms_dbfs:.1f} dBFS')
        im = ax2.imshow(rep['logmel'].T,origin='lower',aspect='auto',extent=(.0075,.9875,-.5,39.5),
                        cmap='magma',vmin=-100,vmax=-15)
        ax2.set(xlabel='Time (s)',ylabel='Mel band',title='Nearest-to-center example: log-mel power')
        fig.colorbar(im,ax=ax2,label='dB',fraction=.04)
    save(fig,'cluster_representatives')
    print('Created 12 measured Phase 2 figures and representative-clip table.')


if __name__ == '__main__':
    main()

"""Run reproducible Phase 2 preparation, feature engineering, PCA and clustering."""
import argparse
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (adjusted_mutual_info_score,adjusted_rand_score,
                             calinski_harabasz_score,davies_bouldin_score,silhouette_score)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from threadpoolctl import threadpool_limits
from audio_features import read_audio,prepare_signal,mel_filterbank,feature_names,extract_features,augment_train

TABLES = ROOT/'reports/tables'


def dump_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def ordered_sample(paths,n,seed=42):
    keys = [hashlib.sha256(f'{seed}:{path}'.encode()).hexdigest() for path in paths]
    return np.argsort(keys)[:min(n,len(keys))]


def locate_raw():
    candidates = [ROOT/'data/raw/mini_speech_commands',ROOT.parent/'data/raw/mini_speech_commands']
    return next((path for path in candidates if path.is_dir()),candidates[0])


def curate(cfg):
    inventory = pd.read_csv(ROOT/'data/phase1_inventory.csv')
    required = ['path','speaker_id','label','split','pcm_sha256','file_sha256']
    if inventory[required].isna().any().any() or not inventory.path.is_unique:
        raise ValueError('Missing identity/label/split or nonunique path; no metadata imputation')
    if not set(inventory.label).issubset(cfg['labels']):
        raise ValueError('Unexpected command label')
    good = inventory[inventory.status.eq('ok')].sort_values('path').copy()
    for key in ['speaker_id','pcm_sha256']:
        if good.groupby(key).split.nunique().gt(1).any():
            raise ValueError(f'{key} crosses partitions')
    if good.groupby('pcm_sha256').label.nunique().gt(1).any():
        raise ValueError('Exact copies have conflicting labels')
    duplicates = good[good.duplicated('pcm_sha256',keep='first')].copy()
    duplicates['decision'] = 'drop exact copy; keep first path alphabetically'
    duplicates[['path','label','speaker_id','split','pcm_sha256','decision']].to_csv(TABLES/'removed_duplicates.csv',index=False)
    clean = good.drop_duplicates('pcm_sha256',keep='first').reset_index(drop=True)
    pd.crosstab(clean.label,clean.split).reindex(cfg['labels'])[['train','validation','test']].to_csv(TABLES/'effective_class_counts.csv')
    clean[['path','label','speaker_id','split','pcm_sha256']].to_csv(ROOT/'data/processed/retained_clips.csv',index=False)
    return inventory,clean


def extract(clean,cfg):
    raw = locate_raw()
    if not raw.is_dir():
        raise SystemExit('Raw WAVs missing. Run python src/download_data.py first.')
    names = feature_names(cfg)
    columns = sum(names.values(),[])
    bank = mel_filterbank(cfg)
    values,preparation = [],[]
    for i,row in enumerate(clean.itertuples(),1):
        x,rate = read_audio(raw/row.path,row.file_sha256)
        y,ledger = prepare_signal(x,rate,cfg)
        feature,_ = extract_features(y,len(x)/rate,cfg,bank)
        values.append(feature)
        preparation.append({'path':row.path,'split':row.split,**ledger})
        if i % 1000 == 0:
            print(f'Prepared and extracted {i}/{len(clean)} clips',flush=True)
    features = pd.DataFrame(values,columns=columns)
    features.insert(0,'path',clean.path)
    features.to_csv(ROOT/'data/processed/features.csv',index=False,float_format='%.10g')
    pd.DataFrame(preparation).to_csv(TABLES/'preparation_ledger.csv',index=False)
    catalog = []
    for family,group in names.items():
        for name in group:
            if family == 'mfcc':
                definition = 'MFCC coefficient time mean/std or five-frame regression-delta mean/std; coefficients 0-12'
                reason = 'Spectral envelope and its temporal change, without using the command label'
            elif family == 'logmel':
                definition = 'Time mean or population std of a 40-band HTK mel log-power representation, in dB'
                reason = 'Frequency-local energy and temporal variation, as an alternative representation'
            else:
                definitions = {
                    'duration_s':('Original unpadded duration, seconds','Length and padding requirements'),
                    'rms_dbfs':('Whole prepared clip RMS in dBFS','Recording level, including padding effect'),
                    'peak_abs':('Maximum absolute prepared amplitude','Amplitude extremes'),
                    'quiet_fraction':('Fraction of overlapping 25 ms frames below -50 dBFS','Low-amplitude occupancy; not SNR or voice activity'),
                    'zero_crossing_rate':('Fraction of adjacent prepared samples with different signs','Time-domain oscillation and high-frequency variation'),
                    'centroid_hz':('Mean power-weighted spectral centroid after pre-emphasis, Hz','Spectral location'),
                    'bandwidth_hz':('Mean power-weighted spectral standard deviation, Hz','Spectral spread'),
                    'rolloff85_hz':('Mean frequency below which 85% of spectral power lies','High-frequency energy extent'),
                    'spectral_flatness':('Mean geometric/arithmetic power ratio; silent frames set to zero','Tonal versus broad-band spectral shape'),
                    'spectral_flux':('Mean RMS difference of adjacent unit-sum power spectra','Time variation of spectral shape'),
                }
                definition,reason = definitions[name]
            catalog.append({'feature':name,'family':family,'definition':definition,'rationale':reason,
                            'used_in_primary_clustering':family == 'mfcc'})
    pd.DataFrame(catalog).to_csv(TABLES/'feature_catalog.csv',index=False)


def transformer(kind,cfg):
    return Pipeline([('nonconstant',VarianceThreshold()),
                     ('scaler',StandardScaler() if kind == 'standard' else RobustScaler()),
                     ('pca',PCA(n_components=cfg['pca_variance'],svd_solver='full',whiten=False))])


def kmeans(k,seed,cfg):
    return KMeans(n_clusters=k,random_state=seed,n_init=cfg['kmeans_n_init'],
                  algorithm='lloyd',max_iter=300)


def analyze(clean,cfg):
    names = feature_names(cfg)
    all_features = pd.read_csv(ROOT/'data/processed/features.csv').set_index('path').loc[clean.path].copy().reset_index()
    if not np.isfinite(all_features[sum(names.values(),[])].to_numpy()).all():
        raise ValueError('Feature missingness requires review; no blind imputation')
    is_train,is_val = clean.split.eq('train').to_numpy(),clean.split.eq('validation').to_numpy()
    meta = clean[is_train].reset_index(drop=True)
    val_meta = clean[is_val].reset_index(drop=True)
    train = all_features.loc[is_train,names['mfcc']].to_numpy()
    pre = transformer('standard',cfg)
    Z = pre.fit_transform(train)
    sampled = ordered_sample(meta.path,cfg['silhouette_sample'])
    sample_distances = euclidean_distances(Z[sampled])
    np.fill_diagonal(sample_distances,0)
    sweep,labels_by_k = [],{}
    for k in cfg['k_values']:
        labels_by_k[k] = []
        for seed in cfg['seeds']:
            model = kmeans(k,seed,cfg).fit(Z)
            assignments = model.labels_
            labels_by_k[k].append(assignments)
            sweep.append({'k':k,'seed':seed,'silhouette':silhouette_score(sample_distances,assignments[sampled],metric='precomputed'),
                          'davies_bouldin':davies_bouldin_score(Z,assignments),
                          'calinski_harabasz':calinski_harabasz_score(Z,assignments),
                          'inertia':model.inertia_, 'smallest_cluster':int(np.bincount(assignments).min())})
        print(f'Completed unlabeled cluster search: k={k}',flush=True)
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(TABLES/'cluster_search_runs.csv',index=False)
    grouped = sweep.groupby('k').agg(silhouette_mean=('silhouette','mean'),silhouette_std=('silhouette','std'),
                                     db_mean=('davies_bouldin','mean'),ch_mean=('calinski_harabasz','mean'),inertia_mean=('inertia','mean'))
    grouped['seed_ari_mean'] = [np.mean([adjusted_rand_score(a,b) for a,b in combinations(labels_by_k[k],2)]) for k in grouped.index]
    grouped.to_csv(TABLES/'cluster_search_summary.csv')
    # Command/speaker/quality labels are not consulted in the selection rule.
    chosen = int(grouped.reset_index().sort_values(['silhouette_mean','k'],ascending=[False,True]).iloc[0].k)
    model = kmeans(chosen,cfg['final_seed'],cfg).fit(Z)
    labels = model.labels_
    valZ = pre.transform(all_features.loc[is_val,names['mfcc']].to_numpy())
    val_labels = model.predict(valZ)
    assignments = meta[['path','label','speaker_id','split']].copy()
    assignments['cluster'] = labels
    assignments['pc1'],assignments['pc2'] = Z[:,:2].T
    assignments.to_csv(TABLES/'train_assignments.csv',index=False)
    validation = val_meta[['path','label','speaker_id','split']].copy()
    validation['cluster'] = val_labels
    validation['pc1'],validation['pc2'] = valZ[:,:2].T
    validation.to_csv(TABLES/'validation_assignments.csv',index=False)
    pd.DataFrame({'component':np.arange(1,pre['pca'].n_components_+1),
                  'explained_variance_ratio':pre['pca'].explained_variance_ratio_,
                  'cumulative':np.cumsum(pre['pca'].explained_variance_ratio_)}).to_csv(TABLES/'pca_variance.csv',index=False)
    pd.DataFrame(pre['pca'].components_[:2].T,index=np.array(names['mfcc'])[pre['nonconstant'].get_support()],
                 columns=['PC1','PC2']).to_csv(TABLES/'pca_loadings.csv',index_label='feature')
    profile = all_features.loc[is_train,names['domain']].reset_index(drop=True)
    profile['cluster'] = labels
    profile.groupby('cluster').median().to_csv(TABLES/'cluster_profile_medians.csv')
    pd.crosstab(labels,meta.label).reindex(columns=cfg['labels'],fill_value=0).to_csv(TABLES/'cluster_commands.csv',index_label='cluster')
    # Quantile edges are learned on training only; used purely for post-hoc diagnostics.
    energy_bins,energy_edges = pd.qcut(profile.rms_dbfs,4,retbins=True,labels=False,duplicates='drop')
    associations = []
    for split,ids,group_labels,frame in [('train',meta,labels,all_features.loc[is_train]),
                                          ('validation',val_meta,val_labels,all_features.loc[is_val])]:
        bins = pd.cut(frame.rms_dbfs,[-np.inf,*energy_edges[1:-1],np.inf],labels=False)
        for variable,values in [('command',ids.label),('speaker_id',ids.speaker_id),('energy_quartile',bins)]:
            associations.append({'split':split,'variable':variable,
                                 'adjusted_mutual_information':adjusted_mutual_info_score(values,group_labels),
                                 'adjusted_rand_index':adjusted_rand_score(values,group_labels)})
    pd.DataFrame(associations).to_csv(TABLES/'cluster_associations.csv',index=False)
    stability = []
    anchor = ordered_sample(meta.path,1500,seed=313)
    speakers = np.sort(meta.speaker_id.unique())
    for i in range(cfg['stability_repeats']):
        seed = 100+i
        rng = np.random.default_rng(seed)
        kept = rng.choice(speakers,size=int(len(speakers)*cfg['speaker_subsample_fraction']),replace=False)
        keep = meta.speaker_id.isin(kept).to_numpy()
        subpre = transformer('standard',cfg)
        subZ = subpre.fit_transform(train[keep])
        submodel = kmeans(chosen,seed,cfg).fit(subZ)
        pred = submodel.predict(subpre.transform(train[anchor]))
        stability.append({'experiment':'80% speaker subsample; refit scaler/PCA/k-means','seed':seed,
                          'training_speakers':len(kept),'training_clips':int(keep.sum()),
                          'anchor_ari':adjusted_rand_score(labels[anchor],pred)})
    # Alternative scaling and representation use the already selected k.
    comparisons = []
    for name,family,scaling in [('MFCC / standard','mfcc','standard'),('MFCC / robust','mfcc','robust'),
                                ('Log-mel / standard','logmel','standard'),
                                ('MFCC without coefficient 0','mfcc_shape','standard')]:
        columns = names['mfcc'] if family == 'mfcc' else names['logmel'] if family == 'logmel' else [c for c in names['mfcc'] if not c.startswith('mfcc_00_')]
        values = all_features.loc[is_train,columns].to_numpy()
        altpre = transformer(scaling,cfg)
        altZ = altpre.fit_transform(values)
        altmodel = kmeans(chosen,cfg['final_seed'],cfg).fit(altZ)
        comparisons.append({'representation':name,'features':len(columns),'pca_components':altpre['pca'].n_components_,
                            'silhouette':silhouette_score(altZ[sampled],altmodel.labels_[sampled]),
                            'ari_to_primary':adjusted_rand_score(labels,altmodel.labels_),
                            'command_ami':adjusted_mutual_info_score(meta.label,altmodel.labels_),
                            'energy_ami':adjusted_mutual_info_score(energy_bins,altmodel.labels_)})
        if name == 'MFCC / robust':
            stability.append({'experiment':'RobustScaler instead of StandardScaler','seed':cfg['final_seed'],
                              'training_speakers':len(speakers),'training_clips':len(meta),
                              'anchor_ari':adjusted_rand_score(labels[anchor],altmodel.predict(altZ[anchor]))})
    pd.DataFrame(stability).to_csv(TABLES/'stability_checks.csv',index=False)
    pd.DataFrame(comparisons).to_csv(TABLES/'representation_comparison.csv',index=False)
    joblib.dump({'preprocessor':pre,'clusterer':model,'feature_columns':names['mfcc'],
                 'config':cfg,'training_paths':meta.path.tolist()},ROOT/'artifacts/mfcc_pca_kmeans.joblib',compress=3)
    # Transform all retained clips for later phases; test is never used in analysis or fitting.
    transformed = pre.transform(all_features[names['mfcc']].to_numpy())
    np.savez_compressed(ROOT/'artifacts/pca_coordinates.npz',path=clean.path.to_numpy(dtype=str),coordinates=transformed)
    augmentation = []
    rng = np.random.default_rng(321)
    raw,bank = locate_raw(),mel_filterbank(cfg)
    selection = ordered_sample(meta.path,cfg['augmentation_sample'],seed=321)
    augmented = []
    for index in selection:
        row = meta.iloc[index]
        x,rate = read_audio(raw/row.path,row.file_sha256)
        prepared,_ = prepare_signal(x,rate,cfg)
        altered,ledger = augment_train(prepared,row.split,rng,cfg)
        feature,_ = extract_features(altered,len(x)/rate,cfg,bank)
        augmented.append(feature[:len(names['mfcc'])])
        augmentation.append({'path':row.path,'split':row.split,'cluster_before':int(labels[index]),**ledger})
    after = model.predict(pre.transform(np.array(augmented)))
    for row,cluster in zip(augmentation,after):
        row['cluster_after'] = int(cluster)
    pd.DataFrame(augmentation).to_csv(TABLES/'augmentation_ledger.csv',index=False)
    summary = {'author':'Nima Balotf','phase':2,'input_clips':len(pd.read_csv(ROOT/'data/phase1_inventory.csv')),
               'retained_clips':len(clean),'removed_exact_copies':len(pd.read_csv(TABLES/'removed_duplicates.csv')),
               'split_counts':clean.split.value_counts().to_dict(),'speaker_counts':clean.groupby('split').speaker_id.nunique().to_dict(),
               'feature_counts':{key:len(value) for key,value in names.items()},'primary_features':len(names['mfcc']),
               'selected_k':chosen,'selection_rule':'Highest mean training silhouette over three seeds; ties prefer smaller k',
               'silhouette_sample':len(sampled),'primary_pca_components':int(pre['pca'].n_components_),
               'retained_variance':float(pre['pca'].explained_variance_ratio_.sum()),
               'pc1_variance':float(pre['pca'].explained_variance_ratio_[0]),'pc2_variance':float(pre['pca'].explained_variance_ratio_[1]),
               'primary_silhouette_mean':float(grouped.loc[chosen,'silhouette_mean']),
               'seed_ari_mean':float(grouped.loc[chosen,'seed_ari_mean']),
               'speaker_subsample_ari_mean':float(pd.DataFrame(stability).iloc[:cfg['stability_repeats']].anchor_ari.mean()),
               'augmentation_same_cluster_fraction':float(np.mean(labels[selection]==after)),
               'augmentation_ari':float(adjusted_rand_score(labels[selection],after)),
               'features_missing':int(all_features[sum(names.values(),[])].isna().sum().sum()),
               'test_used_for_fit_or_selection':False,'augmentation_used_in_fit':False,
               'inventory_sha256':hashlib.sha256((ROOT/'data/phase1_inventory.csv').read_bytes()).hexdigest(),
               'config_sha256':hashlib.sha256((ROOT/'configs/phase2.json').read_bytes()).hexdigest(),
               'packages':{name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','joblib','cloudpickle','narwhals','threadpoolctl','matplotlib','reportlab','Pillow','pypdf']}}
    dump_json(ROOT/'reports/summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-features',action='store_true',help='Reuse the checked feature CSV for analysis')
    args = parser.parse_args()
    cfg = json.loads((ROOT/'configs/phase2.json').read_text())
    for path in [TABLES,ROOT/'data/processed',ROOT/'artifacts']:
        path.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=cfg['thread_limit']):
        inventory,clean = curate(cfg)
        if not args.reuse_features:
            extract(clean,cfg)
        analyze(clean,cfg)


if __name__ == '__main__':
    main()

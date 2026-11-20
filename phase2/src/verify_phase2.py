"""Check signal math, split integrity, fitted statistics, and Phase 2 deliverables."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from pypdf import PdfReader
from PIL import Image
from scipy.fft import dct
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits
from audio_features import read_audio,prepare_signal,mel_filterbank,feature_names,extract_features,augment_train
from run_phase2 import locate_raw,ordered_sample

ROOT = Path(__file__).resolve().parents[1]
T = ROOT/'reports/tables'


def main():
    passed = []
    def check(ok,message):
        if not ok:
            raise AssertionError(message)
        passed.append(message)
    cfg = json.loads((ROOT/'configs/phase2.json').read_text())
    summary = json.loads((ROOT/'reports/summary.json').read_text())
    original = pd.read_csv(ROOT/'data/phase1_inventory.csv')
    retained = pd.read_csv(ROOT/'data/processed/retained_clips.csv')
    features = pd.read_csv(ROOT/'data/processed/features.csv').set_index('path')
    names = feature_names(cfg)
    columns = sum(names.values(),[])
    check(len(original)==8000 and original.path.is_unique,'Pinned 8,000-clip inventory has unique paths')
    expected = original[original.status.eq('ok')].sort_values('path').drop_duplicates('pcm_sha256')
    check(retained.path.tolist()==expected.path.tolist(),'Exact-copy representatives follow the stated alphabetic rule')
    check(len(retained)==7973 and len(original)-len(retained)==27,'Exactly 27 duplicate copies removed')
    check(retained.groupby('speaker_id').split.nunique().eq(1).all(),'No speaker spans partitions')
    check(retained.groupby('pcm_sha256').split.nunique().eq(1).all(),'No exact PCM content spans partitions')
    check(retained.split.value_counts().to_dict()==summary['split_counts'],'Retained split counts match summary')
    check(set(features.index)==set(retained.path) and features.index.is_unique,'Feature rows correspond one-to-one with retained recordings')
    check(features.columns.tolist()==columns and len(columns)==142,'Declared 52 MFCC, 80 log-mel, and 10 domain features are present')
    check(np.isfinite(features.to_numpy()).all(),'All derived measurements are finite; no silent imputation')
    ledger = pd.read_csv(T/'preparation_ledger.csv').set_index('path').loc[retained.path]
    header = original.set_index('path').loc[retained.path]
    check(ledger.target_samples.eq(16000).all(),'Every retained signal has 16,000 output samples')
    check(np.array_equal(ledger.padded_samples,np.maximum(0,16000-header.n_frames)),'Padding counts agree with independent WAV-header arithmetic')
    check(ledger.resampled.sum()==0 and ledger.cropped_samples.sum()==0,'Observed corpus needs neither resampling nor cropping')
    # Synthetic invariants exercise branches not needed by the observed corpus.
    x = .25+np.sin(2*np.pi*1000*np.arange(8000)/16000)*.1
    y,info = prepare_signal(x,16000,cfg)
    check(np.all(y[8000:]==0) and abs(y[:8000].mean())<1e-12,'DC removal precedes right padding and preserves exact padded zeros')
    long = np.sin(2*np.pi*400*np.arange(20000)/16000)
    cropped,_ = prepare_signal(long,16000,cfg)
    check(np.allclose(cropped,(long-long.mean())[2000:18000]),'Long inputs use the documented center crop')
    resampled,record = prepare_signal(np.sin(2*np.pi*700*np.arange(8000)/8000),8000,cfg)
    check(record['resampled'] and len(resampled)==16000,'Rate conversion produces the target length')
    sinusoid = .1*np.sin(2*np.pi*1000*np.arange(16000)/16000)
    signal,_ = prepare_signal(sinusoid,16000,cfg)
    value,rep = extract_features(signal,1,cfg)
    check(rep['frames']==98 and abs(value[-5]-1000)<10,'Known 1 kHz tone has the expected framing and spectral centroid')
    filters,_ = mel_filterbank(cfg)
    check(filters.shape==(40,257) and np.all(filters>=0) and np.all(filters.sum(axis=1)>0),'Mel filters have valid support and dimensions')
    test = np.arange(40,dtype=float)
    index = np.arange(40)
    direct = np.array([np.sum(test*np.cos(np.pi*(index+.5)*k/40))*np.sqrt((1 if k==0 else 2)/40) for k in index])
    check(np.allclose(dct(test,type=2,norm='ortho'),direct,atol=1e-11),'DCT-II agrees with the independent orthonormal formula')
    zero,_ = prepare_signal(np.zeros(16000),16000,cfg)
    zero_features,_ = extract_features(zero,1,cfg)
    check(np.isfinite(zero_features).all(),'A synthetic zero signal is handled by explicit numeric floors')
    for split in ['validation','test']:
        rejected = False
        try:
            augment_train(signal,split,np.random.default_rng(1),cfg)
        except ValueError:
            rejected = True
        check(rejected,f'Augmentation rejects {split} inputs')
    augmentation = pd.read_csv(T/'augmentation_ledger.csv')
    check(augmentation.split.eq('train').all() and len(augmentation)==300,'All 300 recorded perturbations are training-only')
    check(augmentation.shift_samples.abs().le(1280).all() and augmentation.retained_energy_fraction.ge(.99-1e-12).all(),
          'Recorded shifts obey the time and retained-energy limits')
    check(augmentation.requested_gain_db.between(-3,3).all(),'Requested gains obey the declared range')
    bundle = joblib.load(ROOT/'artifacts/mfcc_pca_kmeans.joblib')
    train_meta = retained[retained.split.eq('train')]
    train = features.loc[train_meta.path,names['mfcc']].to_numpy()
    pre = bundle['preprocessor']
    support = pre['nonconstant'].get_support()
    check(bundle['training_paths']==train_meta.path.tolist(),'Fitted bundle records only the exact training identities')
    check(np.allclose(pre['scaler'].mean_,train[:,support].mean(axis=0),rtol=1e-10,atol=1e-10),
          'Stored scaling means independently match training data only')
    check(np.allclose(pre['scaler'].var_,train[:,support].var(axis=0),rtol=1e-10,atol=1e-10),
          'Stored scaling variances independently match training data only')
    check(np.allclose(pre['pca'].components_@pre['pca'].components_.T,np.eye(pre['pca'].n_components_),atol=1e-10),
          'PCA retained directions are orthonormal')
    check(pre['pca'].explained_variance_ratio_.sum()>=.95,'PCA retains the configured variance target')
    train_assign = pd.read_csv(T/'train_assignments.csv')
    check(np.array_equal(bundle['clusterer'].predict(pre.transform(train)),train_assign.cluster),'Saved objects reproduce all training cluster assignments')
    val = retained[retained.split.eq('validation')]
    val_assign = pd.read_csv(T/'validation_assignments.csv')
    check(np.array_equal(bundle['clusterer'].predict(pre.transform(features.loc[val.path,names['mfcc']].to_numpy())),val_assign.cluster),
          'Saved objects reproduce validation assignments without refitting')
    search = pd.read_csv(T/'cluster_search_runs.csv')
    chosen = int(search.groupby('k').silhouette.mean().sort_values(ascending=False).index[0])
    check(chosen==summary['selected_k']==bundle['clusterer'].n_clusters,'Selected k independently agrees with unlabeled mean silhouette')
    check(len(search)==len(cfg['k_values'])*len(cfg['seeds']),'Every declared k and seed was evaluated')
    assoc = pd.read_csv(T/'cluster_associations.csv')
    score = adjusted_mutual_info_score(train_meta.label,train_assign.cluster)
    check(np.isclose(score,assoc[(assoc.split=='train')&(assoc.variable=='command')].adjusted_mutual_information.iloc[0]),
          'Reported command association independently matches the fitted partition')
    coordinates = np.load(ROOT/'artifacts/pca_coordinates.npz')
    check(coordinates['path'].tolist()==retained.path.tolist() and np.isfinite(coordinates['coordinates']).all(),
          'Exported coordinates retain all clip identities without missing values')
    raw = locate_raw()
    sample = retained[retained.split.ne('test')].reset_index(drop=True)
    for i in ordered_sample(sample.path,16,seed=918):
        row = sample.iloc[i]
        waveform,rate = read_audio(raw/row.path,original.set_index('path').loc[row.path,'file_sha256'])
        prepared,_ = prepare_signal(waveform,rate,cfg)
        recomputed,_ = extract_features(prepared,len(waveform)/rate,cfg)
        if not np.allclose(recomputed,features.loc[row.path,columns].to_numpy(),rtol=1e-8,atol=1e-9):
            raise AssertionError('Raw-to-feature replay differs: '+row.path)
    check(True,'Sixteen independently replayed training/validation WAV features match the exported table')
    manifest = json.loads((ROOT/'tmp/report_manifest.json').read_text())
    pdf = ROOT/'reports/phase2.pdf'
    reader = PdfReader(pdf)
    check(len(reader.pages)==manifest['pages']==17,'English report contains the intended 17 pages')
    check(reader.metadata.author=='Nima Balotf','PDF author metadata is correct')
    check(all(page.extract_text().isascii() for page in reader.pages),'Report text uses English script only')
    check(manifest['figures']==12 and min(manifest['content_bottoms_pt'])>=54,'All 12 figure captions and content remain above the footer')
    links = [a.get_object().get('/A',{}).get('/URI') for page in reader.pages for a in page.get('/Annots',[])]
    check(len(links)==5 and all(url and url.startswith('https://') for url in links),'Five direct source links are embedded in the PDF')
    check(hashlib.sha256(pdf.read_bytes()).hexdigest()==manifest['pdf_sha256'],'Current PDF matches the rendered-build manifest')
    for path in (ROOT/'figures').glob('*.png'):
        with Image.open(path) as image:
            image.verify()
    check(len(list((ROOT/'figures').glob('*.png')))==12,'All 12 figure assets are readable')
    check(all((ROOT/name).is_file() for name in ['README.md','FEATURES.md','AI_USAGE.md','PROJECT_CARD.md','requirements.txt']),
          'Required English documentation and disclosure are present')
    text_paths = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix in {'.py','.md','.json','.csv','.txt'}
                  and not any(part in {'output','tmp','raw','archives','__pycache__'} for part in p.relative_to(ROOT).parts)]
    check(all(p.read_text(encoding='utf-8').isascii() for p in text_paths),'All authored source, documentation, settings, and tables use English script')
    result = {'passed':len(passed),'checks':passed,'pdf_sha256':manifest['pdf_sha256'],
              'visual_review':'Rendered pages are inspected separately.'}
    (ROOT/'tmp/verification.json').write_text(json.dumps(result,indent=2))
    print(f'PASS: {len(passed)} meaningful checks; 7,973 recordings; 142 features; 17 report pages.')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()

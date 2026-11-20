"""Build the English Phase 2 report from computed tables and figures."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import report_layout as layout
from report_layout import Report,CW,MUTED

ROOT = Path(__file__).resolve().parents[1]
T = ROOT/'reports/tables'
layout.FIGURE_NOTES = {
    'signal_preparation':'Actual recordings before and after signal preparation',
    'audio_representations':'Log-mel power and MFCC 1-12 detail; c0 omitted from this display only',
    'pca_variance':'Training-only PCA: retained and cumulative variance',
    'pca_commands_clusters':'The same training PCA coordinates, with two post-hoc colorings',
    'pca_energy_speakers':'Recording level and selected speaker IDs across the training map',
    'cluster_selection':'Training silhouette across seeds and the inertia curve',
    'cluster_stability':'Initialization, speaker-subset, and scaling sensitivity',
    'cluster_command_composition':'Within-cluster command composition; rows sum to 100%',
    'cluster_profiles':'Interpretable acoustic profiles of the training clusters',
    'representation_comparison':'Fixed-k representation and scaling comparisons',
    'controlled_augmentation':'A recorded training-only time-shift and gain operation',
    'cluster_representatives':'Nearest-to-center training examples, with a shared log-mel scale',
}


def main():
    s = json.loads((ROOT/'reports/summary.json').read_text())
    cfg = json.loads((ROOT/'configs/phase2.json').read_text())
    ledger = pd.read_csv(T/'preparation_ledger.csv')
    retained = pd.read_csv(ROOT/'data/processed/retained_clips.csv')
    inventory = pd.read_csv(ROOT/'data/phase1_inventory.csv')
    kept = inventory[inventory.path.isin(retained.path)]
    associations = pd.read_csv(T/'cluster_associations.csv').set_index(['split','variable'])
    profiles = pd.read_csv(T/'cluster_profile_medians.csv',index_col=0)
    compare = pd.read_csv(T/'representation_comparison.csv')
    search = pd.read_csv(T/'cluster_search_summary.csv').set_index('k')
    stability = pd.read_csv(T/'stability_checks.csv')
    cases = pd.read_csv(T/'cluster_representatives.csv')
    aug = pd.read_csv(T/'augmentation_ledger.csv')
    target = ROOT/'reports/phase2.pdf'
    temp = ROOT/'tmp/phase2.pdf'
    r = Report(temp,total_pages=17)

    r.new('Audio Features and Hidden Patterns','Phase 2 report | Path D: Google Speech Commands')
    r.p('This phase turns the audited WAV recordings into reusable numerical representations. It standardizes signal length, removes exact copies, extracts audio features, and investigates training-data structure with PCA and k-means.',size=11,leading=18)
    r.section('Computed results')
    r.kpis([[f"{s['retained_clips']:,}",'Distinct recordings retained'],
            [str(sum(s['feature_counts'].values())),'Extracted audio descriptors'],
            [str(s['primary_pca_components']),'Principal components retained'],
            [str(s['selected_k']),'Clusters selected without labels']])
    r.callout('Main finding',f"The highest mean training silhouette is {s['primary_silhouette_mean']:.3f} at k={s['selected_k']}. The clusters are stable across initializations, but their relationship with command labels is weak.")
    r.section('Meaning for the project')
    r.p('The discovered groups distinguish broad recording and spectral conditions. They provide a useful diagnostic for later modeling, while eight-command recognition remains a supervised learning task for the next phase.')
    r.latin('Nima Balotf | Report date: 4 November 2026',size=8.5)

    r.new('Purpose, Scope, and Evaluation Boundaries')
    r.p('The project asks how short English commands can be recognized for unseen speakers. Phase 2 examines whether practical audio features reveal structure before a classifier is fitted. It continues the same Speech Commands mini corpus and speaker partition used in Phase 1 [1,2].')
    r.section('Phase 2 requirements and delivered evidence')
    r.table(['Requirement','Implementation / evidence'],[
        ['Repeatable preparation','Raw-content checks, DC centering, fixed length; run_phase2.py'],
        ['Missingness and duplicates','Explicit failure/review rules; retained and removed clip tables'],
        ['Audio feature engineering','52 MFCC, 80 log-mel, 10 interpretable descriptors'],
        ['Dimension reduction','Training-only PCA retaining at least 95% variance'],
        ['Unsupervised parameter selection','k=2..12; mean silhouette across three seeds'],
        ['Stability and interpretation','Speaker subsets, scaling, feature ablation, cluster profiles'],
        ['Controlled augmentation','300 training-only perturbations, with an operation ledger']
    ],widths=[CW*.36,CW*.64],size=8.6,row_h=29)
    r.section('What is fitted, and where?')
    r.p('The feature definitions, search range, seeds, and thresholds are recorded in the configuration. Constant-feature removal, scaling, PCA, and k-means are fitted using training clips only. Cluster selection uses no command, speaker, or quality labels.')
    r.p('Validation associations are reported after the model is fixed. Test features are transformed and stored for later phases; test outcomes do not enter parameter selection or the figures.',size=9.6)

    r.new('Signal Preparation: Fixed Rate and Length')
    r.figure('signal_preparation',max_h=220)
    r.table(['Operation','Observed retained files / samples'],[
        ['Input / target rate','16,000 Hz / 16,000 Hz'],
        ['Target length','16,000 samples = 1.000 s'],
        ['Clips requiring resampling',str(int(ledger.resampled.sum()))],
        ['Clips padded / cropped',f'{int(ledger.padded_samples.gt(0).sum())} / {int(ledger.cropped_samples.gt(0).sum())}']
    ],widths=[CW*.50,CW*.50],row_h=24,size=8.7)
    r.analysis('The first illustrated recording is shorter than one second; its missing tail is filled with zeros. The second has activity near the end of a full-length clip.',
               'A uniform array length makes feature extraction consistent, while preserving original duration distinguishes recorded signal from added padding. DC centering uses the real samples before padding.',
               'Use the complete one-second window. The implementation resamples when necessary and center-crops longer inputs; neither operation was needed for this corpus. No peak or RMS normalization is applied.',size=9.5,leading=14)

    r.new('Audio Representations and Feature Design')
    r.figure('audio_representations',max_h=205)
    r.table(['Setting','Value'],[
        ['Analysis window / hop','400 samples (25 ms) / 160 samples (10 ms)'],
        ['FFT / window / frame centering','512 / symmetric Hann / disabled'],
        ['Pre-emphasis / power scaling','0.97 / squared Hann-window sum'],
        ['Mel bank','40 unnormalized HTK triangular bands, 20-7,600 Hz'],
        ['Log power / MFCC','10 log10, floor 1e-10 / DCT-II, ortho, 13 coefficients']
    ],widths=[CW*.36,CW*.64],row_h=25,size=8.3)
    r.analysis('Each clip yields 98 frames. The MFCC detail omits the larger coefficient 0 so the remaining variation is visible; all 13 coefficients enter feature extraction. The cosine transform summarizes log-mel shape [3].',
               'Time means and standard deviations summarize spectral shape and variation. Five-frame regression deltas add changes across adjacent frames, though summary features discard detailed event order.',
               'The primary analysis uses 52 MFCC summaries. The 80 log-mel summaries form a comparison, and 10 domain descriptors support interpretation. The full 142-row feature catalog gives each name, definition, and rationale.',size=9.4,leading=14)

    r.new('Missing Data, Exact Copies, and Quality Flags')
    r.section('Effective sample sizes after exact-copy removal')
    r.table(['Partition','Phase 1 clips','Removed copies','Retained clips','IDs'],[
        [split,str(before),str(before-s['split_counts'][split]),str(s['split_counts'][split]),str(s['speaker_counts'][split])]
        for split,before in [('train',6290),('validation',865),('test',845)]
    ],widths=[CW*.20,CW*.20,CW*.20,CW*.20,CW*.20],size=8.6)
    r.p('Within each exact PCM-copy group, the first path alphabetically is retained. No copy group crosses a partition or has conflicting command labels. This removes repeated audio without choosing clips by energy or command frequency.')
    r.section('Missingness decisions follow the variable type')
    r.table(['Condition','Handling / observed outcome'],[
        ['Missing audio / unreadable signal','Stop and inspect; no synthetic waveform filling; 0 observed'],
        ['Missing path, label, ID, or split','Stop; identity and target cannot be inferred safely; 0 observed'],
        ['Non-finite derived audio feature','Stop and inspect the extraction; 0 observed'],
        ['Short but valid waveform','Right-zero padding; retain original duration'],
        ['Low energy, DC, or endpoint flags','Retain flags for review; no automatic exclusion']
    ],widths=[CW*.42,CW*.58],size=8.4,row_h=29)
    r.p(f"The retained corpus includes {int(kept.low_energy.sum())} low-energy clips, {int(kept.high_dc.sum())} clips flagged for DC offset, and {int(kept.saturated_sample_count.gt(0).sum())} clips with endpoint PCM samples. These initial screens prioritize review; they are not calibrated corruption labels.",size=9.6)
    r.p('Padding represents the absent tail of a valid short recording. It is separate from imputing unknown measurements. A blanket mean-filling operation is therefore unnecessary for this complete, finite feature table.',size=9.6)

    r.new('PCA: Compact Coordinates from Training Data')
    r.figure('pca_variance',max_h=235)
    r.table(['Quantity','Computed value'],[
        ['Primary input','52 MFCC summaries; nonconstant columns retained'],
        ['Scaler','Training mean and population standard deviation'],
        ['Retained principal components',str(s['primary_pca_components'])],
        ['Retained / first-two variance',f"{s['retained_variance']:.2%} / {s['pc1_variance']+s['pc2_variance']:.2%}"]
    ],widths=[CW*.5,CW*.5],size=8.8)
    r.analysis(f"{s['primary_pca_components']} components retain {s['retained_variance']:.2%} of standardized training variance. The first two retain {s['pc1_variance']+s['pc2_variance']:.2%}.",
               'The two-dimensional map is a visual summary of a larger feature space. Clustering uses every retained component, without whitening, rather than just the plotted coordinates [4].',
               'Reuse the fitted variance filter, scaler, and PCA for validation and future modeling. The fitted objects and coordinates are exported; fitting is never repeated on validation or test.')

    r.new('Two Views of the Same Training Feature Space')
    r.figure('pca_commands_clusters',max_h=290)
    r.analysis('All training commands overlap substantially in the first two principal components. The cluster coloring divides the same coordinates into two broad groups.',
               'The clusters were obtained without command labels. Their separation in this view does not imply separation of the eight words, and the two axes represent only about one quarter of total standardized variance.',
               'Use full-space cluster metrics and post-hoc association tables to assess the pattern. For command recognition, compare supervised representations and per-command errors in Phase 3.')
    r.p('Points are partially transparent to show overlap. The same training clips, coordinate system, and fitted PCA are used in both panels.',size=9.2,leading=14)

    r.new('Speaker and Recording-Level Relationships')
    r.figure('pca_energy_speakers',max_h=260)
    r.table(['Partition','Command AMI','Speaker AMI','Energy-quartile AMI'],[
        [split,*[f"{associations.loc[(split,var),'adjusted_mutual_information']:.3f}" for var in ['command','speaker_id','energy_quartile']]]
        for split in ['train','validation']
    ],widths=[CW*.25]*4,size=8.6)
    r.analysis('Recording level varies across the map. The highlighted IDs show selected repeated-speaker contributions; the numeric association table includes every ID.',
               f"Training command AMI is {associations.loc[('train','command'),'adjusted_mutual_information']:.3f}, compared with {associations.loc[('train','energy_quartile'),'adjusted_mutual_information']:.3f} for energy quartiles. Speaker AMI is modest, and the many small ID groups limit a simple visual comparison.",
               'Keep the speaker-disjoint evaluation design and analyze performance by recording level. Energy is a measured amplitude proxy; independent noise labels and SNR estimates are unavailable in mini.')

    r.new('Selecting the Number of Clusters')
    r.figure('cluster_selection',max_h=235)
    selected_rows = [2,3,4,8,12]
    r.table(['k','Mean silhouette','SD across seeds','Seed stability ARI'],[
        [str(k),f"{search.loc[k,'silhouette_mean']:.4f}",f"{search.loc[k,'silhouette_std']:.4f}",f"{search.loc[k,'seed_ari_mean']:.3f}"]
        for k in selected_rows
    ],widths=[CW*.12,CW*.30,CW*.28,CW*.30],size=8.6)
    r.analysis(f"k={s['selected_k']} has the highest mean silhouette in the predefined range 2-12. Each candidate uses 20 initializations at seeds 7, 21, and 42; silhouette uses the same 1,200 path-hash-selected training clips.",
               'Silhouette measures within- versus between-cluster distances. The selected value is a coarse grouping under this representation and search range, rather than an estimate of the number of command classes [5].',
               'Keep k=2 for the documented primary analysis. Inertia, Davies-Bouldin, and Calinski-Harabasz are exported as supporting diagnostics; class agreement is evaluated only after selection.',size=9.4,leading=14)

    r.new('What the Two Clusters Represent')
    r.figure('cluster_profiles',max_h=150)
    r.analysis(f"Median prepared RMS is {profiles.loc[0,'rms_dbfs']:.2f} dBFS in C0 and {profiles.loc[1,'rms_dbfs']:.2f} dBFS in C1. C0 also has greater median spectral flatness and a higher centroid.",
               'C0 is a quieter, spectrally flatter group; C1 is louder and less flat. These are descriptive recording/spectral profiles, not verified noise categories.',
               'Keep these measurements as diagnostics and test whether a future classifier depends too strongly on recording conditions.',size=9.4,leading=14)
    r.figure('cluster_command_composition',max_h=150)
    r.analysis('Both clusters contain every command. The relative shares differ, with up more common in C0, but neither row concentrates on a single command.',
               'The low command AMI is consistent with mixed command composition. Stable clustering can describe nuisance variation instead of the semantic target.',
               'Carry the acoustic feature pipeline into supervised comparisons, while evaluating command recognition with class-level metrics.',size=9.4,leading=14)

    r.new('Stability Across Seeds, Speakers, and Scaling')
    r.figure('cluster_stability',max_h=220)
    r.table(['Experiment','Result'],[
        ['Three initialization seeds: mean pairwise ARI',f"{s['seed_ari_mean']:.4f}"],
        ['80% speaker subsets: mean anchor ARI',f"{s['speaker_subsample_ari_mean']:.4f}"],
        ['80% speaker subsets: minimum / maximum',f"{stability.iloc[:5].anchor_ari.min():.4f} / {stability.iloc[:5].anchor_ari.max():.4f}"],
        ['Robust scaling: anchor ARI',f"{stability.iloc[-1].anchor_ari:.4f}"],
        ['Anchor population','Same 1,500 path-selected training clips']
    ],widths=[CW*.65,CW*.35],size=8.5,row_h=27)
    r.analysis('The primary split is almost unchanged across initialization seeds and remains similar when the entire chain is refitted on five different subsets containing 80% of training IDs.',
               'Robust scaling changes assignments more noticeably. Initialization stability therefore does not make the result independent of preprocessing. ARI compares partitions while allowing cluster-number permutations.',
               'Keep preprocessing fixed when comparing later models, and report scaling or feature ablations alongside performance. The stability runs support repeatability, not semantic correctness.')

    r.new('Representation and Feature Sensitivity')
    r.figure('representation_comparison',max_h=220)
    r.table(['Representation','Features','PCs','Silhouette','ARI vs primary'],[
        [name,str(int(row.features)),str(int(row.pca_components)),f'{row.silhouette:.3f}',f'{row.ari_to_primary:.3f}']
        for name,row in zip(['MFCC / standard','MFCC / robust','Log-mel / standard','MFCC without c0'],compare.itertuples())
    ],widths=[CW*.40,CW*.13,CW*.12,CW*.17,CW*.18],size=8.2,row_h=25)
    r.analysis('At the fixed selected k, log-mel summaries produce a higher silhouette but much weaker command agreement and very different assignments. Removing MFCC coefficient 0 also changes the grouping.',
               'A better unsupervised separation is not necessarily a better representation for word recognition. The coefficient-0 ablation suggests that level-related information influences the primary pattern, although it is not its only source.',
               'Retain MFCC and log-mel as candidates for the later supervised comparison. These diagnostic runs use the selected k and do not retrospectively replace the predefined primary analysis.',size=9.4,leading=14)

    r.new('Controlled Training-Only Augmentation')
    r.figure('controlled_augmentation',max_h=230)
    r.table(['Control / outcome','Value'],[
        ['Training clips perturbed',str(len(aug))],
        ['Zero-filled time shift / requested gain','Up to +/-80 ms / uniform -3 to +3 dB'],
        ['Energy retention / amplitude headroom','At least 99% before gain / peak at most 0.99'],
        ['Unchanged cluster / ARI',f"{s['augmentation_same_cluster_fraction']:.1%} / {s['augmentation_ari']:.3f}"]
    ],widths=[CW*.55,CW*.45],size=8.6,row_h=26)
    r.analysis('The ledger records each actual shift, requested and effective gain, retained energy, and cluster assignment before and after the operation.',
               'Most assignments persist under these limited perturbations, while some cross the cluster boundary. This experiment measures cluster sensitivity on training examples, not command-classification accuracy or added-noise robustness.',
               'The augmentation function rejects validation and test inputs. Perturbed copies are not added to this clustering fit; a later training experiment can use the same controls without contaminating evaluation.',size=9.4,leading=14)

    r.new('Representative Training Examples')
    r.figure('cluster_representatives',max_h=260)
    r.table(['Cluster','Command','File','RMS (dBFS)'],[
        [f'C{row.cluster}',row.label,Path(row.path).name,f'{row.rms_dbfs:.2f}'] for row in cases.itertuples()
    ],widths=[CW*.12,CW*.15,CW*.53,CW*.20],size=7.9,row_h=23)
    r.analysis('Examples are ranked by distance to their assigned center in the full retained PCA space. The figure shows the closest example from each cluster; the table lists three per cluster.',
               'The common color and amplitude scales allow comparison of recording level and spectral structure. These geometric representatives illustrate the fitted groups and do not establish which words are most typical in everyday speech.',
               'Use the table to choose listening-review candidates before labeling a group as noisy or altering its recordings. No listening-based diagnosis is asserted here.',size=9.3,leading=14)

    r.new('Findings and Decisions for Phase 3')
    r.section('Evidence-supported findings')
    r.p(f"1. Exact-copy removal leaves {s['retained_clips']:,} recordings and preserves all speaker partitions. A repeatable chain prepares every retained waveform and produces finite audio features.")
    r.p(f"2. Standardized MFCC summaries require {s['primary_pca_components']} PCs to retain at least 95% variance. The two-dimensional map is useful for exploration but omits most of the fitted space.")
    r.p(f"3. k=2 is preferred by the specified training silhouette search. Its modest silhouette ({s['primary_silhouette_mean']:.3f}) and weak command AMI show broad acoustic grouping rather than eight-word separation.")
    r.p('4. Initialization and speaker-subset stability are high, while changes in scaling and representation affect assignments. Stability and task usefulness must be assessed separately.')
    r.section('Practical continuation')
    r.p('Use the retained speaker-disjoint partitions for a classical MFCC classifier and other suitable model families. Fit every learned transformation on train, tune on validation, and keep test for the final comparison. Evaluate Macro-F1, balanced accuracy, class-level precision/recall, and a confusion matrix.')
    r.p('Compare MFCC and log-mel features, an amplitude-related feature ablation, and controlled training augmentation. Inspect low-energy and stop-command errors. Noise robustness needs a documented external noise source and an explicit SNR protocol.')
    r.section('Limits')
    r.p('The corpus contains isolated English commands, lacks independent noise labels and demographic/device metadata, and has many small speaker groups. Spectral summaries discard temporal detail. No complete listening review or supervised audio-model evaluation is part of this phase.',size=9.6)

    r.new('Reproduction and Delivered Files')
    r.section('Run from the Phase 2 folder')
    for line in ['python -m pip install -r requirements.txt','python src/download_data.py',
                 'python src/run_phase2.py','python src/make_figures.py',
                 'python src/build_report.py','python src/verify_phase2.py','python src/package_phase2.py']:
        r.latin(line,size=8.8,gap=20)
    r.p('The downloader reuses the existing project audio when available. Otherwise it obtains the pinned official archive. The preparation run checks each retained WAV against its Phase 1 content hash before extracting features.',size=9.6)
    r.table(['Location','Contents'],[
        ['configs/phase2.json','Fixed feature, reduction, clustering, and augmentation settings'],
        ['data/processed/','Retained clip identities and the full feature table'],
        ['reports/tables/','Counts, search, associations, stability, and interpretation evidence'],
        ['artifacts/','Fitted training preprocessing, k-means, and PCA coordinates'],
        ['figures/','12 reproducible figures'],
        ['reports/phase2.pdf','This English report'],
        ['README.md / FEATURES.md / AI_USAGE.md','Run instructions, feature rationale, and assistance disclosure']
    ],widths=[CW*.42,CW*.58],size=8.4,row_h=27)
    r.p('The delivery ZIP contains the report, code, settings, tables, fitted objects, figures, documentation, and source snapshot. Raw WAVs, dependency folders, temporary build files, and other phases are excluded.',size=9.6)

    r.new('References and Data Attribution')
    references = [
        ('1. Pete Warden (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. arXiv:1804.03209.',
         'https://arxiv.org/abs/1804.03209','Dataset collection and evaluation background.'),
        ('2. TensorFlow. Simple audio recognition: Recognizing keywords.',
         'https://www.tensorflow.org/tutorials/audio/simple_audio','Official mini archive, audio format, and padding context.'),
        ('3. SciPy. scipy.fft.dct API reference.',
         'https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html','DCT-II and orthonormal transform convention for the custom MFCC implementation.'),
        ('4. scikit-learn. PCA API reference.',
         'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html','Training-fitted principal components and variance-based dimension selection.'),
        ('5. scikit-learn. Clustering user guide.',
         'https://scikit-learn.org/stable/modules/clustering.html','k-means, silhouette, and external cluster-agreement metrics.'),
    ]
    for citation,url,purpose in references:
        r.p(citation,size=9.3,leading=15,gap=5)
        r.link(url,url,size=8.1)
        r.p(purpose+' Accessed 4 November 2026.',size=8.7,color=MUTED,leading=14,gap=20)
    r.section('Attribution and implementation')
    r.p('Speech Commands is attributed to Pete Warden / Google; mini packaging is attributed to TensorFlow. The parent data license is CC BY 4.0. This project implements the stated signal measurements and mel/MFCC transform with NumPy and SciPy; scikit-learn supplies fitted preprocessing and clustering.',size=9.5)
    r.save()
    try:
        temp.replace(target)
    except PermissionError:
        # Some Windows viewers allow writes but deny replacement of an open file.
        # Open without truncating first, so a fully write-locked target stays intact.
        payload = temp.read_bytes()
        with target.open('r+b') as stream:
            stream.seek(0)
            stream.write(payload)
            stream.truncate()
        temp.unlink()
    manifest = {'pdf_file':'reports/phase2.pdf','pages':len(r.pages),'figures':r.figure_number,
                'page_titles':r.pages,'content_bottoms_pt':r.page_bottoms,
                'pdf_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (ROOT/'tmp/report_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Created {target.name}: {len(r.pages)} pages, {r.figure_number} figures.')


if __name__ == '__main__':
    main()

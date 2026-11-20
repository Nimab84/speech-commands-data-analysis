# Phase 2 - Audio Features and Unsupervised Patterns

**Nima Balotf | Data Analysis project | Path D: Speech Commands**

This folder contains the complete Phase 2 work: signal preparation, audio features, dimension reduction, clustering, stability analysis, and the English report. It continues the Phase 1 dataset and speaker-disjoint partitions.

## Results

Exact-copy removal leaves **7,973 clips**: 6,266 train, 864 validation, and 843 test. The scripts extract **142 descriptors**, with 52 MFCC summaries used in the primary analysis. PCA retains 40 components and 95.74% of standardized training variance. A training-only search selects k=2 with mean silhouette 0.122.

The primary clusters are stable across initialization and speaker subsets, but command agreement is weak (training AMI 0.015). Their profiles differ in recording level and spectral shape. These are diagnostic acoustic groups, not trained command predictions. The report interprets every figure and separates unsupervised structure from the later classification question.

## Files

- `reports/phase2.pdf`: the 17-page report, with 12 figures and direct references.
- `FEATURES.md` and `reports/tables/feature_catalog.csv`: feature definitions and rationale.
- `src/`: extraction, preparation, clustering, plotting, report, checks, and packaging scripts.
- `configs/phase2.json`: fixed audio and analysis settings.
- `data/phase1_inventory.csv`: the source identity and split snapshot.
- `data/processed/features.csv`: one feature row per retained clip.
- `reports/tables/`: preparation, duplicate, selection, association, stability, and augmentation evidence.
- `artifacts/`: the fitted preprocessing/k-means bundle and PCA coordinates.
- `AI_USAGE.md`: assistance disclosure.
- `output/phase2_path_d.zip`: the delivery ZIP, created by the packaging command.

## Reproduce

Use Python 3.12. Run from this folder:

```powershell
python -m pip install -r requirements.txt
python src/download_data.py
python src/run_phase2.py
python src/make_figures.py
python src/build_report.py
python src/verify_phase2.py
python src/package_phase2.py
```

The downloader reuses `../data/raw/mini_speech_commands` from the main project if present. Otherwise it downloads and verifies the official archive locally. Every retained WAV is checked against its Phase 1 SHA-256 before extracting features. Raw audio is excluded from the delivery ZIP and GitHub upload.

For a later analysis-only replay, `python src/run_phase2.py --reuse-features` uses the exported feature table. Start a full extraction when audio or feature settings change. On Windows, keep the Python environment and its installed libraries in a path containing Latin characters if native BLAS discovery has trouble with a non-Latin environment path; the project folder itself can retain its Persian parent name.

## Design decisions

The original speaker split is preserved. Exact copies are removed within their existing partition, using a fixed alphabetic representative. Quality flags are retained. No unknown, silence, or independent noise labels are invented. Missing audio/IDs and non-finite features trigger inspection rather than blanket imputation.

PCA and k-means fit training data only. Validation associations are post-hoc; test features are transformed for later use and do not influence selection or figures. Augmentation is restricted to 300 training examples and is used as a sensitivity check, not as additional samples in the clustering fit.

## Delivery and GitHub

The package script creates `output/phase2_path_d.zip` and a separate `output/github_upload` folder. The upload folder contains this entire `phase2` directory plus three updated root documentation files for the existing course repository. The required Git tag is `phase-2`; create or publish it after this phase is uploaded. A local tag does not by itself create the GitHub tag.

## References

- [Warden, Speech Commands (2018)](https://arxiv.org/abs/1804.03209)
- [Official TensorFlow mini tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [SciPy DCT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html)
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html)

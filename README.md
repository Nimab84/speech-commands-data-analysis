# Speech Commands - Phase 1

Nima Balotf

Path D of the Data Analysis project: data quality and exploratory analysis of eight spoken English commands.

## Results

The dataset contains **8,000 WAV files**, **8 command classes**, and **1,750 speaker IDs**. All files were readable. The audit found 822 clips shorter than one second, 374 low-energy clips, and 27 extra exact copies.

The speaker-based split contains 6,290 training, 865 validation, and 845 test clips. Speaker IDs and exact audio copies are separate across these sets.

## Files

- `reports/phase1.pdf`: the official report, including interpretations for all 13 figures, screening-threshold rationale, and three direct references.
- `AI_USAGE.md`: the AI assistance disclosure, included in the submission ZIP.
- `PROJECT_CARD.md`: the problem, research questions, and evaluation plan.
- `src/`: scripts for downloading data, running the audit, and building the report.
- `reports/phase1/`: per-file measurements and summary tables.
- `figures/phase1/`: plots, waveforms, and spectrograms.
- `data/README.md`: data source and measurement definitions.
- `configs/phase1.json`: analysis settings.

## Run

Use Python 3.12 and run these commands from the project folder:

```powershell
python -m pip install -r requirements.txt
python src/download_data.py
python src/audit.py
python src/build_report.py
```

The first data download is about 182 MB. Later runs reuse the downloaded archive. The report uses standard Helvetica fonts.

The report builder writes `reports/phase1.pdf`, matching the course submission path. The submission package is `output/phase1_path_d.zip`; it contains that report, `AI_USAGE.md`, the code, figures, configuration, and measured tables.

## Method

The eight commands are `down`, `go`, `left`, `no`, `right`, `stop`, `up`, and `yes`. The split follows TensorFlow's filename-hash rule, grouping all clips with the same speaker ID together. Distribution and relationship plots use the training set. The audit records quality flags and exact duplicates while keeping the raw recordings unchanged.

The command-comparison figure shows the percentage of training clips shorter than one second in each class; counts and denominators are exported in `reports/phase1/train_short_clip_rates.csv`. Quality cut-offs are initial screening choices for review, not validated corruption standards. Their rationale is documented in the report and `data/README.md`.

Phase 1 covers data exploration and a constant baseline. Audio classification models and added-noise experiments are planned for later phases. The mini dataset has no separate unknown-word or noise-only classes.

## References

- [TensorFlow audio tutorial and mini dataset](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Warden, Speech Commands (2018)](https://arxiv.org/abs/1804.03209)
- [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/speech_commands)

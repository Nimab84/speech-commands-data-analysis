# Data and Measurements

## Source

The project uses the [official TensorFlow mini archive](https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip). It contains 8,000 recordings in eight command folders. The archive README references Speech Commands v0.01 and is preserved in `mini_source_README.md`. The download URL, size, and checksum are saved in `source_manifest.json`.

The original collection used prompted recordings from volunteers and crowdsourced contributors, captured with phone or laptop microphones. The mini subset does not include recording dates, device details, or demographic annotations. See [Warden (2018)](https://arxiv.org/abs/1804.03209).

Speech Commands is released under CC BY 4.0. Attribution: Pete Warden / Google; mini packaging: TensorFlow. Raw audio and the download archive are excluded from the submission ZIP.

## File structure

- `path`: the relative WAV path and unique row key.
- `label`: the command folder name.
- `speaker_id`: the eight-character filename prefix before `_nohash_`.
- `utterance_index`: the recording counter in the filename.
- `split`: train, validation, or test, assigned from the speaker ID.
- `file_sha256` and `pcm_sha256`: checksums used to identify exact copies.

## Audio measurements

PCM16 samples are divided by 32768 for measurement. Duration is the frame count divided by the sample rate.

- `signal_min`, `signal_max`, `peak_abs`: signal amplitude range and peak.
- `rms = sqrt(mean(x**2))`; `rms_dbfs = 20 log10(max(rms, 1e-12))`.
- `dc_offset`: mean signal value.
- `zero_fraction`: proportion of samples equal to zero.
- `saturated_sample_count`: samples equal to -32768 or 32767.
- `low_energy_frame_fraction`: proportion of nonoverlapping 20 ms frames below -50 dBFS RMS.
- `zero_crossing_rate`: proportion of adjacent samples with different signs.

## Quality checks

Flag clips below 0.25 s, RMS below -45 dBFS, absolute DC offset above 0.05, all-zero signals, durations above one second, unexpected formats, and read errors. Clips shorter than one second are valid and will need padding in later preprocessing. Flags are retained for review rather than automatic removal.

The four numeric cut-offs are project-specific initial screening heuristics, not published standards for detecting corruption:

- **Clip RMS below -45 dBFS:** about 0.56% of full-scale RMS. This prioritizes very quiet clips for listening; weak speech may still carry the correct command.
- **Frame RMS below -50 dBFS:** about 0.32% of full scale in a 20 ms frame. This lower threshold summarizes quiet intervals within a clip, separately from the whole-clip screen. It is not a voice-activity or SNR estimate.
- **Duration below 0.25 s:** less than one quarter of the nominal one-second window. This prioritizes unusually short recordings for semantic review; a short utterance can be valid.
- **Absolute DC offset above 0.05:** a mean shift above 5% of full scale. Such a shift can reduce amplitude headroom and motivates waveform review before any correction.

These thresholds have not been calibrated against a labeled quality-review set. Listening review and a sensitivity analysis should guide any later removal or correction. The current audit retains flagged clips and their original labels.

Exact copies occur within the same speaker, class, and split. Before modeling, retain the first path alphabetically in each copy group and record the updated class counts.

## Training short-clip comparison

`reports/phase1/train_short_clip_rates.csv` lists the number of clips with `n_frames < sample_rate_hz`, the total training clips, and the within-class percentage for every command. All readable training clips are included, including flagged clips and exact copies. This percentage replaces the uninformative duration boxplot; energy and quiet-frame boxplots retain their existing summaries.

## Spectrograms

The STFT uses a 400-sample Hann window (25 ms), 160-sample hop (10 ms), and a 512-point FFT. Power is divided by the squared window sum, floored at `1e-10`, and converted with `10 log10`. The display range is -100 to 0 dB. Example selection uses a fixed seed of 42.

## Limits

Exact checksums do not detect all near-duplicates. Energy flags indicate amplitude conditions rather than measured SNR. The mini archive has no separate noise-only, silence, or unknown-word classes. Low-energy clips keep their command labels.

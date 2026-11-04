# Project Card - Path D

Nima Balotf

## 1. Dataset

Google Speech Commands mini: all 8,000 recordings of `down`, `go`, `left`, `no`, `right`, `stop`, `up`, and `yes`.

## 2. Title

Audio command recognition for unseen speakers.

## 3. Problem

A voice interface needs to recognize short commands for direction, movement, stopping, and confirmation. The project compares suitable audio representations and classification methods for speakers outside the training set.

## 4. Units

One WAV clip is one observation and one prediction. The speaker ID in the filename is the grouping unit for the split.

## 5. Target population

Short English command recordings under conditions similar to the dataset, from unseen speaker IDs.

## 6. Research questions

1. Can an audio model outperform a constant baseline on commands from unseen speakers, and how does performance vary on low-energy clips? **Main question.**
2. How do duration, energy, and the proportion of quiet frames vary across commands?
3. Do clusters of spectral features reflect command, speaker, or recording energy?

The first question connects the data audit to later feature extraction, model comparison, and error analysis. The second is explored in Phase 1; the third is planned for Phase 2.

## 7. Target

The command label, taken from the parent folder of each WAV file.

## 8. Baseline

Always predict the most frequent training class, with alphabetical tie-breaking. A later audio baseline will use MFCC summaries and logistic regression.

## 9. Metrics

Macro-F1 is the main metric. Accuracy, balanced accuracy, per-class precision and recall, and a confusion matrix will provide additional detail. Errors involving `stop` and opposing command pairs will be examined separately.

## 10. Split

TensorFlow's SHA-1 rule assigns speaker IDs to train, validation, or test using 80/10/10 portions of hash space. The resulting clip counts are 6,290 / 865 / 845. Comparative EDA uses train; the constant baseline is evaluated on validation. Test is reserved for final model evaluation.

## 11. Leakage controls

Keep all recordings and augmentations from each speaker in one partition. Fit scaling, PCA, and feature selection on train and tune with validation. Paths, IDs, and labels are excluded from prediction features. Inspect exact copies before training.

## 12. Unsupervised analysis

Apply PCA and k-means to training MFCC summaries. Compare cluster quality and stability across several seeds, then examine relationships with command, speaker ID, and energy.

## 13. Audio experiments

Compare the classical baseline with a log-mel audio model. Evaluate unseen speakers, examine classification errors, and test a feature or augmentation ablation. Added-noise experiments will use documented SNR values and separate noise sources for training and evaluation.

## 14. Ethics and limitations

Use speaker IDs only for grouping and aggregate analysis. Attribute Speech Commands to Pete Warden / Google and mini packaging to TensorFlow under CC BY 4.0. The dataset is English-only and lacks demographic and recording-device annotations. Independent noise and unrelated speech are needed to evaluate false activations in a continuous voice interface.

## References

- [Official mini tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Speech Commands paper](https://arxiv.org/abs/1804.03209)
- [TensorFlow split rule](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py)

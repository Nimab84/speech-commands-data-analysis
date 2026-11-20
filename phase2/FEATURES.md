# Audio Features and Rationale

The complete list is in `reports/tables/feature_catalog.csv` (142 rows). The primary unsupervised input is the 52-column MFCC family. Command labels, speaker IDs, paths, partition names, and quality flags are metadata and are excluded from fitted features.

## Signal preparation

Decode PCM16 to floating-point mono, resample to 16 kHz only when needed, subtract the mean of the real signal, then right-pad with zeros to 16,000 samples. Longer signals use a center crop. Original duration and per-file operations are retained. All observed inputs are already 16 kHz mono and none requires cropping. Amplitude normalization is disabled, so recording-level differences can be diagnosed.

## Spectral transform

Apply pre-emphasis `y[t] = x[t] - 0.97*x[t-1]`, preserving the first sample. Use a symmetric 400-sample Hann window, 160-sample hop, 512-point FFT, and no frame centering. There are 98 frames per one-second input. Power is divided by the squared window sum.

Use 40 unnormalized triangular HTK mel bands between 20 and 7,600 Hz. The conversion is `mel = 2595*log10(1+Hz/700)`. Log power is `10*log10(max(power, 1e-10))`. MFCCs are the first 13 coefficients of the orthonormal type-II DCT of the log-mel bands. These are explicit project settings, not an assertion that every MFCC implementation produces identical values.

## Feature families

| Family | Columns | Information retained | Use |
| --- | ---: | --- | --- |
| MFCC | 52 | Time means/stds of coefficients 0-12 and of their regression deltas | Primary PCA/clustering |
| Log-mel | 80 | Time means/stds of each of 40 bands | Alternative representation |
| Domain | 10 | Length, level, amplitude extremes, quiet fraction, crossings, and spectral shape/change | Cluster interpretation |

The delta is a five-frame regression derivative: the weighted differences at offsets one and two, divided by ten, with replicated edge frames. Population standard deviations are used. The summaries are compact but do not preserve the full temporal ordering of a word.

The representation illustration displays coefficients 1-12 on a symmetric color scale. Coefficient 0 is omitted from that image only because its larger magnitude would obscure the other coefficients; it remains in the primary 52-feature representation and is removed only in the explicitly labeled ablation.

The domain descriptors are original duration, prepared-clip RMS dBFS, peak absolute amplitude, fraction of overlapping 25 ms frames below -50 dBFS, zero-crossing rate, mean spectral centroid, mean bandwidth, mean 85% rolloff, mean spectral flatness, and mean spectral flux. Spectral shape descriptors use the pre-emphasized power spectrum. Silent-frame flatness is explicitly zero. The quiet-frame definition differs from Phase 1's nonoverlapping 20 ms screen and is documented separately.

## Missingness and fitting

Missing audio or identity/label/split fields require inspection rather than imputation. Non-finite derived values stop the extraction. Floors handle logarithms and zero denominators explicitly; no observed feature is missing. Short valid recordings are padded, not treated as missing utterances.

Exact PCM copies are removed by retaining the first path alphabetically in each group. Flags for quiet, endpoint, or DC conditions do not cause automatic exclusion. Nonconstant-column selection, StandardScaler, PCA, and k-means are fitted only on the retained training recordings. Validation and test use the stored transformation.

## Unsupervised selection and diagnostics

Search k=2 through 12 with 20 starts at each seed 7, 21, and 42. Select the highest mean silhouette, measured on the same 1,200 path-hash-selected training clips; ties prefer smaller k. PCA retains at least 95% training variance without whitening. Command, speaker, and energy-quartile associations are calculated only after selection. Energy-quartile boundaries come from train.

Stability checks use five 80% speaker subsets, refitting the full chain each time and comparing a shared 1,500-clip training anchor. Robust scaling, log-mel summaries, and removal of MFCC coefficient 0 are additional fixed-k diagnostics. A high silhouette or stable partition does not by itself indicate useful command recognition.

## Controlled augmentation

For 300 fixed training examples, sample a zero-filled shift within +/-80 ms and a requested gain within +/-3 dB. Shifts must retain at least 99% of waveform energy before gain; failed proposals are retried, with identity shift as the fallback. Gain is capped to keep absolute amplitude at or below 0.99. Record the actual operation. Validation and test inputs are rejected by the augmentation function. The perturbations are a sensitivity experiment and are not added to the clustering fit.

## References

- [SciPy DCT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html)
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn clustering and metrics](https://scikit-learn.org/stable/modules/clustering.html)

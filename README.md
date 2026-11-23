# Signal-Processing-for-Stroke-EEG
Reproducible EEG preprocessing pipeline for acute stroke motor imagery data ([Liu et al., 2024](https://doi.org/10.1038/s41597-023-02787-8)), producing both sensor-space epochs and spatially filtered, Euclidean-aligned epochs ready for machine learning and deep learning BCI classification.

## Author
Dr. Giovanna Dal Rì, Medical Doctor and Biomedical Engineer

## Motivation
Stroke-induced cortical lesions introduce substantial inter-subject and inter-trial amplitude variability, baseline drift, and ocular artifacts in electroencephalography (EEG) recordings. This modular preprocessing pipeline standardizes artifact rejection, spatial filtering, and covariance alignment for acute stroke motor imagery EEG to support robust Brain-Computer Interface (BCI) decoding.

## Overview
This repository processes raw BIDS-formatted `.edf` recordings from the Liu et al. (2024) acute stroke motor imagery dataset into ML/DL-ready `.fif` epoch files. For each subject, the pipeline outputs a continuous cleaned recording (`CleanEEG/`), sensor-space epochs (`UnfilteredCleanEpochs/`), spatially filtered and Euclidean-aligned epochs (`EpochsEEG/`), and an audit log (`preprocessing_summary.csv`) detailing retained trials and artifact metrics.

## Pipeline
1. Data Loading & Cohort Filtering: Load `.edf` recordings via MNE-Python (500 Hz). Exclude derived/duplicate subjects (`sub-05`, `sub-15`, `sub-49`, which duplicate `sub-04`, `sub-11`, and `sub-47`) by default to prevent data leakage.
2. Channel Cleaning: Drop the unnamed/empty trailing EDF channel (`''`) prior to spatial coordinate assignment.
3. Referencing & Montage: Re-reference EEG channels to `CPz`, mark `HEOL` and `HEOR` as EOG channels, and apply the standard 10-10 electrode coordinates (`standard_1005`). After referencing, `CPz` was dropped due to being a flat channel.
4. Detrending & Wideband Filtering: Apply linear detrending (`polyorder=1`) to remove slow baseline drifts followed by a zero-phase FIR bandpass filter (`1.0–40.0 Hz`) for ICA stability.
5. Ocular Artifact Removal (ICA): Fit FastICA (`random_state=42`) and automatically identify and remove blink and horizontal eye-movement components correlated with frontal (`FP1`, `FP2`) and EOG (`HEOL`, `HEOR`) channels.
6. Residual Blink Quality Control: Quantify any residual high-amplitude ocular deflections (`> 150 µV`) on `FP1` and `FP2` and log metrics to `preprocessing_summary.csv`.
7. Motor Imagery Bandpass Filtering: Apply a zero-phase FIR bandpass filter (`8.0–30.0 Hz`) to isolate sensorimotor mu ($\mu$) and beta ($\beta$) rhythms.
8. Event Extraction & Epoching: Load motor imagery events (`value == 2`) from `*_events.tsv`, assign binary class labels (`left` vs. `right`), extract `[-0.5, 4.0] s` epochs, apply baseline correction (`[-0.5, 0.0] s`), crop to the `[0.0, 4.0] s` active imagery window, and reject trials exceeding `150 µV` peak-to-peak amplitude.
9. Dual Output & Spatial Alignment:
   - Sensor-Space Checkpoint (`*_sensor-epo.fif`): Clean, baseline-corrected epochs saved in Volts for baseline models and ablation studies.
   - Spatially Filtered & Aligned Checkpoint (`*_csd_ea-epo.fif`): Apply Surface Laplacian (Current Source Density, CSD) to sharpen spatial resolution, followed by unsupervised Euclidean Alignment (EA) to whiten trial covariance matrices across subjects.
  
## Results
The figures below illustrate the pipeline's intermediate and final outputs for a representative subject (with no residual blink artifact after ICA and no trials rejected). They are included for visual validation of the preprocessing stages.

### Trial structure (sensor space)

<img width="1280" height="960" alt="sub-04_right_trial_image" src="https://github.com/user-attachments/assets/1a20489a-6b1d-42f6-b22d-18c42c913af7" />
<img width="1280" height="960" alt="sub-04_left_trial_image" src="https://github.com/user-attachments/assets/97c4ebef-6bf3-4f2a-9b95-f150c90c7f38" />

These trial x time images show the mean signal over C3/C4 across all 20 trials of each class, after band-pass filtering (8-30 Hz) and ICA-based ocular artifact removal. The amplitude range (approximately ±10-15 µV) is consistent with clean, artifact-free sensorimotor EEG.

### Effect of spatial filtering and alignment

<img width="2400" height="800" alt="sub-04_left_sensor_vs_csd" src="https://github.com/user-attachments/assets/39100f9f-bcc9-472c-a854-a625c7f371bf" />

This comparison shows the grand-average signal before (sensor space, µV, on the left) and after (Current Source Density + Euclidean Alignment, on the right) spatial processing. 

During validation, inspecting this comparison surfaced a methodological issue worth noting explicitly: the surface Laplacian (CSD) transform introduces an exact rank deficiency in the per-subject trial covariance matrix — a structural property of the transform on a 27-channel montage, not numerical noise — which caused Euclidean Alignment's whitening step to disproportionately amplify that single near-zero-variance direction. This was diagnosed by inspecting the eigenvalue spectrum of the covariance matrix before and after CSD, and resolved by replacing the original fixed-constant regularization with an eigenvalue-floored pseudo-inverse (eigenvalues below a fraction of the matrix's largest eigenvalue are floored before inversion), scaled relative to each subject's own data rather than a fixed absolute constant. 

The CSD output appears visually denser than the sensor-space trace, independent of the unit change from µV to mV/m². This reflects the surface Laplacian's intended behavior of emphasizing local, high-spatial-frequency structure rather than a residual artifact.

## Tech Stack
* Language: Python 3.9+
* Dependencies: See `requirements.txt` (`mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`)

## Materials
* Dataset: Liu, H., Wei, P., Wang, H., et al. (2024). *An EEG motor imagery dataset for brain computer interface in acute stroke patients*. Scientific Data, 11, 131. https://doi.org/10.1038/s41597-023-02787-8
* Duplicate Subject Exclusion: In the public release, subjects `05`, `15`, and `49` are derived duplicates of subjects `04`, `11`, and `47`. They are excluded (`exclude_derived=True`) to avoid bias in cross-subject evaluation.

## License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

Under this license, you are free to use, modify, and distribute the code for academic or commercial purposes, provided that original authorship is attributed and modified files retain notices of changes.

## Citation
If you use this code, the signal processing part or any other part, please cite it as follows:

> Dal Rì, 2026. *Signal Processing for Stroke EEG* (Version 1.0.0) [Computer software]. GitHub. (https://github.com/DrGiovannaDalRi/Signal-Processing-for-Stroke-EEG)

**BibTeX:**
```bibtex
@software{DalRi2026SignalProcessing,
  author = {Dal Rì, Giovanna},
  title = {Signal Processing for Stroke EEG},
  year = {2026},
  version = {v1.0.0},
  url = {(https://github.com/DrGiovannaDalRi/Signal-Processing-for-Stroke-EEG)
}

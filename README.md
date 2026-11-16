# Signal-Processing-for-Stroke-EEG

## Author
Dr. Giovanna Dal Rì, Medical Doctor and Biomedical Engineer

## Motivation
This signal processing script was built in order to read EEG data from stroke patients. The final aim is to implement a Brain Computer Interface applied to stroke patients.

## Pipeline
1. Load EEG data via MNE
2. Detrending (remove slow drift trends)
3. Bandpass filtering (8-30HZ)
4. Interpolate invalid or empty channels
5. Wavelet denoising (db4 wavelet)
6. Set reference (CPz channel)
7. Set montage (10-10 standard montage)
8. ICA Artifact removal
    - FP1, FP2 (frontal blink detectors)
    - HEOL, HEOR (horizontal EOG detectors)
9. Blink detection and removal (remaining on FP1 and FP2 after ICA), using 150 µV threshold
10. Load events of interest
11. Assign class labels
12. Extract epochs of interest (from events of interest)
13. Baseline correction and optimization
14. Large Laplacian Filtering
15. Matching Filtering

## Description
The script contains the signal processing of EEG data of stroke patients, contained in the dataset from Liu et al. (2024). The EEG data are loaded from a raw format and processed into a format ready to be read and used in the training and testing of a Machine Learning/Deep Learning algorithm for classification. The final aim is to implement a Brain Computer Interface able to process raw EEG data from stroke patients.

The raw EEG data of all subjects are first loaded via MNE by defining the extraction parameters. During loading the raw EEG data go through signal processing. 

Firstly, a linear detrending filter (polynomial order = 1) was applied, to remove slow drift trends. Moreover, a bandpass filter was applied between 8 and 30Hz.

Invalid or empty channels were detected and interpolated.

A wavelet denoising was applied for db4.

The reference was set to the CPz channel and the montage was set to the standard 10-10 montage, as stated by the database documentation. 




8. ICA Artifact removal
    - FP1, FP2 (frontal blink detectors)
    - HEOL, HEOR (horizontal EOG detectors)
9. Blink detection and removal (remaining on FP1 and FP2 after ICA), using 150 µV threshold
10. Load events of interest
11. Assign class labels
12. Extract epochs of interest (from events of interest)
13. Baseline correction and optimization
14. Large Laplacian Filtering
15. Matching Filtering


## Tech Stack
* Language: Python 3.10+
* Libraries: see requirements.txt

## Materials
* EEG data is proprietary of Liu et al, (2024), so it cannot be published in this repository. It can be easily found online.

## License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

Under this license, you are free to use, modify, and distribute the code for academic or commercial purposes, provided that original authorship is attributed and modified files retain notices of changes.

## Citation
If you use this code, the signal processing part or any other part, please cite it as follows:

> Dal Rì, 2026. *Signal Processing for Stroke EEG* (Version 1.0.0) [Computer software]. GitHub. [https://github.com/DrGiovannaDalRi/Automatic-Bibliometric-Study](https://github.com/DrGiovannaDalRi/Signal-Processing-for-Stroke-EEG)

**BibTeX:**
```bibtex
@software{DalRi2026SignalProcessing,
  author = {Dal Rì, Giovanna},
  title = {Signal Processing for Stroke EEG},
  year = {2026},
  version = {v1.0.0},
  url = {[[(https://github.com/DrGiovannaDalRi/Automatic-Bibliometric-Study)](https://github.com/DrGiovannaDalRi/Signal-Processing-for-Stroke-EEG)]
}

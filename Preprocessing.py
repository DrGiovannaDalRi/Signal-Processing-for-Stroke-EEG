###SIGNAL PROCESSING FOR ACUTE STROKE IMAGINARY MOVEMENT EEG DATA (Liu, 2024)

"""
Description:
    Reproducible and modular preprocessing pipeline for 29-channel
    motor imagery EEG recordings from acute stroke patients (500 Hz, CPz reference,
    10-10 montage).

Pipeline:
    1. Reference & Montage   : Re-reference to CPz, mark EOG channels (HEOL/HEOR), apply 10-10 montage.
    2. Channel Cleaning      : Interpolate known bad channels via spherical splines.
    3. Drift & Wideband Filt : Linear detrending + zero-phase FIR bandpass (1.0-40.0 Hz).
    4. Ocular Artifact (ICA) : FastICA decomposition with automated EOG/frontal blink removal.
    5. Blink QC & Channel Drop: Residual blink peak check on FP1/FP2; drop EOG/frontal channels.
    6. Narrowband Filtering  : Zero-phase FIR bandpass (8.0-30.0 Hz) isolating mu/beta rhythms.
    7. Epoching & Rejection  : Extract [-0.5, 4.0] s trials, baseline-correct [-0.5, 0.0] s,
                               crop to [0.0, 4.0] s, and reject trials > 150 uV peak-to-peak.
    8. Spatial & Alignment   : Surface Laplacian (CSD) followed by Euclidean Alignment (EA).

Database reference:
    Liu, H., Wei, P., Wang, H., et al. (2024). An EEG motor imagery dataset for brain
    computer interface in acute stroke patients. Scientific Data, 11, 131.
    https://doi.org/10.1038/s41597-023-02787-8

"""

##IMPORT LIBRARIES
import mne
from mne.utils import set_log_level

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.linalg import fractional_matrix_power
import logging
import random

#DEFINE VARIABLES
# Known derived/duplicate subjects in Liu et al. (2024)
DERIVED_SUBJECTS = {"05", "15", "49"}
#Montage
MONTAGE = "standard_1005"
#Sampling Frequency
SAMPLINGFREQUENCY = 500
#Reference channel
REFERENCE = "CPz"

#Nominal EEG channels per dataset = 29 (CPz is the separate reference, dropped after use)
#FP1/FP2 (2 of the 29) are also dropped after blink-artifact detection → 27 channels in final saved files
#Channel count of EOG = 2
#Ground channel = "FPz"

#DEFINE MODULE-LEVEL LOGGER
# Configured once by setup_environment()
logger = logging.getLogger(__name__)

##SETTING UP - RANDOM SEEDS, MNE, LOGGING
def setup_environment(seed: int = 42, log_file: str = "preprocessing.log") -> logging.Logger:
    """Configure deterministic random seeds, MNE verbosity, and logging.
    Seeds Python's and NumPy's random number generators so that any stochastic step in the pipeline is
    reproducible across runs, silences routine MNE console output, and wires up a logger that writes to both the console and a log file."""

    random.seed(seed)
    np.random.seed(seed)
    set_log_level(verbose="WARNING")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)

##LOAD STROKE EEG DATA
class EEGLoader:
    """Orchestrates the full per-subject preprocessing pipeline.

    Wires together the individual processing steps (reference/montage, filtering, ICA, blink QC, epoching, spatial filtering, normalization)
    and runs them in sequence for every subject in the dataset, saving intermediate and final outputs to disk and collecting a QC summary."""

    def __init__(self, root, refmont=None, cleaner=None, filterer=None, ica =None, blink = None, bandpass = None, event_loader=None, epoching = None, largeLaplacian = None, normalization = None, exclude_derived = True):
        self.root = Path(root)
        self.refmont = refmont
        self.cleaner = cleaner
        self.filterer = filterer
        self.ica = ica
        self.blink = blink
        self.bandpass = bandpass
        self.event_loader = event_loader
        self.epoching = epoching
        self.laplacian = largeLaplacian
        self.normalization = normalization
        #Exclude derived = True --> In Liu2024 3 subjects are derived from other 3 subjects, removing them removes a potential bias.
        self.exclude_derived = exclude_derived

    def load_subject(self, subject):
        """Load the raw EDF recording and annotation-derived events for one subject.

        Builds the subject's file path manually (rather than relying on a BIDS reader)
        because the dataset's subject IDs contain hyphens that BIDS path parsing does not accept."""

        subj_dir = self.root / f"sub-{subject}" / "eeg"

        edf_path = subj_dir / f"sub-{subject}_task-motor-imagery_eeg.edf"

        if not edf_path.exists():
            logger.warning(f"File not found for subject {subject}: {edf_path}")
            return None, None, None

        logger.info(f"Loading {edf_path}")

        # Load raw EEG
        raw = mne.io.read_raw_edf(edf_path, preload=True)

        # Load events from annotations
        events, event_id = mne.events_from_annotations(raw)

        logger.info(f"Loaded subject {subject}")
        logger.info(f"Sampling rate: {raw.info['sfreq']}")
        logger.info(f"Channels: {raw.info['ch_names']}")
        logger.info(f"Events found: {len(events)}")
        logger.info(f"Montage: {raw.get_montage()}")
        logger.info(f"Number of channels: {len(raw.info['ch_names'])}")

        return raw, events, event_id

    def load_all_subjects(self):
        """Run the full preprocessing pipeline for every subject.

        Iterates over the dataset's 50 subjects (skipping derived/duplicate ones when `exclude_derived` is True),
        applies every processing step in sequence, saves intermediate (.fif) outputs for both the cleaned
        continuous data and the epochs before/after spatial filtering and normalization, and writes a per-subject QC summary CSV.

        A failure on one subject is logged and skipped rather than aborting the run."""

        all_data = {}

        clean_dir = Path("CleanEEG")
        unfiltered_dir = Path("UnfilteredCleanEpochs")
        epochs_dir = Path("EpochsEEG")
        clean_dir.mkdir(parents=True, exist_ok=True)
        unfiltered_dir.mkdir(parents=True, exist_ok=True)
        epochs_dir.mkdir(parents=True, exist_ok=True)

        qc_records = []

        for i in range(1, 51):
            subject = f"{i:02d}"

            if self.exclude_derived and subject in DERIVED_SUBJECTS:
                logger.info(f"Skipping derived/duplicate subject {subject}")
                continue

            try:
                raw, events, event_id = self.load_subject(subject)

                if raw is None:
                    continue

                # 1. Clean channels
                raw = self.cleaner.apply(raw)

                # 2. Reference + montage
                raw = self.refmont.apply(raw)

                # 3. Filter (detrending and bandpass between 1-40Hz)
                raw = self.filterer.apply(raw)

                # 4. ICA artifact removal
                raw, last_excluded = self.ica.apply(raw)

                # 4. Remaining blink artifacts detection and removal
                raw, last_blink_counts = self.blink.apply(raw)
                #Remove the FP1, FP2, HEOL and HEOR channels (only useful for ICA and blink detection)
                raw.drop_channels(["FP1", "FP2", "HEOL", "HEOR"], on_missing="ignore")

                # 5. Bandpass filtering between 8-30Hz
                raw = self.bandpass.apply(raw)

                #SAVE the .fif file after cleaning
                clean_path = clean_dir / f"sub-{subject}_raw.fif"
                raw.save(clean_path, overwrite=True)

                # 6. Load events from TSV
                events, event_id = self.event_loader.load(subject)

                # 7. Epoch extraction - epoching
                epochs = self.epoching.extract(raw, events, event_id)

                #Compute stats
                n_total_events = len(events)
                n_left = len(epochs["left"])
                n_right = len(epochs["right"])
                n_kept = len(epochs)

                # SAVE the epochs before filtering/normalization .fif file
                epochs_path = unfiltered_dir / f"sub-{subject}_sensor-epo.fif"
                epochs.save(epochs_path, overwrite=True)

                # --- TEMPORARY DIAGNOSTIC: check if sensor-space covariance is already rank-deficient ---
                sensor_data = epochs.get_data(copy=False)
                sensor_covs = np.array([np.cov(trial) for trial in sensor_data])
                sensor_r_bar = np.mean(sensor_covs, axis=0)
                sensor_eigvals = np.linalg.eigvalsh(sensor_r_bar)
                logger.info(
                    f"Sensor-space eigenvalues — min: {sensor_eigvals.min():.3e}, max: {sensor_eigvals.max():.3e}, ratio: {sensor_eigvals.max() / sensor_eigvals.min():.1f}")
                # --- END DIAGNOSTIC ---

                # 8. Large Laplacian Filtering
                epochs = self.laplacian.apply(epochs)

                # 9. Normalization
                epochs = self.normalization.apply(epochs)

                # SAVE the epochs after filtering/normalization .fif file
                epochs_path = epochs_dir / f"sub-{subject}_csd_ea-epo.fif"
                epochs.save(epochs_path, overwrite=True)

                all_data[subject] = {
                    "raw": raw,
                    "events": events,
                    "event_id": event_id
                }

                #Save file
                qc_records.append({
                    "subject": subject,
                    "ica_excluded": str(last_excluded),
                    "n_ica_removed": len(last_excluded),
                    "residual_blinks_FP1": last_blink_counts.get("FP1", 0),
                    "residual_blinks_FP2": last_blink_counts.get("FP2", 0),
                    "initial_trials": n_total_events,
                    "kept_trials": n_kept,
                    "dropped_trials": n_total_events - n_kept,
                    "left_trials": n_left,
                    "right_trials": n_right,
                    "low_trial_warning": (n_left < 10) or (n_right < 10),
                })


            except Exception as e:

                logger.error(f"Subject {subject} failed: {e}", exc_info=True)
                qc_records.append({
                    "subject": subject,
                    "ica_excluded": None,
                    "n_ica_removed": None,
                    "residual_blinks_FP1": None,
                    "residual_blinks_FP2": None,
                    "initial_trials": None,
                    "kept_trials": None,
                    "dropped_trials": None,
                    "left_trials": None,
                    "right_trials": None,
                    "low_trial_warning": None,
                    "error": str(e),
                })
                continue

        qc_df = pd.DataFrame(qc_records)
        qc_df.to_csv(epochs_dir / "preprocessing_summary.csv", index=False)
        logger.info("Saved QC report to EpochsEEG/preprocessing_summary.csv")
        logger.info(f"Finished loading {len(all_data)} subjects.")

        return all_data

#REMOVE EMPTY CHANNELS
class EEGChannelCleaner:
    """Drop empty or placeholder channels"""

    def __init__(self):
        self.bad_channels = ['']  #'' was found as a placeholder channel

    def apply(self, raw):
        raw.drop_channels(self.bad_channels)
        logger.info("Empty channel dropped")

        return raw

#SET REFERENCE AND MONTAGE
class EEGReferenceMontage:
    """Applies CPz reference and 10-05 montage.
    10-05 is the denser superset that includes the 10-10 montage used by the dataset"""

    def __init__(self, reference=REFERENCE, montage=MONTAGE):
        self.reference = reference
        self.montage = montage

    def apply(self, raw):
        #Set reference as CPz
        logger.info(f"Setting EEG reference: {self.reference}")
        raw.set_eeg_reference(ref_channels=[self.reference])
        #Drop CPz channel since it is flat
        raw.drop_channels([self.reference])

        # Mark EOG channels
        eog_channels = [ch for ch in ["HEOL", "HEOR"] if ch in raw.ch_names]
        if eog_channels:
            logger.info(f"Marking EOG channels: {eog_channels}")
            raw.set_channel_types({ch: "eog" for ch in eog_channels})

        # Apply montage
        montage_obj = mne.channels.make_standard_montage(self.montage)
        raw.set_montage(montage_obj, match_case=False)

        return raw

#SIGNAL PROCESSING: DETRENDING + BANDPASS (1-40HZ)
class EEGFilter:
    """Removes slow drift and applies the wideband (1-40 Hz) filter."""

    def __init__(self, l_freq=1.0, h_freq=40.0, detrend=True, polyorder=1):
        #detrend: whether to remove slow drift trends
        #polyorder: polynomial order for detrending (1 = linear trend)

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.detrend = detrend
        self.polyorder = polyorder

    def apply(self, raw):
        # 1. TRENDLINE REMOVAL
        if self.detrend:
            logger.info(f"Removing trendline (polyorder={self.polyorder})")
            raw.apply_function(mne.filter.detrend, picks="eeg", channel_wise=True, order=self.polyorder)

        # 2. BAND-PASS FILTER
        #This bandpass is needed for better functionality of ICA (artifact detection)
        #Liu2024 is already filtered between 0.5-40Hz, this further steps is anyway needed
        logger.info(f"Applying band-pass filter ({self.l_freq}–{self.h_freq} Hz)")
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)

        return raw

#REMOVE ARTIFACTS WITH ICA
class EEGICA:
    """Removes ocular artifacts via ICA using the FP1/FP2/HEOL/HEOR channels."""

    #ICA artifact removal using:
    # - FP1, FP2 (frontal blink detectors)
    # - HEOL, HEOR (horizontal EOG detectors)

    def __init__(self, n_components=20, method="fastica", random_state=42):
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self.ica = None

    def apply(self, raw):
        # 1. Fit ICA
        self.ica = mne.preprocessing.ICA(
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state
        )

        self.ica.fit(raw)

        # 2. Blink + EOG detection
        blink_channels = [ch for ch in ["FP1", "FP2"] if ch in raw.ch_names]
        eog_channels   = [ch for ch in ["HEOL", "HEOR"] if ch in raw.ch_names]

        artifact_inds = []

        # FP1 / FP2 blink detection
        for ch in blink_channels:
            logger.info(f"Detecting blink components using {ch}")
            inds, scores = self.ica.find_bads_eog(raw, ch_name=ch)
            artifact_inds.extend(inds)

        # HEOL / HEOR horizontal eye movement detection
        for ch in eog_channels:
            logger.info(f"Detecting EOG components using {ch}")
            inds, scores = self.ica.find_bads_eog(raw, ch_name=ch)
            artifact_inds.extend(inds)

        # Remove duplicates
        artifact_inds = list(set(artifact_inds))

        if artifact_inds:
            logger.info(f"ICA components marked for removal: {artifact_inds}")
            self.ica.exclude = artifact_inds
            raw_clean = self.ica.apply(raw.copy())
        else:
            logger.info("No ICA components removed")
            raw_clean = raw

        #Save list of removed ICA indices
        last_excluded = artifact_inds

        return raw_clean, last_excluded

#DETECT BLINK ARTIFACTS (if remaining after ICA)
class EEGBlinkQC:
    """Checks for residual blink activity on FP1/FP2 after ICA.

    Uses amplitude thresholding (150 µV) to detect blink peaks."""

    def __init__(self, threshold=150e-6):
        #threshold: amplitude (150 µV default)

        self.threshold = threshold

    def apply(self, raw):
        """Count residual above-threshold samples on FP1/FP2.

        This is a QC check, not a cleaning step: no data is modified or removed here,
        this only reports how much blink contamination, if any, remains after ICA."""

        blink_channels = [ch for ch in ["FP1", "FP2"] if ch in raw.ch_names]

        if not blink_channels:
            logger.info("FP1/FP2 not found, skipping blink QC.")
            return raw, {}

        last_blink_counts = {}
        for ch in blink_channels:
            fp = raw.copy().pick([ch])
            data = fp.get_data()[0]
            blink_samples = np.where(np.abs(data) > self.threshold)[0]
            last_blink_counts[ch] = int(len(blink_samples))

            logger.info(f"{ch}: {last_blink_counts[ch]} residual blink samples")

        return raw, last_blink_counts

###BANDPASS FILTER
class BandpassFilterer:
    """Applies the narrowband (mu/beta) bandpass filter."""

    def __init__(self, l_freq=8.0, h_freq=30.0):

        self.l_freq = l_freq
        self.h_freq = h_freq

    def apply(self, raw):
        """Bandpass-filter the recording to the mu/beta range, waves associated with Imaginary Movements."""

        logger.info(f"Applying band-pass filter ({self.l_freq}–{self.h_freq} Hz)")
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)

        return raw


#LOAD EVENTS
class EEGEvents:
    """Loads trial events and left/right class labels from the events TSV."""
    #Loads events where value == 2 and assigns class labels based on trial_type:
    #1 = left hand, 2 = right hand.

    def __init__(self, tsv_path, sfreq=500):
        self.tsv_path = Path(tsv_path)
        self.sfreq = sfreq
        self.df = pd.read_csv(self.tsv_path, sep="\t")

        # Event IDs for MNE
        self.event_id = {
            "left": 1,
            "right": 2
        }

    def load(self, subject):
        df = self.df.copy()

        # Convert ms → samples
        df["sample"] = (df["onset"] /1000* self.sfreq).astype(int)

        # Keep only value == 2
        df = df[df["value"] == 2].reset_index(drop=True)

        logger.info(f"Found {len(df)} events with value==2")

        # Build MNE events array
        events = np.zeros((len(df), 3), dtype=int)
        events[:, 0] = df["sample"].to_numpy()
        events[:, 1] = 0
        events[:, 2] = df["trial_type"].to_numpy()  # 1=left, 2=right

        return events, self.event_id

#EXTRACT EPOCHS + BASELINE CORRECTION
class EEGEpochs:
    """Extracts baseline-corrected, artifact-rejected epochs around events."""
    #Extracts epochs around events with value==2 and trial_type labels.
    #Baseline correction is also applied

    def __init__(self, tmin=-0.5, tmax=4.0, baseline=(-0.5, 0.0)):
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline

    def extract(self, raw, events, event_id):

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,   # {"left":1, "right":2}
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            preload=True,
            reject_by_annotation=False,
        )

        epochs.crop(tmin=0.0, tmax=4.0)
        epochs.drop_bad(reject=dict(eeg=150e-6))
        #AutoReject().fit_transform(epochs)

        logger.info(f"Total epochs: {len(epochs)}")
        logger.info(f"Left hand:  {len(epochs['left'])}")
        logger.info(f"Right hand: {len(epochs['right'])}")

        return epochs

#LARGE LAPLACIAN
class EEGLaplacian:
    """Applies the surface Laplacian (current source density) transform."""
    def __init__(self, stiffness=4, lambda2=1e-3):
        self.stiffness = stiffness
        self.lambda2 = lambda2

    def apply(self, epochs):
        logger.info("Applying Surface Laplacian (CSD)...")
        return mne.preprocessing.compute_current_source_density(
            epochs, stiffness=self.stiffness, lambda2=self.lambda2
        )

#NORMALIZATION
class EuclideanAligner:
    """Applies Euclidean Alignment (EA) to reduce cross-subject variability.

    EA re-centers each subject's trial covariances to a common reference,
    which is particularly useful for stroke patients, whose EEG can vary
    substantially from healthy/standard reference distributions.
    """

    def __init__(self, eigval_floor_ratio=1e-3):
        """
        Eigenvalues of the per-subject covariance matrix smaller than eigval_floor_ratio * max_eigenvalue`
        are floored to that value before inversion, to avoid amplifying near-zero-variance directions
        """
        self.eigval_floor_ratio = eigval_floor_ratio
    def apply(self, epochs):

        logger.info("Applying Euclidean Alignment (EA)...")

        data = epochs.get_data(copy=False)  # (n_epochs, n_channels, n_times)
        covs = np.array([np.cov(trial) for trial in data])
        r_bar = np.mean(covs, axis=0)

        eigvals, eigvecs = np.linalg.eigh(r_bar)
        floor = self.eigval_floor_ratio * eigvals.max()
        eigvals_clipped = np.clip(eigvals, floor, None)
        logger.info(
            f"EA covariance eigenvalues — min: {eigvals.min():.3e}, "
            f"max: {eigvals.max():.3e}, floor applied: {floor:.3e}"
        )

        r_inv_sqrt = eigvecs @ np.diag(eigvals_clipped ** -0.5) @ eigvecs.T

        epochs._data = np.einsum("ij,njt->nit", r_inv_sqrt, data)

        return epochs

####MAIN
def main():
    """Run the full preprocessing pipeline over all subjects."""

    #Path to edffile
    root = r"C:\Users\---\edffile"
    #Path to task-motor-imagery_events.tsv
    eventsroot = r"C:\Users\---\task-motor-imagery_events.tsv"

    logger = setup_environment(seed=42)

    refmont = EEGReferenceMontage(reference=REFERENCE, montage=MONTAGE)
    cleaner = EEGChannelCleaner()
    filterer = EEGFilter()
    ica= EEGICA()
    blink = EEGBlinkQC()
    bandpass = BandpassFilterer()

    event_loader = EEGEvents(tsv_path=eventsroot, sfreq=SAMPLINGFREQUENCY)
    epoching = EEGEpochs()

    largeLaplacian = EEGLaplacian()
    normalization = EuclideanAligner()

    loader = EEGLoader(root, refmont=refmont, cleaner=cleaner, filterer=filterer, ica=ica, blink = blink, bandpass = bandpass, largeLaplacian= largeLaplacian, event_loader=event_loader, epoching = epoching, normalization = normalization, exclude_derived=True)
    all_data = loader.load_all_subjects()


if __name__ == "__main__":
    main()

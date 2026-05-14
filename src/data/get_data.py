import torch
import subprocess
import wfdb
import pandas as pd
import ast

from pathlib import Path
from typing import Literal

from scipy.signal import resample
from biosppy.signals import ecg

from sklearn.model_selection import StratifiedGroupKFold, train_test_split

class DataGetter:
    def __init__(self,
                 target_fs: int = 500,
                 dataset: Literal["ptbdb", "ptb-xl", "ludb"] = "ptbdb"):
        self.target_fs = target_fs
        self.dataset = dataset

        self.base_dir = Path("data")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.raw_data_dir = self.base_dir / "raw_data"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        self.processed_data_dir = self.base_dir / "processed_data"
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.original_path = None
        self.data_dir = None
        self.target_url = None

        if self.dataset == "ptbdb":
            self.original_path = self.raw_data_dir / "physionet.org" / "files" / "ptbdb" / "1.0.0"
            self.data_dir = self.processed_data_dir / "ptbdb"
            self.target_url = "https://physionet.org/files/ptbdb/1.0.0/"

        elif self.dataset == "ptb-xl":
            self.original_path = self.raw_data_dir / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
            self.data_dir = self.processed_data_dir / "ptb-xl"
            self.target_url = "https://physionet.org/files/ptb-xl/1.0.3/"
        
        elif self.dataset == "ludb":
            self.original_path = self.raw_data_dir / "physionet.org" / "files" / "ludb" / "1.0.1"
            self.data_dir = self.processed_data_dir / "ludb"
            self.target_url = "https://physionet.org/files/ludb/1.0.1/"

        else:
            raise ValueError(f"dataset must be 'ptbdb' or 'ptb-xl', but got {self.dataset}")

    def run(self):
        if not self.original_path.exists():
            self.download_data_if_needed()
            # self.change_folder_structure()
            self.package_data()
            self.create_metadata()
        else:
            print("Data already exists. Skipping download.")

    def download_data_if_needed(self):
        print("Data not found. Downloading...")
        command = [
            "wget",
            "-r",
            "-N",
            "-c",
            "-np",
            self.target_url
        ]
        subprocess.run(command, cwd=self.raw_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        

    def package_data(self):
        if self.dataset == "ptbdb":
            for item in self.original_path.iterdir():
                    if item.is_dir():
                        dest = self.data_dir / item.name[-3:]
                        dest.mkdir(parents=True, exist_ok=True)
                        for file in item.iterdir():
                            if file.suffix == ".hea":
                                packaged_folder = dest / file.stem
                                packaged_folder.mkdir(parents=True, exist_ok=True)

                                record = wfdb.rdrecord(str(self.original_path / item.name / file.stem))
                                signals = record.p_signal

                                ecg_tensors = torch.from_numpy(signals).float()
                                torch.save(ecg_tensors, packaged_folder / "signals.pt")
        elif self.dataset == "ptb-xl":
            info_df = pd.read_csv(self.original_path / "ptbxl_database.csv")
            
            for patient_id, data in zip(info_df["ecg_id"], info_df["filename_hr"]):
                record = wfdb.rdrecord(str(self.original_path / data))
                signals = record.p_signal
                
                patient_folder = "0"*(5-len(str(patient_id))) + f"{patient_id}"
                patient_folder = self.data_dir / patient_folder
                patient_folder.mkdir(parents=True, exist_ok=True)

                ecg_tensors = torch.from_numpy(signals).float()
                torch.save(ecg_tensors, patient_folder / "signals.pt")
        elif self.dataset == "ludb":
            all_data = self.original_path / "data"
            for record in all_data.iterdir():
                if record.suffix == ".hea":
                    patient_id = record.stem
                    patient_folder = "0" * (3 - len(str(patient_id))) + f"{patient_id}"
                    patient_folder = self.data_dir / patient_folder
                    patient_folder.mkdir(parents=True, exist_ok=True)

                    signals = wfdb.rdrecord(str(self.original_path / "data" / record.stem))
                    signals = signals.p_signal
                    
                    ecg_tensors = torch.from_numpy(signals).float()
                    torch.save(ecg_tensors, patient_folder / "signals.pt")

    def create_metadata(self):
        if self.dataset == "ptbdb":
            metadata_file = self.metadata_dir / f"ptbdb_metadata_{self.target_fs}.csv"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)

            patient_ids = []
            pt_paths = []
            labels = []
            subtypes = []
            r_peak_indexes = []
            target_fs = []
            length = []
            fold = []

            pt_folders = [f for f in self.original_path.iterdir() if f.is_dir()]
            pt_folders.sort(key=lambda x: int(x.name[-3:]))

            for pt_folder in pt_folders:
                patient_id = int(pt_folder.name[-3:])
                records = [f for f in pt_folder.iterdir() if f.suffix == ".hea"]
                records.sort()
                for record in records:
                    # if record.suffix == ".hea":
                    pt_path = self.data_dir / pt_folder.name[-3:]
                    pt_path = pt_path / record.stem / "signals.pt"
                    label, subtype = self._get_label_from_hea(record)

                    signal_url = self.original_path / pt_folder.name / record.stem
                    r_peaks, leng = self._get_r_peak_indices(str(signal_url))
                    for peak in r_peaks:
                        patient_ids.append(patient_id)
                        pt_paths.append(str(pt_path))
                        labels.append(label)
                        subtypes.append(subtype)
                        r_peak_indexes.append(peak)
                        target_fs.append(self.target_fs)
                        length.append(leng)
                        fold.append(0)  # Placeholder for fold assignment

            metadata = {
                "patient_number": patient_ids,
                "pt_path": pt_paths,
                "label": labels,
                "subtype": subtypes,
                "r_peak_indexes": r_peak_indexes,
                "target_fs": target_fs,
                "length": length,
                "fold": fold
            }
            df = pd.DataFrame(metadata)
            df = self.train_val_test_split(df)
            df.to_csv(metadata_file, index=False)

            self.create_folds(df=df, metadata_file=metadata_file, random_state=42, n_splits=5)

        elif self.dataset == "ptb-xl":
            metadata_file = self.metadata_dir / f"ptbxl_metadata_{self.target_fs}.csv"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)

            patient_ids = []
            pt_paths = []
            labels = []
            subtypes = []
            r_peak_indexes = []
            target_fs = []
            length = []
            fold = []

            info_data_path = self.original_path / "ptbxl_database.csv"
            info_df = pd.read_csv(info_data_path)

            scp_statements = self.original_path / "scp_statements.csv"
            label, subtype = self._get_label_from_csv(info_df, str(scp_statements))

            for i in range(len(info_df)):
                signal_url = self.original_path / info_df["filename_hr"].iloc[i]
                r_peaks, leng = self._get_r_peak_indices(str(signal_url))

                for r_peak in r_peaks:
                    patient_ids.append(info_df["ecg_id"].iloc[i])

                    pt_path = self.data_dir / ("0"*(5-len(str(info_df["ecg_id"].iloc[i]))) + f"{info_df['ecg_id'].iloc[i]}") / "signals.pt"
                    pt_paths.append(str(pt_path))

                    labels.append(label.iloc[i])
                    subtypes.append(subtype.iloc[i])
                    r_peak_indexes.append(r_peak)
                    length.append(leng)
                    target_fs.append(self.target_fs)
                    fold.append(0)

            metadata = {
                "patient_number": patient_ids,
                "pt_path": pt_paths,
                "label": labels,
                "subtype": subtypes,
                "r_peak_indexes": r_peak_indexes,
                "target_fs": target_fs,
                "length": length,
                "fold": fold
            }

            df = pd.DataFrame(metadata)
            df = self.train_val_test_split(df)
            df.to_csv(metadata_file, index=False)

            self.create_folds(df=df, metadata_file=metadata_file, random_state=42, n_splits=5)

        elif self.dataset == "ludb":
            metadata_file = self.metadata_dir / f"ludb_metadata_{self.target_fs}.csv"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)

            patient_ids = []
            pt_paths = []

            records = [int(f.stem) for f in (self.original_path / "data").iterdir() if f.suffix == ".hea"]
            records.sort()

            for record in records:
                patient_id = int(record)
                pt_path = self.original_path / "data" / f"{record}"
                
                patient_ids.append(patient_id)
                pt_paths.append(str(pt_path))

            metadata = {
                "patient_number": patient_ids,
                "pt_path": pt_paths,
            }
            df = pd.DataFrame(metadata)
            df.to_csv(metadata_file, index=False)
        else:
            raise ValueError(f"dataset must be 'ptbdb', 'ptb-xl', or 'ludb', but got {self.dataset}")

    def _get_label_from_hea(self, filename):
        # Work for ptbdb dataset. For ptb-xl, we will get labels from the csv file.
        label_mapping = {
            "inferior": "IMI",
            "antero-septal": "ASMI",
            "infero-lateral": "ILMI",
            "antero-lateral": "ALMI",
            "anterior": "AMI",
            "no": "NO",
            "infero-postero-lateral": "IPLMI",
            "postero-lateral": "PLMI",
            "posterior": "PMI",
            "lateral": "LMI",
            "infero-poster-lateral": "IPLMI",
            "infero-latera": "ILMI",  # typo fix
            "antero-septo-lateral": "ASLMI",
            "infero-posterior": "IPMI"
        }

        try:
            with open(filename, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "Diagnose" in line:
                        next_line = lines[i + 1].lower()
                        if "myocardial infarction" in next_line:
                            next_second_line = lines[i + 2].lower()
                            subtype_raw = next_second_line.replace("# acute infarction (localization): ", "").strip()
                            subtype_mapped = label_mapping.get(subtype_raw, subtype_raw.upper())
                            return "MI", subtype_mapped
                        elif "healthy" in next_line:
                            return "Healthy", 0
                        else:
                            return "Non_MI", 0

        except Exception as e:
            raise ValueError(f"Error reading file {filename}: {e}")

    def _get_label_from_csv(self, df: pd.DataFrame, scp_statements_file: str):
        # For ptb-xl dataset, we will get labels from the csv file.        
        df["scp_codes"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x))
        agg_df = pd.read_csv(scp_statements_file, index_col=0)
        agg_df = agg_df[agg_df["diagnostic"] == 1]

        def aggregate_superclass(df):
            tmp = []
            for key in df.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        df["diagnostic_superclass"] = df["scp_codes"].apply(aggregate_superclass)

        df["diagnostic_subclass"] = pd.Series([0] * len(df))
        df["diagnostic_subclass"] = df["diagnostic_subclass"].astype(object)

        for i in range(len(df)):
            idx_label = df.index[i]

            if "MI" in df["diagnostic_superclass"].iloc[i]:
                df.loc[idx_label, "diagnostic_superclass"] = "MI"

                special_MI_subclasses = {
                    "INJIN": "IMI",
                    "INJLA": "AMI",
                    "INJIL": "ILMI",
                    "INJAL": "ALMI",
                    "INJAS": "ASMI",
                }

                subclasses = []
 
                for key in df["scp_codes"].iloc[i].keys():
                    if "MI" in key:
                        subclasses.append(key)
                    elif key in special_MI_subclasses.keys():
                        subclasses.append(special_MI_subclasses[key])

                df.at[idx_label, "diagnostic_subclass"] = subclasses
                # print(i, df["diagnostic_subclass"].iloc[i])
            elif "NORM" in df["diagnostic_superclass"].iloc[i]:
                df.loc[idx_label, "diagnostic_superclass"] = "Healthy"
                df.at[idx_label, "diagnostic_subclass"] = 0
            else:
                df.loc[idx_label, "diagnostic_superclass"] = "Non_MI"
                df.at[idx_label, "diagnostic_subclass"] = 0
        return df["diagnostic_superclass"], df["diagnostic_subclass"]

    def _get_r_peak_indices(self, signal_url: str):
        # signal_url = self.original_path / pt_folder.name / record.stem

        rec_sig = wfdb.rdrecord(str(signal_url))
        signals = rec_sig.p_signal

        standard_leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf',
                          'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        lead_indices = [i for i, name in enumerate(rec_sig.sig_name) if name.lower() in standard_leads]
        signals = signals[:, lead_indices]
        original_fs = rec_sig.fs

        if original_fs != self.target_fs:
            num_samples = int(signals.shape[0] * self.target_fs / original_fs)
            signals = resample(signals, num_samples, axis=0)

        r_peaks = ecg.ecg(signal=signals[:, 0], sampling_rate=self.target_fs,
                          show=False)["rpeaks"]

        length = len(signals)
        return r_peaks, length

    def create_folds(self, df: pd.DataFrame,
                     metadata_file: Path,
                     random_state: int = 42,
                     n_splits: int = 5):
        sgkf = StratifiedGroupKFold(n_splits=n_splits,
                                    random_state=random_state,
                                    shuffle=True)

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df["pt_path"], df["label"], df["patient_number"])):
            df.loc[val_idx, "fold"] = fold_idx
        df.to_csv(metadata_file, index=False)

    def train_val_test_split(self, df: pd.DataFrame):
        patient_numbers = df["patient_number"].unique()
        labels = df.groupby("patient_number")["label"].first().loc[patient_numbers]
        
        train_patients, test_patients = train_test_split(patient_numbers,
                                                         test_size=0.2,
                                                         stratify=labels,
                                                         random_state=42)
        train_patients, val_patients = train_test_split(train_patients,
                                                        test_size=0.1,
                                                        stratify=labels.loc[train_patients],
                                                        random_state=42)
        df["split"] = "train"
        df.loc[df["patient_number"].isin(val_patients), "split"] = "val"
        df.loc[df["patient_number"].isin(test_patients), "split"] = "test"
        return df

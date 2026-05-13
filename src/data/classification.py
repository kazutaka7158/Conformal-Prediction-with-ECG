from unittest import signals

import pandas as pd
import lightning as L
import ecgmentations as E

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Literal

from src.utils.denoise import WaveletDenoising

# Logic hiện tại đang sử dụng 1 sliding window xung quanh mỗi R-peak 
# để tạo ra các mẫu dữ liệu. Tuy nhiên, có thể cân nhắc sử dụng 
# toàn bộ tín hiệu ECG để huấn luyện mô hình, thay vì chỉ tập trung 
# vào các đoạn xung quanh R-peak. Điều này có thể giúp mô hình học 
# được nhiều thông tin hơn từ tín hiệu ECG, bao gồm cả các đặc điểm 
# khác như sóng P và T, cũng như các biến đổi trong tín hiệu mà không 
# nhất thiết phải liên quan đến R-peak.
class ECGClassificationDataset(Dataset):
    def __init__(self,
                 metadata_file: str,
                 fold_list: list=None,
                 sample_before: int=198,
                 sample_after: int=200,
                 cleaned_data: bool=False,
                 transform: E.Sequential=None
                 ):
        self.info = pd.read_csv(metadata_file)
        self.fold_list = fold_list
        self.sample_before = sample_before
        self.sample_after = sample_after
        self.cleaned_data = cleaned_data
        self.transform = transform
        self._signal_cache = {}

        if fold_list is not None:
            self.info = self.info[self.info["fold"].isin(fold_list)].reset_index(drop=True)

        valid_mask = (self.info["r_peak_indexes"] - self.sample_before > 0) & \
                     (self.info["length"] - self.info["r_peak_indexes"] > self.sample_after)

        self.info = self.info[valid_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.info)

    def _load_signal_(self, pt_path: str):
        if pt_path in self._signal_cache:
            return self._signal_cache[pt_path]
        else:
            base_signals = torch.load(pt_path)
            signals = []
            if self.cleaned_data:
                 denoiser = WaveletDenoising(wavelet="db6", threshold="improved")
                 for signal in base_signals.T:
                     signal = denoiser.run(signal)
                     signal = torch.tensor(signal).float()
                     signals.append(signal)
                 signals = torch.stack(signals, dim=1)
            else:
                signals = base_signals.float()

            self._signal_cache[pt_path] = signals
            return signals

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.info.loc[idx]

        rpeak_index = row["r_peak_indexes"]
        pt_path = row["pt_path"]
        signals = self._load_signal_(pt_path)

        if self.transform:
            signals = self.transform(ecg=signals.numpy())["ecg"]
            signals = torch.tensor(signals, dtype=torch.float32).t()
        else:
            signals = signals.t()

        start_index = rpeak_index - self.sample_before
        end_index = rpeak_index + self.sample_after + 1
        heartbeat = signals[:12, start_index:end_index]

        target_len = self.sample_before + self.sample_after + 1
        if heartbeat.shape[1] < target_len:
            heartbeat = pad(heartbeat, (0, target_len - heartbeat.shape[1]))
        else:
            heartbeat = heartbeat[:, :target_len]

        if self.transform:
            heartbeat = self.transform(heartbeat)
        return heartbeat, row

class MIHealthyDataset(ECGClassificationDataset):
    def __init__(self,
                 metadata_file: str,
                 fold_list: list=None,
                 sample_before: int=198,
                 sample_after: int=200,
                 cleaned_data: bool=False,
                 transform=None
                 ):
        super().__init__(metadata_file,
                         fold_list,
                         sample_before,
                         sample_after,
                         cleaned_data,
                         transform)
        self.label_dict = {"Healthy": 0, "MI": 1, "Non_MI": 2}

    def __getitem__(self, idx):
        heartbeat, row = super().__getitem__(idx)
        label = self.label_dict[row["label"]]
        return heartbeat, label, row["patient_number"], row["r_peak_indexes"], row["pt_path"]

class ECGSubtypeDataset(ECGClassificationDataset):
    def __init__(self,
                 metadata_file: str,
                 fold_list: list=None,
                 sample_before: int=198,
                 sample_after: int=200,
                 cleaned_data: bool=False,
                 transform=None
                 ):
        super().__init__(metadata_file,
                         fold_list,
                         sample_before,
                         sample_after,
                         cleaned_data,
                         transform)
        self.label_dict = {
            "IMI": 0, "ASMI": 1, "ILMI": 2, "ALMI": 3,
            "AMI": 4, "NO": 5, "IPLMI": 6, "PLMI": 7,
            "PMI": 8, "LMI": 9, "IPLMI": 6, "ILMI": 2,
            "ASLMI": 10, "IPMI": 11
        }
        self.valid_indices = self.info[
            self.info["subtype"].isin(self.label_dict.keys())
        ].index.tolist()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        heartbeat, row = super().__getitem__(real_idx)
        label = self.label_dict[row["subtype"]]

        return heartbeat, label, row["patient_number"], row["r_peak_indexes"], row["pt_path"]

class ECGClassificationDataModule(L.LightningDataModule):
    def __init__(self,
                 dataset_class,
                 metadata_file: str,
                 train_folds: list,
                 test_folds: list,
                 batch_size: int,
                 num_workers: int,
                 split_ratio: float = 0.9,
                 sample_before: int = 198,
                 sample_after: int = 200,
                 transform = None,
                 use_cleaned_data: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset_class", "transform"])
        self.dataset_class = dataset_class
        self.transform = transform
        self.use_cleaned_data = use_cleaned_data

    def setup(self, stage: Literal["fit", "test"]=None):
        # Setup transform based on training mode
        # transform = self._get_transform()

        if stage == "fit" or stage is None:
            full_dataset = self.dataset_class(
                metadata_file=self.hparams.metadata_file,
                fold_list=self.hparams.train_folds,
                sample_before=self.hparams.sample_before,
                sample_after=self.hparams.sample_after,
                cleaned_data=self.use_cleaned_data,
                transform=self.transform)
            print(len(full_dataset))
            train_size = int(self.hparams.split_ratio * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset,
                                                                [train_size, val_size])
        elif stage == "test" or stage is None:
            self.test_dataset = self.dataset_class(
                metadata_file=self.hparams.metadata_file,
                fold_list=self.hparams.test_folds,
                sample_before=self.hparams.sample_before,
                sample_after=self.hparams.sample_after,
                cleaned_data=self.use_cleaned_data,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
        )

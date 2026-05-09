import numpy as np
import pandas as pd
import lightning as L

import wfdb
import torch

from torch.utils.data import Dataset, DataLoader
from src.utils.transform import Compose

class ECGSegmentationDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform: Compose=None,
                 leads=["i", "ii", "iii", "avr", "avl", "avf",
                        "v1", "v2", "v3", "v4", "v5", "v6"]):
        self.info = self._get_signal_list(dataset)
        self.leads = leads
        self.transform = transform
        self._signal_cache = {}

        self.masks = self._get_masks(dataset, self.info[0].shape[1])
        # print(self.masks.shape)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        signal = self.info[index]
        mask = self.masks[index]

        if self.transform:
            signal, mask = self.transform(signal, mask)

        return signal, mask

    def _get_signal_list(self, dataset):
        signal_list = []

        for path in dataset:
            signal = wfdb.rdrecord(path).p_signal
            signal = signal.T
            for lead in range(signal.shape[0]):
                signal_list.append(signal[lead])

        signal_list = np.array(signal_list)
        signal_list = torch.tensor(signal_list, dtype=torch.float32)
        signal_list = signal_list.unsqueeze(dim=1)
        return signal_list

    def _get_masks(self, dataset, length):
        masks = []

        for path in dataset:
            for lead in self.leads:
                mask = self._create_mask(path, length, lead)
                masks.append(mask)

        masks = np.array(masks)
        masks = torch.tensor(masks, dtype=torch.float32)

        return masks

    def _create_mask(self, signal_path, length, lead):
        annotation = wfdb.rdann(signal_path, extension=lead)
        sample = annotation.sample
        symbol = annotation.symbol
        ann_dct = {
            "p": np.zeros(length),
            "qrs": np.zeros(length),
            "t": np.zeros(length),
            "non_label": np.zeros(length)
        }
        on = None
        for t, sym in zip(sample, symbol):
            if sym == '(':
                on = t
            elif sym == ')':
                off = t
                if on != None:
                    ann_dct[key] += np.array([0]*on + [1]*(off-on+1) + [0]*(4999-off))
                    on = None
            else:
                if sym in {'p','t'}:
                    key = sym
                else:
                    assert(sym == 'N')
                    key = 'qrs'
        assert(np.max(ann_dct['p'] + ann_dct['qrs'] + ann_dct['t']) <= 1)
        ann_dct["non_label"] = 1 - ann_dct['p'] - ann_dct['qrs'] - ann_dct['t']
        
        all_seg_target = np.stack([ann_dct['p'], ann_dct['qrs'], ann_dct['t'], ann_dct['non_label']], axis=1)
        mask = torch.tensor(all_seg_target, dtype=torch.float32).T

        return mask


class ECGSegmentationDataModule(L.LightningDataModule):
    def __init__(self,
                 metadata_file,
                 transform=None,
                 split_ratio: float=0.8,
                 batch_size=32,
                 num_workers=4,
                 seed=42):
        super().__init__()
        self.metadata_file = metadata_file
        self.dataset = pd.read_csv(metadata_file)
        self.transform = transform
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = self.dataset["pt_path"]
        self.dataset = self.dataset.to_numpy()
        np.random.seed(seed)
        np.random.shuffle(self.dataset)

    def setup(self, stage=None):
        train_size = int(self.split_ratio * len(self.dataset))
        val_size = (len(self.dataset) - train_size) // 2

        self.train_dataset = ECGSegmentationDataset(
            self.dataset[:train_size],
            transform=self.transform
        )
        self.val_dataset = ECGSegmentationDataset(
            self.dataset[train_size:train_size+val_size],
            transform=self.transform
        )
        self.test_dataset = ECGSegmentationDataset(
            self.dataset[train_size+val_size:],
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

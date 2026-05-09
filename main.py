from src.data import DataGetter
from src.data.classification import (
    MIHealthyDataset,
    ECGSubtypeDataset,
    ECGClassificationDataModule
)

from src.data.segmentation import (
    ECGSegmentationDataset,
    ECGSegmentationDataModule
)

import shutil
from pathlib import Path
import importlib
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging
)
import torch
import pandas as pd

if __name__ == "__main__":
    # data_getter = DataGetter(dataset="ludb", target_fs=500)
    
    # ptbxl_database = pd.read_csv("data/raw_data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv", index_col="ecg_id")
    # scp_statements = "data/raw_data/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
    
    # data_getter.run()
    # print(label)
    # mih = MIHealthyDataset(metadata_file=Path("data/metadata/ptbxl_metadata_500.csv"),)
    # print(mih[10][0])
    # print(mih[10][0].shape)
    # print(label_dataset[10][0])
    # print(label_dataset[10][0].shape)
    # mih_datamodule = ECGClassificationDataModule(
    #     MIHealthyDataset,
    #     metadata_file=Path("data/metadata/ptbxl_metadata_500.csv"),
    #     train_folds=[0, 1, 2, 3],
    #     test_folds=[4],
    #     num_workers=2,
    #     batch_size=32)
    # mih_datamodule.setup(stage="fit")
    # train_loader = mih_datamodule.train_dataloader()
    
    # ecg_segment = ECGSegmentationDataset(dataset=["data/raw_data/physionet.org/files/ludb/1.0.1/data/1",
    #                                               "data/raw_data/physionet.org/files/ludb/1.0.1/data/2",
    #                                               "data/raw_data/physionet.org/files/ludb/1.0.1/data/3",
    #                                               "data/raw_data/physionet.org/files/ludb/1.0.1/data/4",
    #                                               "data/raw_data/physionet.org/files/ludb/1.0.1/data/5",
    #                                               "data/raw_data/physionet.org/files/ludb/1.0.1/data/6",
    #                                               "data/raw_data/physionet.org/files/ludb/1.0.1/data/7"])
    # ecg_segment._get_masks(dataset=["data/raw_data/physionet.org/files/ludb/1.0.1/data/1",
    #                                "data/raw_data/physionet.org/files/ludb/1.0.1/data/2",
    #                                "data/raw_data/physionet.org/files/ludb/1.0.1/data/3",
    #                                "data/raw_data/physionet.org/files/ludb/1.0.1/data/4",
    #                                "data/raw_data/physionet.org/files/ludb/1.0.1/data/5",
    #                                "data/raw_data/physionet.org/files/ludb/1.0.1/data/6",
    #                                "data/raw_data/physionet.org/files/ludb/1.0.1/data/7"], length=5000)
    # print(ecg_segment[0][0].shape)
    # print(ecg_segment[0][1][0].T)
    # print(ecg_segment[0][1][0].T.shape)
    # print(ecg_segment[0][1].shape)

    ecg_datamodule = ECGSegmentationDataModule(
        metadata_file=Path("data/metadata/ludb_metadata_500.csv"),
        split_ratio=0.9,
        batch_size=32,
        num_workers=4
    )
    ecg_datamodule.setup(stage="fit")
    train_loader = ecg_datamodule.train_dataloader()
    batch = next(iter(train_loader))
    waves, masks = batch[0], batch[1]
    print(waves.shape)
    print(masks.shape)
    # # # print(batch)
    
    model = importlib.import_module("src.model.UNet1D")
    model_class = getattr(model, "ECGUNet3pCGM")
    model_class = model_class(
        lr=1e-3
    )
    
    model_class.eval()
    with torch.no_grad():
        output = model_class(waves)
    print(f"Input shape: {waves.shape}")
    print(f"Output shape: {output.shape}")
    print(masks.shape)
    
    dice = model_class._ECGUNet3pCGM__dice_score(output, masks)
    print(f"Dice Score: {dice}")

    # subtype_dataset = ECGSubtypeDataset(metadata_file=Path("data/metadata/ptbdb_metadata_500.csv"),)
    # print(subtype_dataset[10][0])
    # print(subtype_dataset[10][0].shape)
    # subtype_datamodule = ECGClassificationDataModule(
    #     ECGSubtypeDataset,
    #     metadata_file=Path("data/metadata/ptbdb_metadata_500.csv"),
    #     train_folds=[0, 1, 2, 3],
    #     test_folds=[4],
    #     num_workers=2,
    #     batch_size=32)
    # subtype_datamodule.setup(stage="fit")
    # train_loader = subtype_datamodule.train_dataloader()

    # batch = next(iter(train_loader))
    # heartbeats, labels = batch[0], batch[1]
    
    # print(heartbeats, labels)

    # model_module = importlib.import_module("src.model.ECGTransForm")
    # model_class = getattr(model_module, "ECGTransForm")
    
    # model_module = importlib.import_module("src.model.MCDANN")
    # model_class = getattr(model_module, "MCDANN")

    # model_class = model_class(
    #     num_classes=3,
    #     lr=1e-3
    # )

    # model_class.eval()
    # with torch.no_grad():
    #     output = model_class(heartbeats)
    # print(f"Input shape: {heartbeats.shape}")
    # print(f"Output shape: {output.shape}")
    # print(labels)
    # print(torch.argmax(output, dim=1))

    # checkpoint_callback = ModelCheckpoint(
    #                 dirpath=checkpoints_dir,
    #                 filename=f'{current}-ecg-{{epoch:02d}}-{{{val_metric}:.4f}}',
    #                 save_top_k=1,
    #                 monitor=val_metric,
    #                 mode=mode
    #             )

    # trainer = L.Trainer(
    #     max_epochs = 1,
    #     accelerator="auto",
    #     devices=1,
    #     # callbacks=[checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-3)],
    # )

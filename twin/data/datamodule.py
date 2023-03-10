"""
Lightning datamodule
"""
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class TwinDataModule(LightningDataModule):
    """
    Created dataloaders for provided dataset class
    """

    def __init__(  # pylint: disable=unused-argument
        self, dataset_cl, num_workers=0, batch_size=4
    ) -> None:
        super().__init__()
        self.dataset_cl = dataset_cl
        self.save_hyperparameters()

    def get_dataloader(self, segment: str, **dataloader_params) -> DataLoader:
        """
        Create a dataloader with provided parameters
        :param segment: "train", "val" or "test"
        :param dataloader_params: additional parameters for dataloader
        :return: dataloader
        """
        dataset = self.dataset_cl()
        num_workers = dataloader_params.get("num_workers", self.hparams.num_workers)
        if num_workers is None:
            num_workers = 3

        return DataLoader(
            dataset=dataset,
            batch_size=dataloader_params.get("batch_size", self.hparams.batch_size),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=(segment == "train"),
            shuffle=(segment == "train"),
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create train dataloader
        :return: dataloader
        """
        return self.get_dataloader("train")

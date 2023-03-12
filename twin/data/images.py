import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

from twin.utils.plot_image_grid import image_grid


class Images(Dataset):
    """
    Dataset from images folder and COLMAP camera estimations
    """

    def __init__(self, data_dir="./data/milk", size=512, device="cpu") -> None:
        """

        :param data_dir:
        :param device:
        """
        data_dir = Path(data_dir)
        with open(data_dir / "transforms.json", "r") as fp:
            meta = json.load(fp)

        self.size = size
        self.samples = []

        centre = torch.zeros(3)  # ???
        for frame in meta["frames"]:
            fname = data_dir / frame["file_path"]
            transform = frame["transform_matrix"]

            rotation = transform[:, :3, :3]
            translation = transform[:, :3, 3] + rotation @ centre

            self.samples.append(
                {
                    "image": fname,
                    "R": rotation,
                    "T": translation,
                }
            )

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["image"]).resize(
            (self.size, self.size)
        )  # TODO: if not square

        image = torch.as_tensor(np.array(image)).float() / 255
        return {
            "silhouette": image[..., 3:].squeeze(0),  # alpha only
            "image": image[..., :3].squeeze(0),
            "R": sample["R"],
            "T": sample["T"],
        }

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    dataset = Images(
        data_dir="./data/milk",
    )

    it = iter(dataset)

    silhouette_images = []
    rgb_images = []

    for _ in range(20):
        data = next(it)
        silhouette_images.append(data["silhouette"].cpu().numpy())
        rgb_images.append(data["image"].cpu().numpy())

    image_grid(np.array(rgb_images), rows=4, cols=5, rgb=True)
    plt.show()

    print(np.array(silhouette_images).shape)
    image_grid(np.array(silhouette_images), rows=4, cols=5, rgb=False)
    plt.show()

"""
Debug dataset made from a cow mesh
"""
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    look_at_view_transform,
)
from torch.utils.data import Dataset

from twin.utils.plot_image_grid import image_grid


class Synthetic(Dataset):
    """
    Debug dataset made from a cow mesh
    """

    def __init__(
        self, obj_filename="cow_mesh/cow.obj", data_dir="./data", device="cpu"
    ) -> None:
        """

        :param obj_filename:
        :param data_dir:
        :param device:
        """
        self.obj_filename = Path(data_dir) / obj_filename
        self.mesh = load_objs_as_meshes([self.obj_filename], device=device)

        # We scale normalize and center the target mesh to fit in a sphere of radius 1
        # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
        # to its original center and scale.  Note that normalizing the target mesh,
        # speeds up the optimization but is not necessary!
        verts = self.mesh.verts_packed()
        # N = verts.shape[0]
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        self.mesh.offset_verts_(-center)
        self.mesh.scale_verts_((1.0 / float(scale)))
        self.camera_params = None

        # the number of different viewpoints from which we want to render the mesh.
        num_views = 20

        # Get a batch of viewing angles.
        elev = torch.linspace(0, 360, num_views)
        azim = torch.linspace(-180, 180, num_views)

        # Place a point light in front of the object. As mentioned above, the front of
        # the cow is facing the -z direction.
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # Initialize an OpenGL perspective camera that represents a batch of different
        # viewing angles. All the cameras helper methods support mixed type inputs and
        # broadcasting. So we can view the camera from the a distance of dist=2.7, and
        # then specify elevation and azimuth angles for each viewpoint as tensors.
        self.rot, self.trans = look_at_view_transform(dist=2.7, elev=elev, azim=azim)

        # Define the settings for rasterization and shading. Here we set the output
        # image to be of size 128X128. As we are rendering images for visualization
        # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
        # rasterize_meshes.py for explanations of these parameters.  We also leave
        # bin_size and max_faces_per_bin to their default values of None, which sets
        # their values using heuristics and ensures that the faster coarse-to-fine
        # rasterization method is used.  Refer to docs/notes/renderer.md for an
        # explanation of the difference between naive and coarse-to-fine rasterization.
        raster_settings = RasterizationSettings(
            image_size=128,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Create a Phong renderer by composing a rasterizer and a shader. The textured
        # Phong shader will interpolate the texture uv coordinates for each vertex,
        # sample from a texture image and apply the Phong lighting model
        camera = FoVPerspectiveCameras(
            device=device, R=self.rot[None, 1, ...], T=self.trans[None, 1, ...]
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=camera, lights=self.lights),
        )

        # Silhouette renderer
        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=128,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=50,
        )
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader(),
        )

    def __getitem__(self, index: int):
        camera = FoVPerspectiveCameras(
            R=self.rot[None, index, ...], T=self.trans[None, index, ...]
        )
        images = self.renderer(self.mesh, cameras=camera, lights=self.lights)
        silhouette = self.renderer_silhouette(
            self.mesh, cameras=camera, lights=self.lights
        )

        return {
            "silhouette": silhouette[..., 3:].squeeze(0),  # alpha only
            "image": images[..., :3].squeeze(0),
            "R": self.rot[index, ...],
            "T": self.trans[index, ...],
        }

    def __len__(self):
        return len(self.rot)


if __name__ == "__main__":

    dataset = Synthetic(data_dir="./data", obj_filename="cow_mesh/cow.obj")
    # Image.fromarray(
    #     np.array(255 * dataset.mesh.textures._maps_padded[0]).astype(np.uint8)
    # ).show()

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

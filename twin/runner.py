"""
Lightning module
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.renderer import (
    AmbientLights,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import SGD, Optimizer


class TwinRunner(LightningModule):
    """
    Lightning module which describes training process
    """

    def __init__(self, init_shape_level=4, mesh_only_epochs=500, out="./out"):
        """

        :param args:
        :param kwargs:
        """
        super().__init__()
        self.out = Path(out)
        self.out.mkdir(exist_ok=True)
        self.mesh_only_epochs = mesh_only_epochs

        self.src_mesh = ico_sphere(level=init_shape_level)
        verts_shape = self.src_mesh.verts_packed().shape

        self.mesh: Meshes
        # We will learn to deform the source mesh by offsetting its vertices
        # The shape of the deform parameters is equal to the total number of vertices in
        # src_mesh
        self.deform_verts = torch.full(verts_shape, 0.0, requires_grad=True)

        # We will also learn per vertex colors for our sphere mesh that define texture
        # of the mesh
        self.sphere_verts_rgb = torch.full(
            [1, verts_shape[0], 3], 0.5, requires_grad=True
        )

        self.lights = AmbientLights()  # suboptimal
        rot, trans = look_at_view_transform(dist=5, elev=[0], azim=[0])
        camera = FoVPerspectiveCameras(
            device=self.device, R=rot[None, 0, ...], T=trans[None, 0, ...]
        )

        # Rasterization settings for differentiable rendering, where the blur_radius
        # initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable
        # Renderer for Image-based 3D Reasoning', ICCV 2019
        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=128,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=50,
            perspective_correct=False,
        )

        # Silhouette renderer
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, raster_settings=raster_settings_soft
            ),
            shader=SoftSilhouetteShader(),
        )
        # Differentiable soft renderer using per vertex RGB colors for texture
        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, raster_settings=raster_settings_soft
            ),
            shader=SoftPhongShader(
                device=self.device, cameras=camera, lights=self.lights
            ),
        )

        # Optimize using rendered RGB image loss, rendered silhouette image loss, mesh
        # edge loss, mesh normal consistency, and mesh laplacian smoothing
        self.losses = {
            "rgb": {"weight": 1.0, "values": []},
            "silhouette": {"weight": 1.0, "values": []},
            "edge": {"weight": 1.0, "values": []},
            "normal": {"weight": 0.01, "values": []},
            "laplacian": {"weight": 1.0, "values": []},
        }

    @staticmethod
    def update_mesh_shape_prior_losses(mesh):
        """
        Losses to smooth / regularize the mesh shape and the edge length of the predicted mesh
        :param mesh:
        :return:
        """
        loss = dict()

        loss["edge"] = mesh_edge_loss(mesh)

        # mesh normal consistency
        loss["normal"] = mesh_normal_consistency(mesh)

        # mesh laplacian smoothing
        loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")
        return loss

    def configure_optimizers(
        self,
    ) -> Union[Tuple[List[Optimizer], List[Dict[str, Any]]], Optimizer]:
        optimizer = SGD(
            [self.deform_verts, self.sphere_verts_rgb], lr=1.0, momentum=0.9
        )
        # TODO: scheduler
        return optimizer

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        batch_size = batch["image"].size(0)
        loss = {k: torch.tensor(0.0, device=self.device) for k in self.losses}
        if batch_idx == 0:
            # Deform the mesh
            self.mesh = self.src_mesh.offset_verts(self.deform_verts)

            if self.current_epoch >= self.mesh_only_epochs:
                # Add per vertex colors to texture the mesh
                self.mesh.textures = TexturesVertex(
                    verts_features=self.sphere_verts_rgb
                )

            # Losses to smooth /regularize the mesh shape
            loss.update(**self.update_mesh_shape_prior_losses(self.mesh))

        meshes = self.mesh.extend(batch_size)
        cameras = FoVPerspectiveCameras(device=self.device, R=batch["R"], T=batch["T"])
        if self.current_epoch < self.mesh_only_epochs:
            images_predicted = self.renderer_silhouette(
                meshes, cameras=cameras, lights=self.lights
            )
        else:
            images_predicted = self.renderer_textured(
                meshes, cameras=cameras, lights=self.lights
            )
            # # Squared L2 distance between the predicted RGB image and the target
            # # image from our dataset
            predicted_rgb = images_predicted[..., :3]
            loss_rgb = ((predicted_rgb - batch["image"]) ** 2).mean()
            loss["rgb"] += loss_rgb

        # # Squared L2 distance between the predicted silhouette and the target
        # # silhouette from our dataset
        predicted_silhouette = images_predicted[..., 3:]
        loss_silhouette = ((predicted_silhouette - batch["silhouette"]) ** 2).mean()
        loss["silhouette"] += loss_silhouette

        final_loss = sum([l * self.losses[k]["weight"] for k, l in loss.items()])

        # Plot mesh
        if batch_idx == 0 and self.current_epoch % 20 == 0:
            vis = self.visualize_prediction(
                meshes,
                cameras=cameras,
                target_images=batch["silhouette"].squeeze(-1)
                if self.current_epoch < self.mesh_only_epochs
                else batch["image"],
                silhouette=self.current_epoch < self.mesh_only_epochs,
            )

            Image.fromarray(vis.cpu().numpy().astype(np.uint8)).save(
                self.out / f"vis_{self.current_epoch}.png"
            )

        return final_loss

    def visualize_prediction(
        self, predicted_meshes, target_images, cameras, silhouette=False
    ):
        """
        Show a visualization comparing the rendered predicted mesh to the ground truth mesh
        :param predicted_meshes:
        :param target_images:
        :param cameras:
        :param silhouette:
        :return:
        """
        inds = 3 if silhouette else range(3)
        with torch.no_grad():
            if silhouette:
                predicted_images = self.renderer_silhouette(
                    predicted_meshes, cameras=cameras, lights=self.lights
                )
            else:
                predicted_images = self.renderer_textured(
                    predicted_meshes, cameras=cameras, lights=self.lights
                )
            images = predicted_images[..., inds]
            vis = torch.cat([target_images, images], dim=2)
            vis = 255 * torch.cat(vis.split(1), dim=1).squeeze(0)

            # Image.fromarray(vis.cpu().numpy().astype(np.uint8)).show()
            return vis

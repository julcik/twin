"""
Lightning module
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import save_obj
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
    TexturesUV,
    TexturesVertex,
    look_at_view_transform, BlendParams, PointLights,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere, checkerboard
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import SGD, Optimizer

from twin.utils.mesh import uv_unwrapping


class TwinRunner(LightningModule):
    """
    Lightning module which describes training process
    """

    def __init__(self, init_shape_level=4, mesh_only_epochs=5, out="./out", **kwargs):
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
        self.deform_verts = torch.nn.Parameter(
            torch.full(verts_shape, 0.0), requires_grad=True
        )

        # We will also learn per vertex colors for our sphere mesh that define texture
        # of the mesh
        # self.sphere_verts_rgb = torch.full(
        #     [1, verts_shape[0], 3], 0.5, requires_grad=True
        # )
        self.texture_image = torch.nn.Parameter(
            # torch.zeros(512, 512, 3),
            torch.tensor(np.kron([[1, 0] * 16, [0, 1] * 16] * 16, np.ones((16, 16)))).unsqueeze(-1).repeat((1,1,3)).float(),
            requires_grad=True
        )
        self.register_buffer(
            "verts_uvs",
            kwargs.get(
                "verts_uvs", uv_unwrapping(self.src_mesh.verts_packed())
            ),
        )
        self.register_buffer(
            "faces_uvs", kwargs.get("faces_uvs", self.src_mesh.faces_packed())
        )

        self.lights = AmbientLights()  # suboptimal
        # self.lights = PointLights(ambient_color=[[1.0, 1.0, 1.0]],
        #     diffuse_color=[[0, 0, 0]],
        #     specular_color=[[0, 0, 0]],
        #     location=[[0, 0, 1.0]],).to(self.device)

        rot, trans = look_at_view_transform(dist=5, elev=[0], azim=[0])
        camera = FoVPerspectiveCameras(
            device=self.device, R=rot[None, 0, ...], T=trans[None, 0, ...]
        )

        sigma = 1e-4

        # Silhouette renderer
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=128,
                    blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
                    faces_per_pixel=50,
                    perspective_correct=False,
)
            ),
            shader=SoftSilhouetteShader(),
        )
        # Differentiable soft renderer for texture
        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=128,
                    blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
                    faces_per_pixel=50,
                    perspective_correct=False,

                )
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=camera,
                lights=self.lights,
                blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1, 1, 1.0)),
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

        final_verts, final_faces = self.src_mesh.get_mesh_verts_faces(0)
        save_obj(
            self.out / "model_initial.obj",
            verts=final_verts,
            faces=final_faces,
            verts_uvs=self.verts_uvs,
            faces_uvs=self.faces_uvs,
            texture_map=self.texture_image,
        )

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
        optimizer = SGD([self.deform_verts, self.texture_image], lr=1.0, momentum=0.9)
        # TODO: scheduler
        return optimizer

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        batch_size = batch["image"].size(0)
        losses_dict = {k: torch.tensor(0.0, device=self.device) for k in self.losses}

        # Deform the mesh
        self.mesh = self.src_mesh.offset_verts(self.deform_verts)

        if self.current_epoch >= self.mesh_only_epochs:
            # self.current_epoch == self.mesh_only_epochs:
            #     TODO: recreate UV coords with blender

            # Add per vertex colors to texture the mesh
            # self.mesh.textures = TexturesVertex(
            #     verts_features=self.sphere_verts_rgb
            # )

            # https://github.com/facebookresearch/pytorch3d/issues/1473
            self.mesh.textures = TexturesUV(
                maps=self.texture_image.unsqueeze(0),
                faces_uvs=self.faces_uvs.unsqueeze(0),
                verts_uvs=self.verts_uvs.unsqueeze(0),
            )

        # Losses to smooth /regularize the mesh shape
        losses_dict.update(**self.update_mesh_shape_prior_losses(self.mesh))

        meshes = self.mesh.extend(batch_size)
        cameras = FoVPerspectiveCameras(device=self.device, R=batch["R"], T=batch["T"])

        images_predicted = self.renderer_silhouette(
            meshes, cameras=cameras, lights=self.lights
        )
        # # Squared L2 distance between the predicted silhouette and the target
        # # silhouette from our dataset
        predicted_silhouette = images_predicted[..., 3:]
        loss_silhouette = ((predicted_silhouette - batch["silhouette"]) ** 2).mean()
        losses_dict["silhouette"] += loss_silhouette

        if self.current_epoch >= self.mesh_only_epochs:
            images_predicted = self.renderer_textured(
                meshes, cameras=cameras, lights=self.lights
            )
            # # Squared L2 distance between the predicted RGB image and the target
            # # image from our dataset
            predicted_rgb = images_predicted[..., :3]
            loss_rgb = ((predicted_rgb - batch["image"]) ** 2).mean()
            losses_dict["rgb"] += loss_rgb

        final_loss = sum([l * self.losses[k]["weight"] for k, l in losses_dict.items()])

        # Plot mesh
        if batch_idx == 0: # and self.current_epoch % 20 == 0:
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

            final_verts, final_faces = self.mesh.get_mesh_verts_faces(0)
            Image.fromarray(vis.cpu().numpy().astype(np.uint8)).show()
            if self.mesh.textures is not None:
                Image.fromarray(np.array(255 * self.mesh.textures._maps_padded[0].detach().cpu()).astype(np.uint8)).show()
                Image.fromarray(np.array(255 * self.texture_image.detach().cpu()).astype(np.uint8)).show()
                print(">>>>>",
                      self.texture_image.detach().max(),
                      np.array(255 * self.texture_image.detach().cpu()).astype(np.uint8).max(),
                      (self.texture_image.detach() > 0.5).sum())
            save_obj(
                self.out / f"model_{self.current_epoch}.obj",
                verts=final_verts,
                faces=final_faces,
                verts_uvs=self.verts_uvs,
                faces_uvs=self.faces_uvs,
                texture_map=self.texture_image,
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

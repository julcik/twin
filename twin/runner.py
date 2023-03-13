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
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    look_at_view_transform, HardPhongShader, PointLights,
)
from pytorch3d.structures import Meshes
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import SGD, Optimizer, Adam

from twin.utils.mesh import make_sphere
from torchvision.transforms import functional as ttf
from torch.nn import functional as F

class TwinRunner(LightningModule):
    """
    Lightning module which describes training process
    """

    def __init__(
        self,
        init_shape_level=3,
        mesh_only_epochs=5,
        out="./out",
        size=128,
        camera_params=None,
        **kwargs,
    ):
        # pylint: disable=too-many-instance-attributes, too-many-arguments
        """

        :param args:
        :param kwargs:
        """
        super().__init__()
        self.out = Path(out)
        self.out.mkdir(exist_ok=True)
        self.mesh_only_epochs = mesh_only_epochs
        self.camera_params = {} if camera_params is None else camera_params

        # Initial mesh - UV sphere
        self.src_mesh = make_sphere(level=init_shape_level, device=device)
        verts_shape = self.src_mesh.verts_packed().shape

        # Deformed mesh
        self.mesh: Meshes = None
        self.deform_verts = torch.nn.Parameter(
            torch.full(verts_shape, 0.0), requires_grad=True
        )

        # Texture
        self.texture_image = torch.nn.Parameter(
            torch.full((512, 512, 3), fill_value=0.5), requires_grad=True
        )
        self.register_buffer(
            "verts_uvs",
            kwargs.get("verts_uvs", self.src_mesh.textures._verts_uvs_padded[0]),
        )
        self.register_buffer("faces_uvs", self.src_mesh.textures._faces_uvs_padded[0])

        # Rendering
        # self.lights = AmbientLights().to(device) # suboptimal?
        self.lights_location = torch.nn.Parameter(torch.tensor([[5.0, 5.0, -5.0]], device=device), requires_grad=True)
        self.lights = PointLights(device=device, location=self.lights_location)

        rot, trans = look_at_view_transform(dist=5, elev=[0], azim=[0])
        camera = FoVPerspectiveCameras(
            device=self.device,
            R=rot[None, 0, ...],
            T=trans[None, 0, ...],
            **self.camera_params,
        )

        sigma = 1e-4

        # Silhouette renderer
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=size,
                    blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
                    faces_per_pixel=50,
                    perspective_correct=False,
                ),
            ),
            shader=SoftSilhouetteShader(),
        ).to(device)
        # Differentiable renderer for texture

        blur = 1.e-6 #5.e-5 # TODO: decay?
        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=size,
                    blur_radius=blur,
                    faces_per_pixel=5,
                    perspective_correct=False,
                ),
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=camera,
                lights=self.lights,
                blend_params=BlendParams(
                    sigma=blur, gamma=1e-4, background_color=(1.0, 1.0, 1.0)
                ),
            ),
        ).to(device)

        # Loss weights
        self.losses = {
            "rgb": {"weight": 10.0, "values": []},
            "silhouette": {"weight": 1.0, "values": []},
            "edge": {"weight": 1.0, "values": []},
            "normal": {"weight": 0.01, "values": []},
            "laplacian": {"weight": 1.0, "values": []},
        }

        self.save_textured_mesh(self.out / "model_initial.obj")

    def save_textured_mesh(self, path):
        if self.mesh is not None:
            final_verts, final_faces = self.mesh.get_mesh_verts_faces(0)
        else:
            final_verts, final_faces = self.src_mesh.get_mesh_verts_faces(0)
        save_obj(
            path,
            verts=final_verts,
            faces=final_faces,
            verts_uvs=self.verts_uvs,
            faces_uvs=self.faces_uvs,
            texture_map=F.sigmoid(10*self.texture_image),
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
        # optimizer = SGD([self.deform_verts, self.texture_image], lr=1.0, momentum=0.9)
        optimizer = Adam([self.deform_verts, self.texture_image, self.lights_location], lr=0.01,)
        # TODO: scheduler
        return optimizer

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        batch_size = batch["image"].size(0)
        losses_dict = {k: torch.tensor(0.0, device=self.device) for k in self.losses}

        # Deform the mesh
        self.mesh = self.src_mesh.offset_verts(self.deform_verts)



        # Losses to smooth /regularize the mesh shape
        losses_dict.update(**self.update_mesh_shape_prior_losses(self.mesh))

        meshes = self.mesh.extend(batch_size)
        cameras = FoVPerspectiveCameras(
            device=self.device, R=batch["R"], T=batch["T"], **self.camera_params
        )

        images_predicted = self.renderer_silhouette(
            meshes, cameras=cameras, lights=self.lights
        )
        # # Squared L2 distance between the predicted silhouette and the target
        # # silhouette from our dataset
        predicted_silhouette = images_predicted[..., 3:]
        loss_silhouette = ((predicted_silhouette - batch["silhouette"]) ** 2).mean()
        losses_dict["silhouette"] += loss_silhouette

        if self.current_epoch >= self.mesh_only_epochs:
            if self.current_epoch <= 2*self.mesh_only_epochs and batch_idx == 0:
                with torch.no_grad():
                    self.texture_image.data = ttf.gaussian_blur(self.texture_image.moveaxis(-1,0), (5,5), (3,3)).moveaxis(0,-1)

            # do not try to deform mesh to match texture, detach
            self.mesh = self.src_mesh.offset_verts(self.deform_verts.detach())

            # https://github.com/facebookresearch/pytorch3d/issues/1473
            self.mesh.textures = TexturesUV(
                # to speedup texture training
                maps=F.sigmoid(10*self.texture_image.unsqueeze(0)),
                faces_uvs=self.faces_uvs.unsqueeze(0),
                verts_uvs=self.verts_uvs.unsqueeze(0),
            )

            meshes = self.mesh.extend(batch_size)

            images_predicted = self.renderer_textured(
                meshes, cameras=cameras, lights=self.lights
            )
            # # Squared L2 distance between the predicted RGB image and the target
            # # image from our dataset
            predicted_rgb = images_predicted[..., :3]
            loss_rgb = ((predicted_rgb - batch["image"]) ** 2).mean()
            losses_dict["rgb"] += loss_rgb

        self.log_dict(losses_dict)
        final_loss = sum([l * self.losses[k]["weight"] for k, l in losses_dict.items()])

        # Plot mesh
        if batch_idx == 0 and self.current_epoch % 20 == 0:
            print("self.lights_location",self.lights_location)
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

            self.save_textured_mesh(self.out / f"model_{self.current_epoch}.obj")

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

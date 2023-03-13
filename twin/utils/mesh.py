"""
Mesh related utils
"""
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from torch import Tensor


def make_sphere(level: int = 1, device=None):
    """
    Create UV sphere
    """
    if device is None:
        device = torch.device("cpu")
    if level < 0:
        raise ValueError("level must be >= 0.")
    if level not in [1, 2, 3]:
        raise NotImplementedError

    obj_filename = (
        Path(__file__).parents[1]
        / "data"
        / "sphere"
        / f"uvsphere{'' if level==1 else str(level)}.obj"
    )
    mesh = load_objs_as_meshes([obj_filename], device=device)
    verts, faces = mesh.get_mesh_verts_faces(0)

    # else:
    #     mesh = make_sphere(level - 1, device)
    #     subdivide = SubdivideMeshes()
    #     # TODO: need to subdivide texture also
    #     mesh = subdivide(mesh)
    #     verts = mesh.verts_list()[0]
    #     verts /= verts.norm(p=2, dim=1, keepdim=True)
    #     faces = mesh.faces_list()[0]

    return Meshes(verts=[verts], faces=[faces], textures=mesh.textures)


def uv_unwrapping(points3d: Tensor):
    """
    UV unwrapping for sphere
    https://en.wikipedia.org/wiki/UV_mapping
    imply center is (0,0,0)
    """

    rad = torch.norm(points3d, dim=-1)

    unit = points3d / rad[..., None]
    point_u = torch.atan2(unit[..., 2], unit[..., 0]) / (2 * np.pi)
    point_v = torch.asin(unit[..., 1]) / np.pi

    point_uv = torch.stack([point_u, point_v], dim=-1)

    return point_uv


# if __name__ == "__main__":
#     mesh = make_sphere(0)
#     verts, faces = mesh.get_mesh_verts_faces(0)
#     verts_uvs = mesh.textures._verts_uvs_padded[0]
#     faces_uvs = mesh.textures._faces_uvs_padded[0]
#
#     print(verts.shape, faces.shape, verts_uvs.shape, faces_uvs.shape)
#     print(verts_uvs.shape[0] - verts.shape[0])

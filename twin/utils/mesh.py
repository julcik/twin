"""
Mesh related utils
"""
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from torch import Tensor


def prep_blender_uvunwrap(
    verts: Tensor,
    faces: Tensor,
    simplify: bool = False,
) -> Tuple[Meshes, Tensor, Tensor]:
    """Preprocess marching-cubed mesh with blender to reduce resolution, find uv-mapping
    https://github.com/shubham-goel/ds/blob/2ef898894abb33c82e73790eedf30995ddc9d1c1/src/data/init_shape.py
    """
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        logging.warning(
            f"Got zero sized inputs: {list(verts.shape)} {list(faces.shape)}"
        )
    device = verts.device
    finp = os.path.join(os.getcwd(), "blender-uvunwrap.obj")
    fout = os.path.join(os.getcwd(), "blender-uvunwrap-uv.obj")
    logging.info(f"Saving mesh to {finp}")
    save_obj(finp, verts, faces, decimal_places=10)

    blender_unwrap_file = Path(__file__).parents[2] / "scripts" / "blender_unwrap.py"

    blender_call_cmd = (
        f"blender -b -P {blender_unwrap_file} -- {finp} {fout} {simplify} "
        f"> prep_blender_uvunwrap.out 2>&1"
    )
    logging.info(f"Calling blender: {blender_call_cmd}")
    if os.system(blender_call_cmd) != 0:
        raise ChildProcessError("Blender preprocessing falied")
    logging.info("done.")

    verts, faces, aux = load_obj(fout)
    logging.info(f"loaded mesh: v{list(verts.shape)} f{list(faces.verts_idx.shape)}")
    return (
        Meshes(verts[None].to(device), faces.verts_idx[None].to(device)),
        aux.verts_uvs.to(device) if aux.verts_uvs is not None else None,
        faces.textures_idx.to(device) if faces.textures_idx is not None else None,
    )


def make_sphere(level: int = 1, device=None):
    """
    Create UV sphere
    """
    if device is None:
        device = torch.device("cpu")
    if level < 0:
        raise ValueError("level must be >= 0.")
    if level not in [1,2,3]:
        raise NotImplementedError

    obj_filename = Path(__file__).parents[1] / "data" / "sphere" / f"uvsphere{'' if level==1 else str(level)}.obj"
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

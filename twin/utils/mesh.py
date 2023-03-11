import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from pytorch3d.io import load_obj, save_obj
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
        logging.warn(f"Got zero sized inputs: {list(verts.shape)} {list(faces.shape)}")
    device = verts.device
    finp = os.path.join(os.getcwd(), f"blender-uvunwrap.obj")
    fout = os.path.join(os.getcwd(), f"blender-uvunwrap-uv.obj")
    logging.info(f"Saving mesh to {finp}")
    save_obj(finp, verts, faces, decimal_places=10)

    blender_unwrap_file = Path(__file__).parents[2] / "scripts" / "blender_unwrap.py"

    blender_call_cmd = f"blender -b -P {blender_unwrap_file} -- {finp} {fout} {simplify} > prep_blender_uvunwrap.out 2>&1"
    logging.info(f"Calling blender: {blender_call_cmd}")
    if os.system(blender_call_cmd) != 0:
        raise ChildProcessError("Blender preprocessing falied")
    logging.info(f"done.")

    verts, faces, aux = load_obj(fout)
    logging.info(f"loaded mesh: v{list(verts.shape)} f{list(faces.verts_idx.shape)}")
    return (
        Meshes(verts[None].to(device), faces.verts_idx[None].to(device)),
        aux.verts_uvs.to(device) if aux.verts_uvs is not None else None,
        faces.textures_idx.to(device) if faces.textures_idx is not None else None,
    )


def convert_3d_to_uv_coordinates(X: Tensor):
    """
    UV coordinates for sphere
    X : N,3
    Returns UV: N,2 normalized to [-1, 1]
    U: Azimuth: Angle with +X [-pi,pi]
    V: Inclination: Angle with +Z [0,pi]
    """
    eps = 1e-4
    rad = torch.norm(X, dim=-1).clamp(min=eps)
    theta = torch.acos(
        (X[..., 2] / rad).clamp(min=-1 + eps, max=1 - eps)
    )  # Inclination: Angle with +Z [0,pi]
    phi = torch.atan2(X[..., 1], X[..., 0])  # Azimuth: Angle with +X [-pi,pi]
    vv = (theta / np.pi) * 2 - 1
    uu = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    uv = torch.stack([uu, vv], dim=-1)

    return uv

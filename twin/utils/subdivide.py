import torch
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.renderer.mesh.rasterize_meshes import edge_function
from torch import nn

from twin.utils.mesh import make_sphere


class SubdivideLargeFaces(SubdivideMeshes):

    def __init__(self, meshes=None) -> None:
        super(SubdivideLargeFaces, self).__init__()

        self.precomputed = False
        self._N = -1
        if meshes is not None:
            mesh = meshes[0]
            with torch.no_grad():
                subdivided_faces = self.subdivide_faces(mesh)
                if subdivided_faces.shape[1] != 3:
                    raise ValueError("faces can only have three vertices")
                self.register_buffer("_subdivided_faces", subdivided_faces)
                self.precomputed = True

    def subdivide_faces(self, meshes):
        verts_packed = meshes.verts_packed() # (sum(V_n), 3)
        device = verts_packed.device
        with torch.no_grad():
            edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
            verts_edges = verts_packed[edges_packed] # (sum(E_n), 2, 3)
            v0, v1 = verts_edges.unbind(1)
            edges_lens = (v0 - v1).norm(dim=1, p=2)
            median_len = torch.median(edges_lens)*.99

            faces_packed = meshes.faces_packed()

            face_to_edge = meshes.faces_packed_to_edges_packed() # (sum(F_n), 3)
            need_split_edge0 = edges_lens[face_to_edge[:,0]] > median_len
            need_split_edge1 = edges_lens[face_to_edge[:,1]] > 3*median_len
            need_split_edge2 = edges_lens[face_to_edge[:,2]] > median_len

            faces_packed_to_edges_packed = (
                    face_to_edge + verts_packed.shape[0]
            )


            # Need to process all cases separately (3 or 2 edges split (split 3 in this case), and split 1)
            # Case 0 : all split
            need_split_all = (need_split_edge0 & need_split_edge1) | (need_split_edge0 & need_split_edge2) | (need_split_edge1 & need_split_edge2)
            f0 = torch.stack(
                [
                    faces_packed[need_split_all, 0],
                    faces_packed_to_edges_packed[need_split_all, 2],
                    faces_packed_to_edges_packed[need_split_all, 1],
                ],
                dim=1,
            )
            f1 = torch.stack(
                [
                    faces_packed[need_split_all, 1],
                    faces_packed_to_edges_packed[need_split_all, 0],
                    faces_packed_to_edges_packed[need_split_all, 2],
                ],
                dim=1,
            )
            f2 = torch.stack(
                [
                    faces_packed[need_split_all, 2],
                    faces_packed_to_edges_packed[need_split_all, 1],
                    faces_packed_to_edges_packed[need_split_all, 0],
                ],
                dim=1,
            )
            f3 = faces_packed_to_edges_packed
            subdivided_faces_packed_case0 = torch.cat(
                [f0, f1, f2, f3], dim=0
            )  # (4*sum(F_n), 3)
            # Case 1 : split one
            need_split_only0 = need_split_edge0 & ~need_split_edge1 & ~need_split_edge2
            f0_0 = torch.stack(
                [
                    faces_packed[need_split_only0, 0],
                    faces_packed_to_edges_packed[need_split_only0, 2],
                    faces_packed[need_split_only0, 2],
                ],
                dim=1,
            )
            f1_0 = torch.stack(
                [
                    faces_packed[need_split_only0, 1],
                    faces_packed_to_edges_packed[need_split_only0, 0],
                    faces_packed[need_split_only0, 2],
                ],
                dim=1,
            )
            need_split_only1 = ~need_split_edge0 & need_split_edge1 & ~need_split_edge2
            f0_1 = torch.stack(
                [
                    faces_packed[need_split_only1, 0],
                    faces_packed[need_split_only1, 1],
                    faces_packed_to_edges_packed[need_split_only1, 1],
                ],
                dim=1,
            )
            f2_1 = torch.stack(
                [
                    faces_packed[need_split_only1, 2],
                    faces_packed_to_edges_packed[need_split_only1, 1],
                    faces_packed[need_split_only1, 1],
                ],
                dim=1,
            )
            need_split_only2 = ~need_split_edge0 & ~need_split_edge1 & need_split_edge2
            f1_2 = torch.stack(
                [
                    faces_packed[need_split_only2, 1],
                    faces_packed_to_edges_packed[need_split_only2, 0],
                    faces_packed[need_split_only2, 0],
                ],
                dim=1,
            )
            f2_2 = torch.stack(
                [
                    faces_packed[need_split_only2, 2],
                    faces_packed[need_split_only2, 0],
                    faces_packed_to_edges_packed[need_split_only2, 0],
                ],
                dim=1,
            )
            subdivided_faces_packed_case1 = torch.cat(
                [f0_0, f1_0, f0_1, f2_1, f1_2, f2_2], dim=0
            )

            subdivided_faces_packed = torch.cat(
                [subdivided_faces_packed_case0, subdivided_faces_packed_case1], dim=0
            )

            return subdivided_faces_packed

if __name__ == "__main__":
    mesh = make_sphere(1)
    subdivide = SubdivideLargeFaces()
    mesh = subdivide(mesh)
# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.topology import face_to_subface_idcs


def interior_angles(
    triangles: torch.Tensor,
    signed: bool = False,
    face_normals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the interior angles in a triangle embedded in 3D.

    Args:
        triangles (Tensor[Float, [Bs, F, 3, 3]): Per-triangle vertex positions.
        signed (bool, optional): Whether to use face normals to orient the triangle.
            This can be useful in case of inverted triangles with angles > 180 degrees.
            When `signed=True`, user must specify `face_normals`. Defaults to False.
        face_normals (Tensor[Float, [Bs, F, 3]] | None): Per-triangle normals
            used to orient the triangles. Must be provided when `signed=True`.
            See `iskra.geometry.triangle_normals()`. Defaults to None.

    Raises:
        ValueError: If `signed=True`, but `face_normals=None`.

    Returns:
        Tensor[Float, [Bs, F, 3]]: Interior angles of each corner in mesh.
    """
    # Get vertices opposite the corner vertex:
    idcs: list[tuple[int, ...]] = face_to_subface_idcs(2, 1)
    opposite_vecs = torch.stack([triangles[..., nbh_idx, :] for nbh_idx in idcs], -3)
    vecs = opposite_vecs - triangles[..., :, None, :]
    vecs = torch.nn.functional.normalize(vecs, dim=-1)

    cos_theta = torch.linalg.vecdot(vecs[..., :, 0, :], vecs[..., :, 1, :], dim=-1)
    cross = torch.linalg.cross(vecs[..., :, 0, :], vecs[..., :, 1, :], dim=-1)
    if signed:
        if face_normals is None:
            raise ValueError("face_normals must not be None if signed=True")
        sin_theta = torch.linalg.vecdot(cross, face_normals[:, None, :], dim=-1)
    else:
        sin_theta = torch.linalg.vector_norm(cross, dim=-1)
    angles = torch.atan2(sin_theta, cos_theta)
    return angles

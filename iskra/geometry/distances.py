# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import torch

from iskra.geometry.barycentric import (
    barycentric_interpolate,
    tetrahedron_barycentric_coordinates,
    triangle_barycentric_coordinates,
)
from iskra.geometry.broadcast import (
    atleast_nd,
    broadcast_tensors,
    point_simplex_broadcast,
)
from iskra.geometry.normals import edge_normals, triangle_normals
from iskra.topology import face_to_subface_idcs


def simplex_codim(simplices: torch.Tensor) -> int:
    """Codimension of a set of simplices.

    Returns `Dim - SDim` where `Dim` is the ambient dimension and `SDim` is the
    simplex dimension, i.e., `SDim := S - 1` where `S` is the number of simplex corners.

    Args:
        simplices (Tensor[Float, [Bs, S, Dim]]): Simplices tensor s.t. second to
            last dimension represent corners, last represents coordinates.

    Returns:
        int: Codimension.
    """
    return simplices.shape[-1] - simplices.shape[-2] + 1


def edge_to_line(edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the origin-normal representation of a line passing through edges.

    Args:
        edges (Tensor[Bs, 2, 2]): Set of edges in 2D.

    Returns:
        Tensor[Float, [Bs, 2]]: Line origins, taken to be `edges[..., 0, :]`.
        Tensor[Float, [Bs, 2]]: Line normals.
    """
    origin = edges[..., 0, :]
    normal = edge_normals(edges)
    return origin, normal


def triangle_to_plane(triangles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the origin-normal representation of a plane passing through a triangle.

    Args:
        triangles (Tensor[Bs, 3, 3]): Set of triangles in 3D.

    Returns:
        Tensor[Float, [Bs, 3]]: Plane origins, taken to be `triangles[..., 0, :]`.
        Tensor[Float, [Bs, 3]]: Plane normals.
    """
    origin = triangles[..., 0, :]
    normal = triangle_normals(triangles)
    return origin, normal


def hyperplane_project(
    x: torch.Tensor, origin: torch.Tensor, normal: torch.Tensor
) -> torch.Tensor:
    """Projects points `x` onto (hyper)planes defined in the origin-normal form.

    Can be used to project 2D points onto a line or 3D points onto a plane.
    Useful in tandem with `iskra.edge_to_line()` and `iskra.triangle_to_plane()`.

    !!! tip
        Internally broadcasts `x`, `origin`, `normals` together, meaning it can be used
        to project many points onto a single hyperplane, one point onto many hyperplanes
        or many points to many hyperplanes, all by adding singleton dimensions to the
        batches before calling the function. `Bs3` are the output batch dimensions
        obtained by broadcasting `Bs1` and `Bs2` together.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points in Dim-dimensional space.
        origin (Tensor[Float, [Bs2, Dim]]): Hyperplane origins.
        normal (Tensor[Float, [Bs2, Dim]]): Hyperplane normals.

    Returns:
        Tensor[Float, [Bs3, Dim]]: Closest projection of `x` onto
            its corresponding hyperplane.
    """
    x, origin, normal = broadcast_tensors(x, origin, normal)
    t: torch.Tensor = torch.linalg.vecdot(x - origin, normal)
    return x - t[..., None] * normal


def clamped_length_sqr(
    x: torch.Tensor,
    dim: int | tuple[int, ...] = -1,
    keepdim: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Computes the _squared_ lengths of a set of vectors and then clamps them.

    Helps avoid numerical issues when computing lengths of vectors:

    * The gradient of the square-root tends to infinitey as we get closer to zero.
    * We often wish to divide by squared length, which also explodes around zero.

    Args:
        x (Tensor[Float, [Bs, Dim]]): Set of vectors.
        dim (int | tuple[int, ...]): The dimension(s) along which to compute lengths.
        keepdim (bool): Whether to reduce the selected dimensions
            or to keep them with the length one.
        eps (float): The _squared length_ will be clamped to this minimum value.

    Returns:
        Tensor[Float, [Bs] | [Bs, 1]]: Squared lengths of vectors along dimension dim.
            Last dimension of output is 1 if `keepdim=True`, otherwise it is removed.
    """
    sqr_distance = torch.sum(x * x, dim=dim, keepdim=keepdim)
    return sqr_distance.clamp_min(eps)


def clamped_length(
    x: torch.Tensor,
    dim: int | tuple[int, ...] = -1,
    keepdim: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes the lengths of a set of vectors and then clamps them.

    Helps avoid numerical issues when computing lengths of vectors, such as:

    * The gradient of the square-root tends to infinitey as we get closer to zero.
    * We often wish to divide by length, which also explodes around zero.

    Args:
        x (Tensor[Float, [Bs, Dim]]): Set of vectors.
        dim (int | tuple[int, ...]): The dimension(s) along which to compute lengths.
        keepdim (bool): Whether to reduce the selected dimensions
            or to keep them with the length one.
        eps (float): The _length_ will be clamped to this minimum value.

    Returns:
        Tensor[Any, [Bs] | [Bs, 1]]: Lengths of vectors along dimension dim.
            Last dimension of output is 1 if `keepdim=True`, otherwise it is removed.
    """
    return torch.sqrt(clamped_length_sqr(x, dim=dim, keepdim=keepdim, eps=eps * eps))


def point_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    ord: int | float | str = 2,
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the distance between batches of vectors x_i and y_i.

    Unlike PyTorch's `torch.cdist`, this function works with `torch.func` transforms
    and allows for other norms allowed by `torch.linalg.vector_norm`, at the cost of
    higher memory usage. `torch.cdist` is recommended if memory usage is important.

    Args:
        x (Tensor[Float, [Bs, Dim]]): Batch of `Dim`-dimensional vectors.
        y (Tensor[Float, [Bs, Dim]]): Batch of `Dim`-dimensional vectors.
        ord (int | float | str, optional): Order of p-norm.
            follows same convention as PyTorch's `vector_norm`. Defaults to 2.
        keepdim (bool, optional): Whether to keep the last dimension after reduction.
            Defaults to False.

    Raises:
        ValueError: Tensors x and y must have the same shape.

    Returns:
        Tensor[Float, [Bs] | [Bs, 1]]: Last dimension of output is 1 if `keepdim=True`,
            otherwise it is removed.
    """
    if x.ndim != y.ndim:
        raise ValueError(
            f"Tensors have mismatching number of dimensions: {x.shape} != {y.shape}."
        )
    x, y = broadcast_tensors(x, y)
    if ord == 2:
        # Use specialized routine for 2-norm:
        diff = x - y
        sqr_distance = torch.sum(diff * diff, -1, keepdim=keepdim)
        distance = torch.sqrt(sqr_distance.clamp_min(1e-12))
    elif (isinstance(ord, int) or isinstance(ord, float)) and ord % 2 == 0:
        # Clamp other even L-norms to avoid NaNs:
        sqr_distance = torch.sum((x - y) ** ord, -1, keepdim=keepdim)
        distance = torch.pow(sqr_distance.clamp_min(1e-12), 1 / ord)
    else:
        # For other norms use default PyTorch behavior:
        distance: torch.Tensor = torch.linalg.vector_norm(
            x - y, axis=-1, ord=ord, keepdim=keepdim
        )
    return distance


def edge_project(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Projects a point onto the closest point on an edge.

    !!! tip
        Internally broadcasts `x`, `edges` together, meaning it can be used
        to project many points onto a single edge, one point onto many edges
        or many points to many edges, all by adding singleton dimensions
        to the batches before calling the function. `Bs3` are the output batch
        dimensions obtained by broadcasting `Bs1` and `Bs2` together.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        edges (Tensor[Float, [Bs2, 2, Dim]]): Set of edges.

    Returns:
        Tensor[Float, [Bs3, Dim]]: Closest projection of `x` onto
            its corresponding edge.
    """
    x, edges = point_simplex_broadcast(x, edges)
    origin = edges[..., 0, :]
    edge_vectors = edges[..., 1, :] - edges[..., 0, :]
    length = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True)
    edge_vectors = edge_vectors / (length + 1e-12)

    t = torch.linalg.vecdot((x - origin) / (length + 1e-12), edge_vectors)
    t = torch.clamp(t, min=0, max=1)
    bary = torch.stack([1 - t, t], -1)
    return barycentric_interpolate(edges, bary)


def triangle_project(x: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Projects a point onto the closest point on an triangle.

    !!! tip
        Internally broadcasts `x`, `triangles` together, meaning it can be used
        to project many points onto a single triangle, one point onto many triangles
        or many points to many triangles, all by adding singleton dimensions
        to the batches before calling the function. `Bs3` are the output batch
        dimensions obtained by broadcasting `Bs1` and `Bs2` together.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        triangles (Tensor[Float, [Bs2, 3, Dim]]): Set of triangles.

    Returns:
        Tensor[Float, [Bs3, Dim]]: Closest projection of `x` onto
            its corresponding triangle.
    """
    x, triangles = point_simplex_broadcast(x, triangles)
    if simplex_codim(triangles) > 0:
        # Project onto triangle plane:
        origin, normal = triangle_to_plane(triangles)
        x = hyperplane_project(x, origin, normal)

    idcs = face_to_subface_idcs(2)
    edges = torch.stack([triangles[..., idx, :] for idx in idcs], -3)
    projections = edge_project(x[..., None, :], edges)
    distances = point_dist(x[..., None, :], projections)
    min_distance, min_idx = torch.min(distances, -1, keepdim=True)
    closest_point_shape = min_distance.shape + projections.shape[-1:]
    gather_idx = min_idx[..., None].expand(closest_point_shape)
    closest_edge_point = torch.gather(projections, -2, gather_idx)
    closest_edge_point = closest_edge_point.squeeze(-2)

    # Compute barycentric coordinates and clamp to triangle interior:
    bary, valid = triangle_barycentric_coordinates(x, triangles)
    is_inside = torch.all(bary >= 0, -1) & valid
    x[~is_inside] = closest_edge_point[~is_inside]
    return x


def tetrahedron_project(x: torch.Tensor, tetrahedra: torch.Tensor) -> torch.Tensor:
    """Projects a 3D point onto the closest point in a tetrahedron.

    !!! tip
        Internally broadcasts `x`, `tetrahedra` together, meaning it can be used
        to project many points onto a single tetrahedron, one point onto many tetrahedra
        or many points to many tetrahedra, all by adding singleton dimensions
        to the batches before calling the function. `Bs3` are the output batch
        dimensions obtained by broadcasting `Bs1` and `Bs2` together.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        tetrahedra (Tensor[Float, [Bs2, 4, Dim]]): Set of tetrahedra.

    Raises:
        ValueError: We only support 3D tetrahedra. Feel free to raise an issue
            if you have a valid use case for 4D ones!

    Returns:
        Tensor[Float, [Bs3, Dim]]: Closest projection of `x` onto
            its corresponding tetrahedron.
    """
    x, tetrahedra = point_simplex_broadcast(x, tetrahedra)
    if tetrahedra.shape[-1] != 3 or x.shape[-1] != 3:
        raise ValueError("Only 3D tetrahedra are supported.")

    idcs = face_to_subface_idcs(3)
    triangles = torch.stack([tetrahedra[..., idx, :] for idx in idcs], -3)
    projections = triangle_project(x[..., None, :], triangles)
    distances = point_dist(x[..., None, :], projections)
    min_distance, min_idx = torch.min(distances, -1, keepdim=True)
    closest_point_shape = min_distance.shape + projections.shape[-1:]
    gather_idx = min_idx[..., None].expand(closest_point_shape)
    closest_triangle_point = torch.gather(projections, -2, gather_idx)
    closest_triangle_point = closest_triangle_point.squeeze(-2)

    # Compute barycentric coordinates and clamp to triangle interior:
    bary, valid = tetrahedron_barycentric_coordinates(x, tetrahedra)
    is_inside = torch.all(bary >= 0, -1) & valid
    x[~is_inside] = closest_triangle_point[~is_inside]
    return x


def simplex_project(x: torch.Tensor, simplices: torch.Tensor) -> torch.Tensor:
    """Projects a point onto the closest point on a simplex.

    !!! tip
        Internally broadcasts `x`, `simplices` together, meaning it can be used
        to project many points onto a single simplex, one point onto many simplices
        or many points to many simplices, all by adding singleton dimensions
        to the batches before calling the function. `Bs3` are the output batch
        dimensions obtained by broadcasting `Bs1` and `Bs2` together.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        simplices (Tensor[Float, [Bs2, S, Dim]]): Set of simplices.

    Returns:
        Tensor[Float, [Bs3, Dim]]: Closest projection of `x` onto
            its corresponding simplex.
    """
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        return tetrahedron_project(x, simplices)
    elif n_simplex_verts == 3:
        return triangle_project(x, simplices)
    elif n_simplex_verts == 2:
        return edge_project(x, simplices)
    else:
        raise NotImplementedError(
            "simplex_project only supports edges, triangles, and tetrahedra."
        )


def triangle_udf(x: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Unsigned distance function of a 2D point to the boundary of a triangle.

    !!! tip
        Internally broadcasts `x`, `triangles` together, meaning it can be used
        with many points and a single triangle, one point and many triangles
        or many points and many triangles, all by adding singleton dimensions
        to the batches before calling the function. `Bs3` are the output batch
        dimensions obtained by broadcasting `Bs1` and `Bs2` together.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        triangles (Tensor[Float, [Bs2, 3, Dim]]): Set of triangles.

    Raises:
        ValueError: The point and triangle must be in 2D.

    Returns:
        Tensor[Float, [Bs3]]: Unsigned distance of point to triangle boundary.
    """
    x, triangles = point_simplex_broadcast(x, triangles)
    if triangles.shape[-1] != 2 or x.shape[-1] != 2:
        raise ValueError("Only 2D triangles are supported.")

    idcs = face_to_subface_idcs(2)
    edges = torch.stack([triangles[..., idx, :] for idx in idcs], -3)
    projections = edge_project(x[..., None, :], edges)
    distances = point_dist(x[..., None, :], projections)
    min_distance: torch.Tensor = torch.min(distances, -1, keepdim=False)[0]
    return min_distance


def tetrahedron_udf(x: torch.Tensor, tetrahedra: torch.Tensor) -> torch.Tensor:
    """Unsigned distance function of a 3D point to the boundary of a tetrahedron.

    !!! tip
        Internally broadcasts `x`, `tetrahedra` together, meaning it can be used
        with many points and a single tetrahedron, one point and many tetrahedra
        or many points and many tetrahedra, all by adding singleton dimensions
        to the batches before calling the function. `Bs3` are the output batch
        dimensions obtained by broadcasting `Bs1` and `Bs2` together.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        tetrahedra (Tensor[Float, [Bs2, 4, Dim]]): Set of tetrahedra.

    Raises:
        ValueError: The point and tetrahedron must be in 3D.

    Returns:
        Tensor[Float, [Bs3]]: Unsigned distance of point to tetrahedron boundary.
    """
    x, tetrahedra = point_simplex_broadcast(x, tetrahedra)
    if tetrahedra.shape[-1] != 3 or x.shape[-1] != 3:
        raise ValueError("Only 3D tetrahedra are supported.")

    idcs = face_to_subface_idcs(3)
    triangles = torch.stack([tetrahedra[..., idx, :] for idx in idcs], -3)
    projections = triangle_project(x[..., None, :], triangles)
    distances = point_dist(x[..., None, :], projections)
    min_distance: torch.Tensor = torch.min(distances, -1, keepdim=False)[0]
    return min_distance


def point_edge_dist(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Distance of a point to its closest point on an edge.

    Implements `point_dist(x, edge_project(x, edges))` using `iskra.point_dist`
    and `iskra.edge_project`. See documentation of these two functions
    for more information.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        edges (Tensor[Float, [Bs2, 2, Dim]]): Set of edges.

    Returns:
        Tensor[Float, [Bs3]]: Distance of `x` to closest point on
            its corresponding edge. `Bs3` is the broadcasted shape when
            broadcasting `Bs1` and `Bs2`.
    """
    return point_dist(x, edge_project(x, edges))


def point_triangle_dist(x: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Distance of a point to its closest point on an triangle.

    Implements `point_dist(x, triangle_project(x, triangles))` using `iskra.point_dist`
    and `iskra.triangle_project`. See documentation of these two functions
    for more information.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        triangles (Tensor[Float, [Bs2, 3, Dim]]): Set of triangles.

    Returns:
        Tensor[Float, [Bs3]]: Distance of `x` to closest point on
            its corresponding triangle. `Bs3` is the broadcasted shape when
            broadcasting `Bs1` and `Bs2`.
    """
    return point_dist(x, triangle_project(x, triangles))


def point_tetrahedron_dist(x: torch.Tensor, tetrahedra: torch.Tensor) -> torch.Tensor:
    """Distance of a point to its closest point on an tetrahedron.

    Implements `point_dist(x, tetrahedron_project(x, tetrahedra))` using
    `iskra.point_dist` and `iskra.tetrahedron_project`. See documentation
    of these two functions for more information.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        tetrahedra (Tensor[Float, [Bs2, 4, Dim]]): Set of tetrahedra.

    Returns:
        Tensor[Float, [Bs3]]: Distance of `x` to closest point on
            its corresponding tetrahedron. `Bs3` is the broadcasted shape when
            broadcasting `Bs1` and `Bs2`.
    """
    return point_dist(x, tetrahedron_project(x, tetrahedra))


def point_simplex_dist(x: torch.Tensor, simplices: torch.Tensor) -> torch.Tensor:
    """Distance of a point to its closest point on an tetrahedron.

    Thin dispatcher to one of `iskra.point_edge_dist`, `iskra.point_triangle_dist`,
    or `point_tetrahedron_dist`. See documentation of these functions for more
    information.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        simplices (Tensor[Float, [Bs2, S, Dim]]): Set of simplices.

    Raises:
        NotImplementedError: Only supports edges, triangles, and tetrahedra.

    Returns:
        Tensor[Float, [Bs3]]: Distance of `x` to closest point on
            its corresponding simplex. `Bs3` is the broadcasted shape when
            broadcasting `Bs1` and `Bs2`.
    """
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        return point_tetrahedron_dist(x, simplices)
    elif n_simplex_verts == 3:
        return point_triangle_dist(x, simplices)
    elif n_simplex_verts == 2:
        return point_edge_dist(x, simplices)
    else:
        raise NotImplementedError(
            "point_simplex_dist only supports edges, triangles, and tetrahedra."
        )


def point_dist_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    ord: int | float | str = 2,
) -> torch.Tensor:
    x, y = atleast_nd(2, x, y)
    # TODO: why is this not using point_dist and instead using cdist? Memory?
    return torch.cdist(x, y, p=ord)


def point_edge_dist_matrix(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Pairwise distances between a set of points and a set of edges.

    Args:
        x (Tensor[Float, [Bs1, N, Dim]]): Set of points.
        edges (Tensor[Float, [Bs2, M, 2, Dim]]): Set of edges.

    Returns:
        Tensor[Float, [Bs3, N, M]]: Pairwise point-edge distances.
    """
    (x,) = atleast_nd(2, x)
    (edges,) = atleast_nd(3, edges)
    x, edges = x[..., :, None, :], edges[..., None, :, :, :]
    return point_edge_dist(x, edges)


def point_triangle_dist_matrix(
    x: torch.Tensor, triangles: torch.Tensor
) -> torch.Tensor:
    """Pairwise distances between a set of points and a set of edges.

    Args:
        x (Tensor[Float, [Bs1, N, Dim]]): Set of points.
        triangles (Tensor[Float, [Bs2, M, 3, Dim]]): Set of triangles.

    Returns:
        Tensor[Float, [Bs3, N, M]]: Pairwise point-triangle distances.
    """
    (x,) = atleast_nd(2, x)
    (triangles,) = atleast_nd(3, triangles)
    x, triangles = x[..., :, None, :], triangles[..., None, :, :, :]
    return point_triangle_dist(x, triangles)


def point_tetrahedron_dist_matrix(
    x: torch.Tensor, tetrahedra: torch.Tensor
) -> torch.Tensor:
    """Pairwise distances between a set of points and a set of tetrahedra.

    Args:
        x (Tensor[Float, [Bs1, N, Dim]]): Set of points.
        tetrahedra (Tensor[Float, [Bs2, M, 4, Dim]]): Set of tetrahedra.

    Returns:
        Tensor[Float, [Bs3, N, M]]: Pairwise point-tetrahedron distances.
    """
    (x,) = atleast_nd(2, x)
    (tetrahedra,) = atleast_nd(3, tetrahedra)
    x, tetrahedra = x[..., :, None, :], tetrahedra[..., None, :, :, :]
    return point_tetrahedron_dist(x, tetrahedra)


def point_simplex_dist_matrix(x: torch.Tensor, simplices: torch.Tensor) -> torch.Tensor:
    """Pairwise distances between a set of points and a set of simplices.

    Args:
        x (Tensor[Float, [Bs1, N, Dim]]): Set of points.
        simplices (Tensor[Float, [Bs2, M, S, Dim]]): Set of simplices.

    Raises:
        NotImplementedError: Only supports edges, triangles, and tetrahedra.

    Returns:
        Tensor[Float, [Bs3, N, M]]: Pairwise point-simplex distances.
    """
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 4:
        return point_tetrahedron_dist_matrix(x, simplices)
    elif n_simplex_verts == 3:
        return point_triangle_dist_matrix(x, simplices)
    elif n_simplex_verts == 2:
        return point_edge_dist_matrix(x, simplices)
    else:
        raise NotImplementedError(
            "point_simplex_dist_matrix only supports edges, triangles, and tetrahedra."
        )


def closest_edge(
    x: torch.Tensor, edges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finds the closest edges to a set of points.

    `Bs3` is the shape that results from broadcasting `Bs1` and `Bs2`.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        edges (Tensor[Float, [Bs2, 2, Dim]]): Set of edges.

    Returns:
        Tensor[Float, [Bs3, Dim]]: Closest projection of `x` onto the
            set of edges.
        Tensor[Float, [Bs3]]: Distance of `x` to the closest projection.
        Tensor[Float, [Bs3]]: Index of the edge that was projected on.
    """
    (x,) = atleast_nd(2, x)
    (edges,) = atleast_nd(3, edges)
    x, edges = x[..., :, None, :], edges[..., None, :, :, :]
    projections = edge_project(x, edges)
    distances = point_dist(x, projections)
    closest_distance, prim_idx = torch.min(distances, -1)
    gather_idx = prim_idx[..., None, None].expand(
        *(-1,) * (projections.ndim - 1), projections.shape[-1]
    )
    closest_projection = torch.gather(projections, -2, gather_idx).squeeze(-2)
    return closest_projection, closest_distance, prim_idx


def closest_triangle(
    x: torch.Tensor, triangles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finds the closest triangles to a set of points.

    `Bs3` is the shape that results from broadcasting `Bs1` and `Bs2`.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        triangles (Tensor[Float, [Bs2, 3, Dim]]): Set of triangles.

    Returns:
        Tensor[Float, [Bs3, Dim]]: Closest projection of `x` onto the
            set of triangles.
        Tensor[Float, [Bs3]]: Distance of `x` to the closest projection.
        Tensor[Float, [Bs3]]: Index of the triangle that was projected on.
    """
    (x,) = atleast_nd(2, x)
    (triangles,) = atleast_nd(3, triangles)
    x, triangles = x[..., :, None, :], triangles[..., None, :, :, :]
    projections = triangle_project(x, triangles)
    distances = point_dist(x, projections)
    closest_distance, prim_idx = torch.min(distances, -1)
    gather_idx = prim_idx[..., None, None].expand(
        *(-1,) * (projections.ndim - 1), projections.shape[-1]
    )
    closest_projection = torch.gather(projections, -2, gather_idx).squeeze(-2)
    return closest_projection, closest_distance, prim_idx


def closest_simplex(
    x: torch.Tensor, simplices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finds the closest simplex to a set of points.

    Thin dispatcher to one of `iskra.closest_edge`, `iskra.closest_triangle`.
    See documentation of these functions for more information.
    `Bs3` is the shape that results from broadcasting `Bs1` and `Bs2`.

    Args:
        x (Tensor[Float, [Bs1, Dim]]): Set of points.
        simplices (Tensor[Float, [Bs2, S, Dim]]): Set of simplices.

    Returns:
        Tensor[Float, [Bs3, Dim]]: Closest projection of `x` onto the
            set of simplices.
        Tensor[Float, [Bs3]]: Distance of `x` to the closest projection.
        Tensor[Float, [Bs3]]: Index of the simplex that was projected on.
    """
    assert x.shape[-1] == simplices.shape[-1]

    n_simplex_verts = simplices.shape[-2]
    if n_simplex_verts == 3:
        return closest_triangle(x, simplices)
    elif n_simplex_verts == 2:
        return closest_edge(x, simplices)
    else:
        raise NotImplementedError("closest_simplex only supports edges and triangles.")

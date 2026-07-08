import numpy as np
import pytest
import torch

from iskra.geometry import interior_angles, triangle_normals, vertex_normals
from iskra.topology import face_index


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture
def triangles() -> tuple[torch.Tensor, torch.Tensor]:
    # Same angle triangle
    h = np.sqrt(3) / 2
    verts = torch.tensor(
        [
            [-h, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, +h],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 1, 2], [1, 3, 2]],
    )
    return verts, faces


@pytest.fixture
def triangles_larger() -> tuple[torch.Tensor, torch.Tensor]:
    # Increasing the height of one of the triangles by two
    # increases its area by two and changes angles by some amount :)
    h = np.sqrt(3) / 2
    verts = torch.tensor(
        [
            [-h, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, +2 * h],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 1, 2], [1, 3, 2]],
    )
    return verts, faces


def test_triangle_normals(device, triangles):
    verts, faces = triangles
    verts = verts.to(device)
    faces = faces.to(device)
    triangles = face_index(verts, faces)
    normals = triangle_normals(triangles)
    expected_normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        device=device,
    )
    torch.testing.assert_close(normals, expected_normals)


def test_vertex_normals(device, triangles):
    verts, faces = triangles
    verts = verts.to(device)
    faces = faces.to(device)
    triangles = face_index(verts, faces)

    norm = np.sqrt(0.5**2 + 0.5**2)
    expected_normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [-0.5 / norm, 0.0, 0.5 / norm],
            [-0.5 / norm, 0.0, 0.5 / norm],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    angle_vert_normals = vertex_normals(verts, faces, "angle")
    torch.testing.assert_close(angle_vert_normals, expected_normals)
    # Following should be the same as angle-weighted because
    # two triangles are the same:
    area_vert_normals = vertex_normals(verts, faces, "area")
    torch.testing.assert_close(area_vert_normals, expected_normals)
    graph_vert_normals = vertex_normals(verts, faces, "graph")
    torch.testing.assert_close(graph_vert_normals, expected_normals)


def test_vertex_normals_larger(device, triangles_larger):
    verts, faces = triangles_larger
    verts = verts.to(device)
    faces = faces.to(device)

    norm = np.sqrt(1.0**2 + 0.5**2)
    expected_area_normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [-1.0 / norm, 0.0, 0.5 / norm],
            [-1.0 / norm, 0.0, 0.5 / norm],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    area_vert_normals = vertex_normals(verts, faces, "area")
    torch.testing.assert_close(area_vert_normals, expected_area_normals)

    norm = np.sqrt(0.5**2 + 0.5**2)
    expected_graph_normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [-0.5 / norm, 0.0, 0.5 / norm],
            [-0.5 / norm, 0.0, 0.5 / norm],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    graph_vert_normals = vertex_normals(verts, faces, "graph")
    torch.testing.assert_close(graph_vert_normals, expected_graph_normals)

    triangles = face_index(verts, faces)
    angles = interior_angles(triangles)
    norm_1 = np.sqrt(angles[0, 1] ** 2 + angles[1, 0] ** 2)
    norm_2 = np.sqrt(angles[0, 2] ** 2 + angles[0, 2] ** 2)
    expected_graph_normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [-angles[0, 1] / norm_1, 0.0, angles[1, 0] / norm_2],
            [-angles[0, 2] / norm_1, 0.0, angles[0, 2] / norm_2],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )

import numpy as np
import pytest
import scipy
import torch

import iskra.sparse as sp
from tests.template import assert_equal


def test_isect_indices_basic():
    a = torch.tensor([[0, 1, 2], [3, 4, 5]])
    b = torch.tensor([[1, 2, 3], [4, 5, 6]])
    a_mask, b_mask = sp.isect_indices(a, b)
    assert torch.equal(a_mask, torch.tensor([False, True, True]))
    assert torch.equal(b_mask, torch.tensor([True, True, False]))


def test_isect_indices_empty():
    a = torch.tensor([[0, 1], [2, 3]])
    b = torch.tensor([[4, 5], [6, 7]])
    a_mask, b_mask = sp.isect_indices(a, b)
    assert not a_mask.any()
    assert not b_mask.any()


def test_isect_indices_order_independent():
    a = torch.tensor([[2, 1, 0], [5, 4, 3]])
    b = torch.tensor([[1, 2], [4, 5]])
    a_mask, b_mask = sp.isect_indices(a, b)
    assert torch.equal(a_mask, torch.tensor([True, True, False]))
    assert torch.equal(b_mask, torch.tensor([True, True]))


def test_mul_sparse_sparse_basic():
    a_idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    a_val = torch.tensor([2.0, 3.0, 4.0])
    a = sp.coo_tensor(a_idx, a_val, size=(4, 6))

    b_idx = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b_val = torch.tensor([5.0, 6.0, 7.0])
    b = sp.coo_tensor(b_idx, b_val, size=(4, 6))

    out = sp.mul_sparse_sparse(a, b)

    exp_idx = torch.tensor([[1, 2], [4, 5]])
    exp_val = torch.tensor([3.0 * 5.0, 4.0 * 6.0])

    assert torch.equal(out.indices(), exp_idx)
    assert torch.equal(out.values(), exp_val)


def test_mul_sparse_sparse_shape_mismatch():
    a = sp.coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), size=(2, 2))
    b = sp.coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), size=(3, 3))
    with pytest.raises(ValueError):
        sp.mul_sparse_sparse(a, b)


def test_mul_sparse_sparse_partial_intersection():
    a_idx = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    a_val = torch.tensor([2.0, 3.0, 4.0, 5.0])
    a = sp.coo_tensor(a_idx, a_val, size=(4, 4))

    b_idx = torch.tensor([[1, 3], [1, 3]])
    b_val = torch.tensor([10.0, 20.0])
    b = sp.coo_tensor(b_idx, b_val, size=(4, 4))

    out = sp.mul_sparse_sparse(a, b).coalesce()

    exp_idx = torch.tensor([[1, 3], [1, 3]])
    exp_val = torch.tensor([3.0 * 10.0, 5.0 * 20.0])

    assert torch.equal(out.indices(), exp_idx)
    assert torch.equal(out.values(), exp_val)


def test_cat_sparse():
    a = sp.coo_tensor(
        torch.tensor([[0, 1], [0, 1]]), torch.tensor([1.0, 2.0]), size=(3, 2)
    )
    b = sp.coo_tensor(
        torch.tensor([[0, 2], [0, 1]]), torch.tensor([3.0, 4.0]), size=(3, 2)
    )

    out = sp.cat([a, b], dim=1).coalesce()
    exp = sp.coo_tensor(
        torch.tensor([[0, 1, 0, 2], [0, 1, 2, 3]]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        size=(3, 4),
    ).coalesce()

    assert torch.equal(out.indices(), exp.indices())
    assert torch.equal(out.values(), exp.values())
    assert out.shape == exp.shape

    out = sp.cat([a, b], dim=0).coalesce()
    exp = sp.coo_tensor(
        torch.tensor([[0, 1, 3, 5], [0, 1, 0, 1]]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        size=(6, 2),
    ).coalesce()

    assert torch.equal(out.indices(), exp.indices())
    assert torch.equal(out.values(), exp.values())
    assert out.shape == exp.shape


def test_indexing():
    a_idx = torch.tensor([[0, 1, 1, 2, 3], [0, 1, 0, 2, 3]])
    a_val = torch.tensor([2.0, 3.0, 6.0, 4.0, 5.0])
    a = sp.coo_tensor(a_idx, a_val, size=(4, 4))
    a_dense = a.to_dense()

    # We check for a few test that slicing matches what we'd expect manually,
    # to hopefully also catch if `SparseTensor.to_dense` goes awry.
    assert_equal(a[2, 2].to_dense(), torch.tensor(4.0))
    assert_equal(a[0, 1:2].to_dense(), torch.tensor([0.0]))

    assert_equal(a[2, 2].to_dense(), a_dense[2, 2])
    assert_equal(a[3].to_dense(), a_dense[3])
    assert_equal(a[0, 1:2].to_dense(), a_dense[0, 1:2])
    assert_equal(a[0, 1:3].to_dense(), a_dense[0, 1:3])
    assert_equal(a[0:2, 1:3].to_dense(), a_dense[0:2, 1:3])

    assert_equal(a[0, torch.tensor([0])].to_dense(), a_dense[0, torch.tensor([0])])
    assert_equal(a[0, torch.arange(1, 2)].to_dense(), a_dense[0, torch.arange(1, 2)])
    assert_equal(a[0, torch.arange(1, 3)].to_dense(), a_dense[0, torch.arange(1, 3)])
    assert_equal(
        a[0:2, torch.arange(1, 3)].to_dense(), a_dense[0:2, torch.arange(1, 3)]
    )

    mask = torch.tensor([True, False, True, False])
    assert_equal(a[1, mask].to_dense(), a_dense[1, mask])
    assert_equal(a[2:, mask].to_dense(), a_dense[2:, mask])

    assert_equal(a[0:2, None, 1:3].to_dense(), a_dense[0:2, None, 1:3])
    assert_equal(a[3, None, 2].to_dense(), a_dense[3, None, 2])
    assert_equal(a[None, 3, None].to_dense(), a_dense[None, 3, None])
    assert_equal(a[None, mask, :].to_dense(), a_dense[None, mask, :])

    # Following are different!
    print(a[(torch.arange(0, 2)), torch.arange(1, 3)].to_dense())
    print(a.to_dense()[torch.arange(0, 2), torch.arange(1, 3)])

    print(a[(0, 1), (1, 2)].to_dense())
    print(a.to_dense()[(0, 1), (1, 2)])

    print(a[mask, mask].to_dense())
    print(a_dense[mask, mask])

    print(a[(True, False, True, False), (True, False, True, False)].to_dense())
    print(a_dense[(True, False, True, False), (True, False, True, False)])


def test_backward_coo():
    a_idx = torch.tensor([[0, 1, 1, 2, 3], [0, 0, 1, 2, 3]])
    a_val_list = [2.0, 6.0, 3.0, 4.0, 5.0]
    expected_grad = torch.tensor([4.0, 12.0, 6.0, 8.0, 10.0])
    expected_loss = torch.tensor(90.0)

    # Test base use-case:
    a_val = torch.tensor(a_val_list, requires_grad=True)
    a = sp.coo_tensor(a_idx, a_val, size=(4, 4)).coalesce()
    loss = a.square().sum()
    loss.backward()

    assert isinstance(a, sp.SparseTensor)
    assert not a.is_leaf
    assert a_val.grad is not None
    assert_equal(loss, expected_loss)
    assert_equal(a_val.grad, expected_grad)

    # Test with `.values()` call:
    a_val = torch.tensor(a_val_list, requires_grad=True)
    a = sp.coo_tensor(a_idx, a_val, size=(4, 4)).coalesce()
    loss = a.values().square().sum()
    loss.backward()

    assert isinstance(a, sp.SparseTensor)
    assert not a.is_leaf
    assert a_val.grad is not None
    assert_equal(loss, expected_loss)
    assert_equal(a_val.grad, expected_grad)

    # Test with `sp.alias()` call:
    a_val = torch.tensor(a_val_list, requires_grad=True)
    a = sp.coo_tensor(a_idx, a_val, size=(4, 4)).coalesce()
    a = sp.alias(a)
    loss = a.square().sum()
    loss.backward()

    assert isinstance(a, sp.SparseTensor)
    assert not a.is_leaf
    assert a_val.grad is not None
    assert_equal(loss, expected_loss)
    assert_equal(a_val.grad, expected_grad)

    # Test with `sp.SparseTensor` as leaf tensor:
    a_val = torch.tensor(a_val_list)
    a = sp.coo_tensor(a_idx, a_val, size=(4, 4), requires_grad=True)
    loss = a.square().sum()
    loss.backward()

    assert isinstance(a, sp.SparseTensor)
    assert a.is_leaf
    assert a_val.grad is None
    assert_equal(loss, expected_loss)
    assert_equal(a.grad.indices(), a.indices())
    assert_equal(a.grad.values(), expected_grad)


def test_backward_csr():
    crow = torch.tensor([0, 1, 3, 4, 5])
    col = torch.tensor([0, 0, 1, 2, 3])
    val_list = [2.0, 6.0, 3.0, 4.0, 5.0]
    expected_grad = torch.tensor([4.0, 12.0, 6.0, 8.0, 10.0])
    expected_loss = torch.tensor(90.0)

    # Test base use-case:
    a_val = torch.tensor(val_list, requires_grad=True)
    a = sp.csr_tensor(crow, col, a_val, size=(4, 4))
    loss = a.square().sum()
    loss.backward()

    assert isinstance(a, sp.SparseTensor)
    assert not a.is_leaf
    assert a_val.grad is not None
    assert_equal(loss, expected_loss)
    assert_equal(a_val.grad, expected_grad)

    # Test with `.values()` call:
    a_val = torch.tensor(val_list, requires_grad=True)
    a = sp.csr_tensor(crow, col, a_val, size=(4, 4))
    loss = a.values().square().sum()
    loss.backward()

    assert isinstance(a, sp.SparseTensor)
    assert not a.is_leaf
    assert a_val.grad is not None
    assert_equal(loss, expected_loss)
    assert_equal(a_val.grad, expected_grad)

    # Test with `sp.alias()` call:
    a_val = torch.tensor(val_list, requires_grad=True)
    a = sp.csr_tensor(crow, col, a_val, size=(4, 4))
    a = sp.alias(a)
    loss = a.square().sum()
    loss.backward()

    assert isinstance(a, sp.SparseTensor)
    assert not a.is_leaf
    assert a_val.grad is not None
    assert_equal(loss, expected_loss)
    assert_equal(a_val.grad, expected_grad)

    # Test with `sp.SparseTensor` as leaf tensor:
    a_val = torch.tensor(val_list)
    a = sp.csr_tensor(crow, col, a_val, size=(4, 4), requires_grad=True)
    loss = a.square().sum()
    loss.backward()

    assert isinstance(a, sp.SparseTensor)
    assert a.is_leaf
    assert a_val.grad is None
    assert_equal(loss, expected_loss)
    assert_equal(a.grad.crow_indices(), a.crow_indices())
    assert_equal(a.grad.col_indices(), a.col_indices())
    assert_equal(a.grad.values(), expected_grad)


def test_scipy_conversion():
    a_idx = torch.tensor([[0, 1, 1, 2, 3], [0, 0, 1, 2, 3]])
    a_val = torch.tensor([2.0, 6.0, 3.0, 4.0, 5.0])
    a_coo = sp.coo_tensor(a_idx, a_val, size=(4, 4))
    a_coo_sp: scipy.sparse.sparray = a_coo.scipy()
    assert a_coo_sp.format == "coo"
    assert_equal(a_coo.values().numpy(), a_coo_sp.data, atol=0, rtol=0)
    assert_equal(a_coo.indices().numpy(), np.concat([a_coo_sp.coords]), atol=0, rtol=0)

    a_coo_ret = sp.from_scipy(a_coo_sp)
    assert a_coo_ret.layout == torch.sparse_coo
    assert_equal(a_coo_ret.values().numpy(), a_coo_sp.data, atol=0, rtol=0)
    assert_equal(
        a_coo_ret.indices().numpy(), np.concat([a_coo_sp.coords]), atol=0, rtol=0
    )

    a_csr: sp.SparseTensor = a_coo.to_sparse_csr()
    a_csr_sp: scipy.sparse.sparray = a_csr.scipy()
    assert a_csr_sp.format == "csr"
    assert_equal(a_csr.values().numpy(), a_csr_sp.data, atol=0, rtol=0)
    assert_equal(a_csr.col_indices().numpy(), a_csr_sp.indices, atol=0, rtol=0)
    assert_equal(a_csr.crow_indices().numpy(), a_csr_sp.indptr, atol=0, rtol=0)

    a_csr_ret = sp.from_scipy(a_csr_sp)
    assert a_csr_ret.layout == torch.sparse_csr
    assert_equal(a_csr_ret.values().numpy(), a_csr_sp.data, atol=0, rtol=0)
    assert_equal(a_csr_ret.col_indices().numpy(), a_csr_sp.indices, atol=0, rtol=0)
    assert_equal(a_csr_ret.crow_indices().numpy(), a_csr_sp.indptr, atol=0, rtol=0)

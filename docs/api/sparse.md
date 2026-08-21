# Sparse Linear Algebra

`iskra ✨` provides a fully fledged sparse tensor framework to help you write geometry processing code.
Its functionality primarily covers sparse COO tensors for most operations and should be used, e.g., while you're assembling sparse linear operators.
Support for CSR tensors is very limited, but they are preferred in cases of matrix multiplication and sparse linear solves.
For example, PyTorch has no (proper) support for sparse COO matrix multiplication, so iskra will conver to CSR under the hood every time you attempt to multiply with a COO matrix.
Therefore, a common pattern would be to do all operations in the COO format, and convert to CSR right before you are about to do matrix multiplies or linear solves.

It was important that the sparse API was written directly in Python instead of as custom kernels.
This allows the code to remain hackable, we benefit from low-level kernels and bindings in PyTorch itself, and code distribution becomes significantly simpler.
Ideally, PyTorch itself would catch up on the sparse tensor functionality and I would not have to maintain a sparse tensor library :)

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: module.rst
   iskra.sparse
   iskra.sparse_linalg

.. include:: generated/iskra.sparse.rst
.. include:: generated/iskra.sparse_linalg.rst
```

```{toctree}
:maxdepth: 1
:hidden:

generated/iskra.sparse
generated/iskra.sparse_linalg
```
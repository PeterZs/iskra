# Low-Level Geometry

The low-level geometry functionality of `iskra` ✨. 
Functions in this module deal directly with low-level geometric constructs necessary to build up higher level algorithms.
Except in a few select cases (like `iskra.vertex_normals`), the functions in this module are unaware of any mesh or mesh topology and operate directly on points, simplices, coordinate systems, or things like quaternions.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: module.rst
   iskra.geometry.barycentric
   iskra.geometry.distances
   iskra.geometry.volume
   iskra.geometry.angles
   iskra.geometry.bbox
   iskra.geometry.coordinate_system
   iskra.geometry.element_quality
   iskra.geometry.normals

.. include:: generated/iskra.geometry.barycentric.rst
.. include:: generated/iskra.geometry.distances.rst
.. include:: generated/iskra.geometry.volume.rst
.. include:: generated/iskra.geometry.angles.rst
.. include:: generated/iskra.geometry.bbox.rst
.. include:: generated/iskra.geometry.coordinate_system.rst
.. include:: generated/iskra.geometry.element_quality.rst
.. include:: generated/iskra.geometry.normals.rst
```
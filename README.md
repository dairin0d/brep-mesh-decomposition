# brep-mesh-decomposition

An experimental attempt at implementing a basic end-to-end conversion from triangulated meshes to CAD [boundary representations](https://en.wikipedia.org/wiki/Boundary_representation) (the process usually called in literature as whatever strikes the authors' fancy: mesh decomposition, model/topology reconstruction, mesh/shape/surface segmentation, structure/primitive recovery, primitive recognition, reverse engineering... lol :D)

This came into existence as a result of an effort to create a [STEP](https://ap238.org/SMRL_v8_final/data/resource_docs/geometric_and_topological_representation/sys/contents.htm) exporter for Blender. Since I couldn't find any open-source implementations of B-Rep extraction methods (aside from [ML-based approaches](https://github.com/QiujieDong/Mesh_Segmentation), [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) and [EfPiSoft](https://efpisoft.sourceforge.net/), which deal with only a part of the whole puzzle), I ended up cobbling together my own solution, based on ideas from the few papers that my current math background was sufficient to understand (plus a lot of experimentation).

The result is far from being robust or reliable, but, at least on the cases I tested, in principle it can produce usable outputs at correctly selected values of fitting tolerance. Note, however, that this repository contains just the mesh decomposition part (without the mesh preprocessing and STEP exporting code), because it doesn't depend on Blender-specific functionality. Perhaps I'll add the STEP serialization part too, someday, but that is as of yet uncertain.

The current state of the code is highly experimental, quite messy and lacks documentation, so browse at your own risk ;-) However, even something like this is better than nothing, and perhaps may even serve as a somewhat useful reference for people who would seek to add similar functionality into other software.

I'm currently busy with other projects, and not sure if/when I'll get around to tinker with this further. That said, if anyone is feeling generous, domain-knowledge tips and code contributions are welcome :-)

## References

These are the articles which contained some ideas I ended up implementing:

* [Direct least-squares fitting of algebraic surfaces (1987)](https://doi.org/10.1145/37401.37420)
* [Curvatures estimation on triangular mesh (2005)](https://doi.org/10.1631/jzus.2005.AS0128)
* [Hierarchical mesh segmentation based on fitting primitives (2006)](https://doi.org/10.1007/s00371-006-0375-x)
* [Simple primitive recognition via hierarchical face clustering (2020)](https://doi.org/10.1007/s41095-020-0192-6)

While examining the available research on the topic, I came across plenty of other publications, some of which are listed below. They either had some undesirable aspects for my purpose or required advanced mathematical background to understand/implement, but nevertheless the ideas seemed interesting enough to keep the references around.

Medium-level math:

* [Hierarchical Structure Recovery of Point-Sampled Surfaces (2010)](https://doi.org/10.1111/j.1467-8659.2010.01658.x)
* [Recovering Primitives in 3D CAD meshes (2011)](https://doi.org/10.1117/12.872665)
* [A comprehensive process of reverse engineering from 3D meshes to CAD models (2013)](https://doi.org/10.1016/J.CAD.2013.06.004)
* [Extracting shape features from a surface mesh using geometric reasoning (2020)](https://doi.org/10.1016/j.procir.2020.02.142)

Advanced math:

* [Structure Recovery via Hybrid Variational Surface Approximation (2005)](https://doi.org/10.1111/j.1467-8659.2005.00852.x)
* [A framework for 3D model reconstruction in reverse engineering (2012)](https://doi.org/10.1016/j.cie.2012.07.009)
* [Variational mesh segmentation via quadric surface fitting (2012)](https://doi.org/10.1016/j.cad.2012.04.005)
* [CAD mesh models segmentation into swept surfaces (2017)](https://doi.org/10.1007/S00170-017-0437-4)
* [Shape Segmentation Using Local Slippage Analysis (2019)](https://doi.org/10.1145/1057432.1057461)
* [Blending Surface Segmentation and Editing for 3D Models (2020)](https://doi.org/10.1109/TVCG.2020.3045450)
* [Recognizing geometric primitives in 3D point clouds of mechanical CAD objects (2023)](https://doi.org/10.1016/j.cad.2023.103479)

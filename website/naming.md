# Name & branding

## Why “Grizzly”

**Grizzly** is a subtle homage to **Grizzly Peak**, the ridge behind Lawrence Berkeley National Laboratory. It nods to the project’s roots near the **MAESTRO** beamline while staying **institutionally independent** as a standalone open package. The name also suggests heavy-duty computational strength — the intended feel of a GPU-scale matrix-element engine.

## Why “ME”

**ME** stands for **matrix element** — the core quantity GrizzlyME accelerates for ARPES simulation.

## The acronym

**G.R.I.Z.Z.L.Y.** = **G**PU **R**endered **I**nner-products for **Z**one-by-**Z**one **L**attice **Y**ields.

That backronym matches the technical idea: batched (zone-by-zone / k-mesh) inner products that yield photoemission intensities on the GPU.

## Relation to chinook

[chinook](https://github.com/rpday/chinook) is the established CPU-oriented framework for ARPES matrix elements from tight-binding models. GrizzlyME is a **high-performance companion**: same scientific workflow for spinless models today, with the hot path (matrix elements, diagonalization, spectral assembly) rewritten in PyTorch for CPU and CUDA. Chinook remains the dependency for TB setup and radial-integral definitions.

> chinook’s public docs do not narrate the origin of the name “chinook”; GrizzlyME documents its own naming here for transparency and community context.

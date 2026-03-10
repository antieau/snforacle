# Finite-Field Matrix API

Functions for matrices whose entries are elements of F_p (integers mod a prime p).

Over a field every nonzero element is a unit, so:

- The **SNF** is always `diag(1, ..., 1, 0, ..., 0)` where the number of 1s equals the rank.
- The **HNF** is the unique reduced row echelon form (RREF).

## Functions

::: snforacle.ff_interface.ff_smith_normal_form

::: snforacle.ff_interface.ff_smith_normal_form_with_transforms

::: snforacle.ff_interface.ff_hermite_normal_form

::: snforacle.ff_interface.ff_hermite_normal_form_with_transform

::: snforacle.ff_interface.ff_rank

## Input models

::: snforacle.ff_schema.DenseFFMatrix

::: snforacle.ff_schema.SparseFFMatrix

## Output models

::: snforacle.ff_schema.FFSNFResult

::: snforacle.ff_schema.FFSNFWithTransformsResult

::: snforacle.ff_schema.FFHNFResult

::: snforacle.ff_schema.FFHNFWithTransformResult

::: snforacle.ff_schema.FFRankResult

# Integer Matrix API

Functions for computing SNF, HNF, and elementary divisors of integer matrices.

All functions accept a `DenseIntMatrix`, `SparseIntMatrix`, or a plain `dict` conforming to either schema. See [Input Formats](../input-formats.md) for details.

## Functions

::: snforacle.interface.smith_normal_form

::: snforacle.interface.smith_normal_form_with_transforms

::: snforacle.interface.hermite_normal_form

::: snforacle.interface.hermite_normal_form_with_transform

::: snforacle.interface.elementary_divisors

## Input models

::: snforacle.schema.DenseIntMatrix

::: snforacle.schema.SparseIntMatrix

::: snforacle.schema.SparseEntry

## Output models

::: snforacle.schema.SNFResult

::: snforacle.schema.SNFWithTransformsResult

::: snforacle.schema.HNFResult

::: snforacle.schema.HNFWithTransformResult

::: snforacle.schema.ElementaryDivisorsResult

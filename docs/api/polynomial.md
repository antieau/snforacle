# Polynomial Matrix API

Functions for matrices whose entries are polynomials over a finite field F_p[x].

Polynomials are represented as coefficient lists `[c_0, c_1, ..., c_d]` (constant term first), with coefficients in `[0, p-1]`. The zero polynomial is `[]`. No trailing zeros are permitted.

All invariant factors and HNF pivots are returned as **monic polynomials**.

## Functions

::: snforacle.poly_interface.poly_smith_normal_form

::: snforacle.poly_interface.poly_smith_normal_form_with_transforms

::: snforacle.poly_interface.poly_hermite_normal_form

::: snforacle.poly_interface.poly_hermite_normal_form_with_transform

::: snforacle.poly_interface.poly_elementary_divisors

## Input models

::: snforacle.poly_schema.DensePolyMatrix

::: snforacle.poly_schema.SparsePolyMatrix

## Output models

::: snforacle.poly_schema.PolySNFResult

::: snforacle.poly_schema.PolySNFWithTransformsResult

::: snforacle.poly_schema.PolyHNFResult

::: snforacle.poly_schema.PolyHNFWithTransformResult

::: snforacle.poly_schema.PolyElementaryDivisorsResult

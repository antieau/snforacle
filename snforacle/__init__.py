from snforacle.interface import (
    elementary_divisors,
    hermite_normal_form,
    hermite_normal_form_with_transform,
    smith_normal_form,
    smith_normal_form_with_transforms,
)
from snforacle.schema import (
    DenseIntMatrix,
    ElementaryDivisorsResult,
    HNFResult,
    HNFWithTransformResult,
    SNFResult,
    SNFWithTransformsResult,
    SparseIntMatrix,
)
from snforacle.poly_interface import (
    poly_smith_normal_form,
    poly_smith_normal_form_with_transforms,
    poly_hermite_normal_form,
    poly_hermite_normal_form_with_transform,
    poly_elementary_divisors,
)
from snforacle.poly_schema import (
    DensePolyMatrix,
    SparsePolyMatrix,
    PolySNFResult,
    PolySNFWithTransformsResult,
    PolyHNFResult,
    PolyHNFWithTransformResult,
    PolyElementaryDivisorsResult,
)
from snforacle.ff_interface import (
    ff_smith_normal_form,
    ff_smith_normal_form_with_transforms,
    ff_hermite_normal_form,
    ff_hermite_normal_form_with_transform,
    ff_rank,
)
from snforacle.ff_schema import (
    DenseFFMatrix,
    SparseFFMatrix,
    FFSNFResult,
    FFSNFWithTransformsResult,
    FFHNFResult,
    FFHNFWithTransformResult,
    FFRankResult,
)

__all__ = [
    # Integer matrix operations
    "smith_normal_form",
    "smith_normal_form_with_transforms",
    "hermite_normal_form",
    "hermite_normal_form_with_transform",
    "elementary_divisors",
    "DenseIntMatrix",
    "SparseIntMatrix",
    "SNFResult",
    "SNFWithTransformsResult",
    "HNFResult",
    "HNFWithTransformResult",
    "ElementaryDivisorsResult",
    # Polynomial matrix operations over F_p[x]
    "poly_smith_normal_form",
    "poly_smith_normal_form_with_transforms",
    "poly_hermite_normal_form",
    "poly_hermite_normal_form_with_transform",
    "poly_elementary_divisors",
    "DensePolyMatrix",
    "SparsePolyMatrix",
    "PolySNFResult",
    "PolySNFWithTransformsResult",
    "PolyHNFResult",
    "PolyHNFWithTransformResult",
    "PolyElementaryDivisorsResult",
    # Finite field matrix operations over F_p
    "ff_smith_normal_form",
    "ff_smith_normal_form_with_transforms",
    "ff_hermite_normal_form",
    "ff_hermite_normal_form_with_transform",
    "ff_rank",
    "DenseFFMatrix",
    "SparseFFMatrix",
    "FFSNFResult",
    "FFSNFWithTransformsResult",
    "FFHNFResult",
    "FFHNFWithTransformResult",
    "FFRankResult",
]

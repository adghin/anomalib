"""Implementation of Precision metric based on TorchMetrics."""
"""@author: adghin"""

import logging
from typing import Any, Literal

from torchmetrics.classification import BinaryPrecision

logger = logging.getLogger(__name__)

class PRECISION(BinaryPrecision):
    """
    This class returns the precision metric at fixed threshold.
    """
    def __init__(
            self,
            threshold: float = 0.5,
            multidim_average: Literal["global"] | Literal["samplewise"] = "global",
            ignore_index: int | None = None,
            validate_args: bool = True,
            **kwargs: Any,  # noqa: ANN401
        ) -> None:
            super().__init__(threshold, multidim_average, ignore_index, validate_args, **kwargs)
    

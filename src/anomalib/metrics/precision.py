"""Implementation of Precision metric based on TorchMetrics."""
"""@author: adghin"""

import logging

import torch
from torchmetrics import Metric

from anomalib.metrics.precision_recall_curve import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification import binary_precision

logger = logging.getLogger(__name__)

class PRECISION():
    """Compute precision metric"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.precision_recall_curve = BinaryPrecisionRecallCurve()

        self.threshold: torch.Tensor

    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> None:
        """Update the precision-recall curve metric."""
        del args, kwargs  # These variables are not used.

        self.precision_recall_curve.update(preds, target)

    def compute(self) -> torch.Tensor:

        preds, targets, _ = self.precision_recall_curve._binary_precision_recall_curve_format()

        print(preds)
        print(targets)

    def reset(self) -> None:
        """Reset the metric."""
        self.precision_recall_curve.reset()
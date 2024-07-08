"""Implementation of Precision metric based on TorchMetrics."""
"""@author: adghin"""

import logging

import torch
from torchmetrics import Metric

from anomalib.metrics.precision_recall_curve import BinaryPrecisionRecallCurve

logger = logging.getLogger(__name__)

class PRECISION(BinaryPrecisionRecallCurve):
    """
    This class returns the recall metric, which is computed in the BinaryPrecisionRecallCurve class from
    anomalib.metrics.precision_recall_curve. This is needed just for the sake of consistencty with the anomalib metrics collection.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.precision_recall_curve = BinaryPrecisionRecallCurve()

    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> None:
        """Update the precision-recall curve metric."""
        del args, kwargs  # These variables are not used.

        self.precision_recall_curve.update(preds, target)

    def compute(self) -> torch.Tensor:
        """
        Returns: tensor with the precision value to be logged in the results
        """
        precision: torch.Tensor
        precision = self.precision_recall_curve.compute_precision()
        print(precision)
        return precision
    
    def reset(self) -> None:
        """Reset the metric."""
        self.precision_recall_curve.reset()
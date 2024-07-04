"""Implementation of Precision metric based on TorchMetrics."""
"""@author: adghin"""

import logging

import torch
from torchmetrics import Metric

from anomalib.metrics.precision_recall_curve import BinaryPrecisionRecallCurve

logger = logging.getLogger(__name__)

class RECALL(BinaryPrecisionRecallCurve):
    """Compute recall metric"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.precision_recall_curve = BinaryPrecisionRecallCurve()

        self.threshold: torch.Tensor

    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> None:
        """Update the precision-recall curve metric."""
        del args, kwargs  # These variables are not used.

        self.precision_recall_curve.update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute the value of precision score.

        Returns:
            Value of the precision score.
        """
        recall: torch.Tensor

        precision, recall, thresholds = self.precision_recall_curve.compute()
        
        return recall

    def reset(self) -> None:
        """Reset the metric."""
        self.precision_recall_curve.reset()
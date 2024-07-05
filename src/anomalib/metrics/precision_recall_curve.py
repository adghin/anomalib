"""Custom PrecisionRecallCurve.

The one in torchmetrics adds a sigmoid operation on top of the thresholds.
See: https://github.com/Lightning-AI/torchmetrics/issues/1526
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torchmetrics.classification import BinaryPrecisionRecallCurve as _BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_update,
)

from torchmetrics.functional.classification import binary_precision
from torchmetrics.functional.classification import binary_recall

class BinaryPrecisionRecallCurve(_BinaryPrecisionRecallCurve):
    """Binary precision-recall curve with without threshold prediction normalization."""

    @staticmethod
    def _binary_precision_recall_curve_format(
        preds: Tensor,
        target: Tensor,
        thresholds: int | list[float] | Tensor | None = None,
        ignore_index: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Similar to torchmetrics' ``_binary_precision_recall_curve_format`` except it does not apply sigmoid."""
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]

        thresholds = _adjust_threshold_arg(thresholds, preds.device)
        return preds, target, thresholds

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state with new predictions and targets.

        Unlike the base class, this accepts raw predictions and targets.

        Args:
            preds (Tensor): Predicted probabilities
            target (Tensor): Ground truth labels
        """
        preds, target, _ = BinaryPrecisionRecallCurve._binary_precision_recall_curve_format(
            preds,
            target,
            self.thresholds,
            self.ignore_index,
        )
        state = _binary_precision_recall_curve_update(preds, target, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute_precision(self) -> torch.Tensor:
        """
        Compute the precision based on preds and targets

        Returns: tensor with the value of the precision
        """

        prec: torch.Tensor

        preds = torch.cat(self.preds) if isinstance(self.preds, list) else self.preds
        target = torch.cat(self.target) if isinstance(self.target, list) else self.target

        prec = binary_precision(preds,target)
        return prec

    def compute_recall(self) -> torch.Tensor:
        """
        Compute the recall based on preds and targets

        Returns: tensor with the value of the recall
        """
        rec: torch.Tensor

        preds = torch.cat(self.preds) if isinstance(self.preds, list) else self.preds
        target = torch.cat(self.target) if isinstance(self.target, list) else self.target

        rec = binary_recall(preds,target)
        return rec



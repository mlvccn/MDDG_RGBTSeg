import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, List
from torch.nn.modules.loss import _Loss
import numpy as np

BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    
def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

def label_smoothed_nll_loss(
    lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
) -> torch.Tensor:
    """
    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    #  epsilon=self.smooth_factor
    #- **y_pred** - torch.Tensor of shape (N, C, H, W) **y_true** - torch.Tensor of shape (N, H, W)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)
    # print(target.shape)
    # print(dim)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss
# modify to adpate pixel segmentation 
class LabelSmoothingEncouragingLoss(nn.Module):
    """
    Encouraging Loss with label smoothing for semantic segmentation.
    """
    def __init__(self, smoothing=0.1, log_end=0.75, ignore_index=255):
        """
        :param smoothing: label smoothing factor
        :param log_end: threshold for modifying the bonus function
        :param ignore_index: label index to be ignored in loss calculation
        """
        super(LabelSmoothingEncouragingLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.log_end = log_end
        self.ignore_index = ignore_index

    def forward(self, x, target):
        """
        :param x: torch.Tensor of shape (N, C, H, W), predicted logits
        :param target: torch.Tensor of shape (N, H, W), ground truth labels
        """
        N, C, H, W = x.shape
        logprobs = F.log_softmax(x, dim=1)
        probs = torch.exp(logprobs)
        bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=1e-5))  # likelihood bonus
        
        if self.log_end != 1.0:
            log_end = self.log_end
            y_log_end = torch.log(torch.ones_like(probs) - log_end)
            bonus_after_log_end = 1.0 / (log_end - torch.ones_like(probs)) * (probs - log_end) + y_log_end
            bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)
        # Ignore index handling
        mask = target.squeeze(1) != self.ignore_index
        # Reshape target for indexing: (N, H, W) -> (N, 1, H, W)
        target = target.unsqueeze(1)
        # print((bonus - logprobs).shape)
        # Gather log probabilities and bonus for target classes
        target[target == self.ignore_index] = 0
        el_loss = torch.gather((bonus - logprobs), dim=1, index=target).squeeze(1)
        smooth_loss = (bonus - logprobs).mean(dim=1)
        loss = self.confidence * el_loss + self.smoothing * smooth_loss
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    
class SoftCrossEntropyLoss(nn.Module):

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        ignore_index: Optional[int] = -100,
        dim: int = 1,
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing
        
        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])
        
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)
    
class DiceLoss(_Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss=False,
        from_logits=True,
        smooth: float = 0.0,
        ignore_index=None,
        eps=1e-7,
    ):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        # if classes is not None:
        #     assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
        #     classes = torch.Tensor(classes).to(de)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss * self.classes

        return loss.mean()
    
# class Dice(nn.Module):
#     def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
#         """
#         delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
#         """
#         super().__init__()
#         self.delta = delta
#         self.aux_weights = aux_weights

#     def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
#         # preds in shape [B, C, H, W] and labels in shape [B, H, W]
#         num_classes = preds.shape[1]
#         labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
#         tp = torch.sum(labels*preds, dim=(2, 3))
#         fn = torch.sum(labels*(1-preds), dim=(2, 3))
#         fp = torch.sum((1-labels)*preds, dim=(2, 3))

#         dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
#         dice_score = torch.sum(1 - dice_score, dim=-1)

#         dice_score = dice_score / num_classes
#         return dice_score.mean()

#     def forward(self, preds, targets: Tensor) -> Tensor:
#         if isinstance(preds, tuple):
#             return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
#         return self._forward(preds, targets)


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice','SoftCrossEntropyLoss','LabelSmoothingEncouragingLoss']


# if cfg['class_weight'] == 'enet':
#         self.class_weight = np.array(
#                 [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
#             self.binary_class_weight = np.array([1.5121, 10.2388])
#         elif cfg['class_weight'] == 'median_freq_balancing':
#             self.class_weight = np.array(
#                 [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
#             self.binary_class_weight = np.array([0.5454, 6.0061])

def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return DiceLoss(mode='multiclass',ignore_index = ignore_label, classes = cls_weights)
    if loss_fn_name =='SoftCrossEntropyLoss':
        return SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index = ignore_label)
    if loss_fn_name =='CrossEntropy':
        return nn.CrossEntropyLoss(ignore_index = ignore_label, weight = cls_weights, label_smoothing = 0.1)
    if loss_fn_name =='LabelSmoothingEncouragingLoss':
        return LabelSmoothingEncouragingLoss()    
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = DiceLoss()
    y = loss_fn(pred, label)
    print(y)
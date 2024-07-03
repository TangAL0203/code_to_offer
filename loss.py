import torch



# Smooth L1 Loss 相比L1 loss 改进了零点不平滑问题。且loss值较小的区域，梯度更大一些，这样网络学习更好。
# 相比于L2 loss，在 x 较大的时候不像 L2 对异常值敏感，是一个缓慢变化的loss。

def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta,
                       0.5 * diff * diff / beta,  # |x| <= 1, 0.5 * x^2
                       diff - 0.5 * beta)  # |x| > 1, |x| - 0.5
    return loss


## focal loss: FL(p_t) = -alpha_t * (1 - p_t) ^ gamma * log(p_t), alpha_t只在正样本的时候用

@LOSS.register
class FocalLoss2d(nn.Module):
    """Sigmoid focal loss.

    Args:
        num_classes (int): Num_classes including background, C+1, C is number
            of foreground categories.
        alpha (float): A weighting factor for pos-sample, (1-alpha) is for
            neg-sample.
        gamma (float): Gamma used in focal loss to compress the contribution
            of easy examples.
        loss_weight (float): Global weight of loss. Defaults is 1.0.
        eps (float): A small value to avoid zero denominator.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].

    Returns:
        dict: A dict containing the calculated loss, the key of loss is
        loss_name.
    """
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, loss_weight=1.0,
                 eps=1e-12, reduction='mean'):
        super(FocalLoss2d, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, loss_input) -> Tensor:
        """
        Args:
            loss_input: A dict contains the following keys:
                loss (Tensor): Loss value which calculated elsewhere, just
                    return it.
                pred (Tensor): Cls pred, with shape(N, C), C is num_classes of
                    foreground.
                target (Tensor): Cls target, with shape(N,), values in [0, C-1]
                    represent the foreground, C or negative value represent the
                    background.
                weight (Tensor): The weight of loss for each prediction.
                    Default is None.
                avg_factor (float): Normalized factor.
                points_per_strides (list[int]): Points num list of all strides.
                valid_classes_list (list[list]): The len of the outer list is
                    N(batch_size), the inner list is the list of valid cls ids.
        """
        loss = loss_input.get('loss', None)
        if loss is not None:
            assert isinstance(loss, Tensor)
            return loss
        pred = loss_input.get('pred')
        target = loss_input.get('target')
        weight = loss_input.get('weight', None)
        avg_factor = loss_input.get('avg_factor', None)
        points_per_strides = loss_input.get('points_per_strides', None)
        valid_classes_list = loss_input.get('valid_classes_list', None)
        with autocast(enabled=False):
            # cast to fp32
            pred = pred.float().sigmoid()
            target[target < 0] = self.num_classes - 1
            one_hot = F.one_hot(target, self.num_classes)  # N x C+1
            one_hot = one_hot[:, :self.num_classes - 1]  # N x C
            pt = torch.where(torch.eq(one_hot, 1), pred, 1 - pred)
            t = torch.ones_like(one_hot)
            at = torch.where(
                torch.eq(one_hot, 1), self.alpha * t, (1 - self.alpha) * t)
            loss = -at * torch.pow((1 - pt), self.gamma) * torch.log(torch.minimum(pt + self.eps, t))  # noqa

            # for two datasets use same head, apply mask on loss
            if valid_classes_list is not None and loss.shape[-1] > 1:
                valid_loss_mask = torch.zeros_like(loss)
                start_indexs = [0, ] + list(
                    itertools.accumulate([xx * len(valid_classes_list) for xx in points_per_strides]))[:-1]  # noqa
                for str_id, points_per_stride in enumerate(points_per_strides):
                    for ii, valid_classes in enumerate(valid_classes_list):
                        start_idx = start_indexs[str_id] + ii * points_per_stride  # noqa
                        end_idx = start_indexs[str_id] + (ii + 1) * points_per_stride  # noqa
                        valid_loss_mask[start_idx: end_idx][:, valid_classes] = 1.0  # noqa
                loss = loss * valid_loss_mask

            if weight is not None:
                if weight.shape != loss.shape:
                    if weight.size(0) == loss.size(0):
                        # For most cases, weight is of shape
                        # (num_priors, ), which means it does not have
                        # the second axis num_class
                        weight = weight.view(-1, 1)
                    else:
                        assert weight.numel() == loss.numel()
                        weight = weight.view(loss.size(0), -1)
                assert weight.ndim == loss.ndim

            loss = weight_reduce_loss(loss, weight, self.reduction, avg_factor)
        return loss


@LOSSES.register_module()
@LOSS.register
class GIoULoss2D(nn.Module):
    """Generalized Intersection over Union Loss.

    Args:
        loss_weight (float): Global weight of loss. Defaults is 1.0.
        eps (float): A small value to avoid zero denominator.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].

    Returns:
        dict: A dict containing the calculated loss, the key of loss is
        loss_name.
    """
    def __init__(self, loss_weight=1.0, eps=1e-6, reduction='mean'):
        super(GIoULoss2D, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction

    @staticmethod
    def _cal_giou_loss(pred, target, eps=1e-6):
        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        # union
        ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + eps
        # IoU
        ious = overlap / union
        # enclose area
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps
        # GIoU
        gious = ious - (enclose_area - union) / enclose_area
        loss = 1 - gious
        return loss

    def forward(self, loss_input: Dict) -> Tensor:
        """
        Args:
            loss_input: A dict contains the following keys:
                pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2,
                    y2), represent upper-left and lower-right point, with shape
                    (N, 4).
                target (torch.Tensor): Corresponding gt_boxes, the same shape
                    as pred.
                weight (torch.Tensor): Element-wise weight loss weight, with
                    shape(N,).
                avg_factor (float): Average factor that is used to average the
                    loss.
        """
        pred = loss_input.get('pred')
        target = loss_input.get('target')
        weight = loss_input.get('weight', None)
        avg_factor = loss_input.get('avg_factor', None)
        with autocast(enabled=False):
            # cast to fp32
            loss = None
            if pred is not None:
                pred = pred.float()
                if weight is not None and not torch.any(weight > 0):
                    return (pred * weight).sum()

                if weight is not None and weight.dim() > 1:
                    # reduce the weight of shape (n, 4) to (n,) to match the
                    # giou_loss of shape (n,)
                    assert weight.shape == pred.shape
                    weight = weight.mean(-1)

                loss = self._cal_giou_loss(pred, target, self.eps)
                loss = weight_reduce_loss(loss, weight, self.reduction,
                                          avg_factor)

        return loss
import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t
    
    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        
        input, target = self.saved_variables
        grad_input = grad_target = None
        
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None
        
        return grad_input, grad_target


def f1_score(y_true, y_pred, threshold, eps=1e-9):
    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()
    
    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))
    
    return torch.mean((precision * recall).div(precision + recall + eps).mul(2))


def dice_loss(input, target):
    smooth = 1.
    
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    
    return s / (i + 1)

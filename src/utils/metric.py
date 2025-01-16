import torch


########### DeepSDF ###########
def clamp(delta, x):
    return torch.clamp(x, min=-delta, max=delta)


def mse(outputs, targets):
    outputs = outputs.reshape(-1, 1).detach().cpu().numpy()
    targets = targets.reshape(-1, 1).detach().cpu().numpy()
    return ((outputs - targets) ** 2).sum()  # taking sum just to track the progress


def deepsdf_loss(l1_criterion, pred, gt, delta=1):
    return l1_criterion(clamp(delta, pred), clamp(delta, gt))


class DeepSDFLoss(torch.nn.Module):
    def __init__(self, delta=1.0):
        super(DeepSDFLoss, self).__init__()
        self.delta = delta
        self.criterion = torch.nn.L1Loss(reduction="sum")

    def forward(self, pred, gt):
        pred = pred.reshape(-1, 1)
        gt = gt.reshape(-1, 1)
        return self.criterion(clamp(self.delta, pred), clamp(self.delta, gt))


########### Latent Navigator ###########
def regressor_criterion(pred, y, eps=1e-12):
    loss = -(
        y * pred.clamp(min=eps).log() + (1 - y) * (1 - pred).clamp(min=eps).log()
    ).mean()
    return loss

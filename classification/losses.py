import torch
import torch.nn as nn
import torch.nn.functional as F

class SCELoss(torch.nn.Module):
    def __init__(self, alpha= 0.1, beta = 1, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # CCE
        ce = self.cross_entropy(pred, target)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = num_classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            # print(true_dist)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class GCELoss(nn.Module):
    def __init__(self, thresh = 0.5, q = 0.7, dim=-1, weight = None):
        super(GCELoss, self).__init__()
        self.thresh = thresh
        self.weight = weight
        self.dim = dim
        self.gce_loss_q = q
    
    def forward(self, pred, target):
        input = pred
        if self.gce_loss_q == 0:
            if input.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(input.view(-1), input.float())
            else:
                ce_loss = nn.CrossEntropyLoss(reduction='none')
                loss = ce_loss(input, target)
        else:
            if input.size(-1) == 1:
                pred = torch.sigmoid(input)
                pred = torch.cat((1-pred, pred), dim=-1)
            else:
                pred = F.softmax(input, dim=-1)
            pred_ = torch.gather(pred, dim=-1, index=torch.unsqueeze(target, -1))
            w = pred_ > self.thresh
            loss = (1 - pred_ ** self.gce_loss_q) / self.gce_loss_q
            # print(pred_, w)
            loss = loss[w].mean()    
        return loss

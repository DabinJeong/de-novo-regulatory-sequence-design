import torch


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        """
        Compute the cross-entropy loss between logits and targets.

        Args:
            logits: Tensor of shape (batch_size, num_classes)
            targets: Tensor of shape (batch_size,) with class indices

        Returns:
            loss: Scalar tensor representing the cross-entropy loss
        """
        loss = self.loss_fn(logits, targets)
        return loss
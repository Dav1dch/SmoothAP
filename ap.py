import torch

def sigmoid(tensor, temp=1.0):
    """temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

class SmoothAP(torch.nn.Module):

    def __init__(self):
        super(SmoothAP, self).__init__()

    def forward(self, sim_all, pos_mask_):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims)"""
        """_summary_

        Example: pred: [1, 0.9, 0.7, 0.6 0.3, 0.2]
        gt: [1, 1, 0, 1, 0, 0] smoothap = 0.833  forward = 0.833
        gt: [1, 1, 1, 1, 0, 0] smoothap = 1      forward = 1
        gt: [1, 1, 0, 1, 0, 1] smoothap = 0.7555 forward = 0.755

        Returns:
            _type_: _description_
        """
        # sim_mask = sim_all[1:]
        sim_mask = sim_all
        pos_mask = pos_mask_[:, 1:]

        if torch.sum(pos_mask) == 0:
            return torch.tensor(0.0001, requires_grad=True)

        d = sim_mask.squeeze().unsqueeze(0)
        d_repeat = d.repeat(len(sim_mask), 1)
        D = d_repeat - d_repeat.T
        D = sigmoid(D, 0.01)
        D_ = D * (1 - torch.eye(len(sim_mask)))
        D_pos = D_ * pos_mask

        R = 1 + torch.sum(D_, 1)
        R_pos = (1 + torch.sum(D_pos, 1)) * pos_mask
        R_neg = R - R_pos
        R = R_neg + R_pos

        ap = torch.zeros(1, requires_grad=True)
        ap_ = (1 / torch.sum(pos_mask)) * torch.sum(R_pos / R)

        ap = ap + ap_

        return ap_

def test():
    sim = torch.tensor([0.9, 0.7, 0.6, 0.3, 0.2])
    gt = torch.tensor([True, True, False, True, False, False]).unsqueeze(0)
    ap =SmoothAP()
    print(ap(sim, gt))

if __name__ == "__main__":
    test()

import torch
import torch.nn as nn
import torch.nn.functional as F


def down_shift(x, pad=None):
    # Removes last row
    xs = [int(y) for y in x.size()]
    x = x[:, :, :xs[2]-1, :]
    pad = nn.ZeroPad2d((0,0,1,0)) if pad is None else pad
    return pad(x)

def right_shift(x, pad=None):
    # Removes last column
    xs = [int(y) for y in x.size()]
    x = x[:, :, :, :xs[3]-1]
    pad = nn.ZeroPad2d((1,0,0,0)) if pad is None else pad
    return pad(x)

def concat_elu(x):
    # Concatenated exponential linear unit.
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))

def log_sum_exp(x):
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_prob_from_logits(x):
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))

def mix_logistic_loss_1d(x, l, likelihood=False):
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + (torch.zeros(xs + [nr_mix]).cuda()).detach()

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    if likelihood:
        log_probs = torch.sum(log_pdf_mid, dim=3) + log_prob_from_logits(logit_probs)
        return log_sum_exp(log_probs)

    log_probs = torch.sum(log_pdf_mid, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))


def mix_logistic_loss(x, l, likelihood=False):
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + (torch.zeros(xs + [nr_mix]).cuda()).detach()
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    mid_in = inv_stdv * centered_x
    log_probs = mid_in - log_scales - 2. * F.softplus(mid_in)

    if likelihood:
        log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
        return log_sum_exp(log_probs)

    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda: one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


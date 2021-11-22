import torch
from utils.helpers import *

def sample_from_discretized_mix_logistic_1d(x, model, nr_mix, u=None, clamp=True):
    # Pytorch ordering
    l = model(x)
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]  # [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # for mean, scale

    # sample mixture indicator from softmax
    if u is None:
        u = l.new_empty(l.shape[0], l.shape[1] * l.shape[2] * (nr_mix + 1))
        u.uniform_(1e-5, 1. - 1e-5)
    mixture_u, sample_u = torch.split(u, [l.shape[1] * l.shape[2] * nr_mix,
                                          l.shape[1] * l.shape[2] * 1], dim=-1)
    mixture_u = mixture_u.reshape(l.shape[0], l.shape[1], l.shape[2], nr_mix)
    sample_u = sample_u.reshape(l.shape[0], l.shape[1], l.shape[2], 1)

    mixture_u = logit_probs.data - torch.log(- torch.log(mixture_u))
    _, argmax = mixture_u.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)

    x = means + torch.exp(log_scales) * (torch.log(sample_u) - torch.log(1. - sample_u))
    if clamp:
        x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    else:
        x0 = x[:, :, :, 0]

    out = x0.unsqueeze(1)
    return out

def sample_from_discretized_mix_logistic(x, model, nr_mix, u=None, T=1, clamp=True):
    # Pytorch ordering
    l = model(x)
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix] / T
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    if u is None:
        u = l.new_empty(l.shape[0], l.shape[1] * l.shape[2] * (nr_mix + 3))
        u.uniform_(1e-5, 1. - 1e-5)

    mixture_u, sample_u = torch.split(u, [l.shape[1] * l.shape[2] * nr_mix,
                                          l.shape[1] * l.shape[2] * 3], dim=-1)
    mixture_u = mixture_u.reshape(l.shape[0], l.shape[1], l.shape[2], nr_mix)
    sample_u = sample_u.reshape(l.shape[0], l.shape[1], l.shape[2], 3)

    mixture_u = logit_probs.data - torch.log(- torch.log(mixture_u))
    _, argmax = mixture_u.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.) + np.log(T)
    coeffs = torch.sum(torch.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling

    x = means + torch.exp(log_scales) * (torch.log(sample_u) - torch.log(1. - sample_u))
    if clamp:
        x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    else:
        x0 = x[:, :, :, 0]

    if clamp:
        x1 = torch.clamp(torch.clamp(
            x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    else:
        x1 = x[:, :, :, 1] + coeffs[:, :, :, 0] * x0

    if clamp:
        x2 = torch.clamp(torch.clamp(
            x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)
    else:
        x2 = x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out
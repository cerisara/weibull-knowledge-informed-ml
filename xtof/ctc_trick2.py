import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


class CustomCTCFunction(Function):
    @staticmethod
    def forward(
        ctx,
        log_prob,
        targets,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    ):
        with torch.enable_grad():
            log_prob.requires_grad_()
            loss = F.ctc_loss(log_prob,
                targets,
                input_lengths,
                target_lengths,
                blank,
                reduction="none",
                zero_infinity=zero_infinity
            )
            ctx.save_for_backward(
                log_prob,
                loss
            )
        ctx.save_grad_input = None
        return loss.clone()

    @staticmethod
    def backward(ctx, grad_output):
        log_prob, loss = (
            ctx.saved_tensors
        )

        if ctx.save_grad_input is None:
            ctx.save_grad_input = torch.autograd.grad(loss, [log_prob], loss.new_ones(*loss.shape))[0]

        gradout = grad_output
        grad_input = ctx.save_grad_input.clone()
        grad_input.subtract_(log_prob.exp()).mul_(gradout.unsqueeze(0).unsqueeze(-1))

        return grad_input, None, None, None, None, None


custom_ctc_fn = CustomCTCFunction.apply


def custom_ctc_loss(
    log_prob,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    """The custom ctc loss. ``log_prob`` should be log probability, but we do not need applying ``log_softmax`` before ctc loss or requiring ``log_prob.exp().sum(dim=-1) == 1``.

    Parameters:
        log_prob (T, N, C): C = number of characters in alphabet including blank
                            T = input length
                            N = batch size
                            log probability of the outputs (e.g. torch.log_softmax of logits)
        targets (N, S): S = maximum number of characters in target sequences
        input_lengths (N): lengths of log_prob
        target_lengths (N): lengths of targets
        blank (int): index of blank tokens (default 0)
        reduction (str): reduction methods applied to the output. 'none' | 'mean' | 'sum'
        zero_infinity (bool): if true imputer loss will zero out infinities.
                              infinities mostly occur when it is impossible to generate
                              target sequences using input sequences
                              (e.g. input sequences are shorter than target sequences)
    """

    loss = custom_ctc_fn(
        log_prob,
        targets,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    )

    if zero_infinity:
        inf = float("inf")
        loss = torch.where(loss == inf, loss.new_zeros(1), loss)

    if reduction == "mean":
        target_length = target_lengths.to(loss).clamp(min=1)

        return (loss / target_length).mean()

    elif reduction == "sum":
        return loss.sum()

    elif reduction == "none":
        return loss

    else:
        raise ValueError(
            f"Supported reduction modes are: mean, sum, none; got {reduction}"
        )


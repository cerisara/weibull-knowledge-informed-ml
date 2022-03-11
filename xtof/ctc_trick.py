# CTC vanilla and CTC via crossentropy are equal, and their gradients as well. In this reformulation it's easier to experiment with modifications of CTC.
# References on CTC regularization:
#  "A Novel Re-weighting Method for Connectionist Temporal Classification", Li et al, https://arxiv.org/abs/1904.10619
#  "Focal CTC Loss for Chinese Optical Character Recognition on Unbalanced Datasets", Feng et al, https://www.hindawi.com/journals/complexity/2019/9345861/
#  "Improved training for online end-to-end speech recognition systems", Kim et al, https://arxiv.org/abs/1711.02212

import torch
import torch.nn.functional as F

## generate example data
# generation is not very stable because of this bug https://github.com/pytorch/pytorch/issues/31557
torch.manual_seed(1)
B, C, T, t, blank = 16, 64, 32, 8, 0
logits = torch.randn(B, C, T).requires_grad_()
input_lengths = torch.full((B,), T, dtype = torch.long)
target_lengths = torch.full((B,), t, dtype = torch.long)
targets = torch.randint(blank + 1, C, (B, t), dtype = torch.long)

## compute CTC alignment targets
log_probs = F.log_softmax(logits, dim = 1)
ctc_loss = F.ctc_loss(log_probs.permute(2, 0, 1), targets, input_lengths, target_lengths, blank = blank, reduction = 'none')

print(ctc_loss.size())
exit()

ctc_grad, = torch.autograd.grad(ctc_loss, (logits,), retain_graph = True)

print(ctc_grad.size())
exit()

temporal_mask = (torch.arange(logits.shape[-1], device = input_lengths.device, dtype = input_lengths.dtype).unsqueeze(0) < input_lengths.unsqueeze(1))[:, None, :]
alignment_targets = (log_probs.exp() * temporal_mask - ctc_grad).detach()
ctc_loss_via_crossentropy = (-alignment_targets * log_probs).sum()
ctc_grad, = torch.autograd.grad(ctc_loss, logits, retain_graph = True)
ctc_grad_via_crossentropy, = torch.autograd.grad(ctc_loss_via_crossentropy, logits, retain_graph = True)

assert torch.allclose(ctc_grad, ctc_grad_via_crossentropy, rtol = 1e-3)



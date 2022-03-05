import torch
import torch.nn as nn

class TSProjector(nn.Module):
    def __init__(self,a):
        super(TSProjector, self).__init__()
        self.b = 10
        self.c = torch.rand(self.b,100)
        self.mlpx = nn.Linear(a,self.b)
        # c = (b,100)

    def forward(self,x):
        # x = (B,T,a)
        # assign each timestep to one of the 100 with Viterbi
        B = x.size(0)
        z = self.mlpx(x)
        # z = (B,T,b)
        st  = torch.bmm(z,c)
        # st = (B,T,100)
        stt = torch.zeros(B,100)
        

    def training_step(self,x,y):
        # 1- CTC loss for a few epochs to train to align

        # 2- RUL-pred (MSE) loss after Viterbi align

    def beam_search_decoder(post, k):
    """Beam Search Decoder

    Parameters:

        post(Tensor) – the posterior of network.
        k(int) – beam size of decoder.

    Outputs:

        indices(Tensor) – a beam of index sequence.
        log_prob(Tensor) – a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)

    """

    batch_size, seq_length, _ = post.shape
    log_post = post.log()
    log_prob, indices = log_post[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1)
        log_prob, index = log_prob.view(batch_size, -1).topk(k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
    return indices, log_prob


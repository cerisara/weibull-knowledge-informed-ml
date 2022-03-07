import torch
import torch.nn as nn

class TSProjector(nn.Module):
    def __init__(self,a):
        super(TSProjector, self).__init__()
        self.b = 10
        self.c = torch.rand(self.b,100)
        self.mlpx = nn.Linear(a,self.b)
        # c = (b,100)
        self.VitTransMat1 = torch.eye(100)
        self.VitTransMat2 = torch.zeros(100,100)
        for i in range(0,99): self.VitTransMat2[i,i+1]=1.

    def forward(self,x):
        # x = (B,T,a)
        # assign each timestep to one of the 100 with Viterbi
        B,T = x.size(0),x.size(1)
        z = self.mlpx(x)
        # z = (B,T,b)
        # now compute distances btw every input frame and every 100 tgt frame:
        allsims = torch.matmul(z,self.c).detach()
        # allsims = (B,T,b) * (b,100) = (B,T,100)
        # now Viterbi (outside the pytorch graph - non differentiable)
        st = allsims[:,0,:]
        bt = []
        for t in range(1,T):
            stt1 = torch.matmul(st,self.VitTransMat1) # (B,100) * (100,100) = (B,100)
            stt2 = torch.matmul(st,self.VitTransMat2) # (B,100) * (100,100) = (B,100)
            stt3 = torch.stack((stt1,stt2),dim=0) # (2,B,100)
            st, sttidx = torch.max(stt3,dim=0) # (B,100)
            # TODO: force au debut sttidx=1 dans le coin superieur gauche
            bt.append(sttidx)
            st += allsims[:,t,:]
            

    def training_step(self,x,y):
        # 1- CTC loss for a few epochs to train to align

        # 2- RUL-pred (MSE) loss after Viterbi align
        pass

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

def toytest():
    mod=TSProjector(7)
    x = torch.rand(3,13,7)
    mod(x)

toytest()


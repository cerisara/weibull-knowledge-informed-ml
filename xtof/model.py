import torch
import torch.nn as nn

class TSProjector(nn.Module):
    def __init__(self,a):
        super(TSProjector, self).__init__()
        self.b = 10
        self.d = 10
        # centroides pour chacune des 100 trames target
        self.c = torch.rand(self.b,100) # c = (b,100)
        # mlpx encode les inputs en vecteurs de dim b
        self.mlpx = nn.Linear(a,self.b)
        # matrices de transitions diagonales pour Viterbi (ce ne sont pas des parametres)
        self.VitTransMat1 = torch.eye(100)
        self.VitTransMat2 = torch.zeros(100,100)
        for i in range(0,99): self.VitTransMat2[i,i+1]=1.
        # projection dependant de chacune des 100 trames target
        self.proj2tgt = nn.Linear(100,self.b,self.d)

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
            # force sttidx=1 (changement d'etat) dans le coin superieur gauche
            sttidx[:,t:] = 1
            bt.append(sttidx)
            st += allsims[:,t,:]
        # backtrack
        states=torch.zeros(B,T).long()
        states[:,T-1]=100-1
        for t in range(T-1,0,-1):
            btprev = bt[t-1]
            for b in range(B):
                if btprev[b,states[b,t]] == 0: prev=states[b,t]
                else: prev=states[b,t]-1
                if b==0: print("dbug",t,states[b,t],btprev[b,states[b,t]])
                states[b,t-1]= prev
      
        for t in range(T):
            print("VIT",t,states[0,t].item())
        exit()
  
        # avec cet alignement, on peut projeter 

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
    x = torch.rand(3,130,7)
    mod(x)

toytest()


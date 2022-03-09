import torch
import torch.nn as nn
import pytorch_lightning as pl

class TSProjector(pl.LightningModule):
    def __init__(self,a):
        super(TSProjector, self).__init__()
        self.b = 10
        # centroides pour chacune des 100 trames target
        self.c = torch.rand(self.b,100) # c = (b,100)
        # mlpx encode les inputs en vecteurs de dim b
        self.mlpx = nn.Sequential(nn.Linear(a,self.b),nn.ReLU(),
                                nn.Linear(self.b,self.b),nn.ReLU(),
                                nn.Linear(self.b,self.b))
        # matrices de transitions diagonales pour Viterbi (ce ne sont pas des parametres)
        self.VitTransMat1 = torch.eye(100)
        self.VitTransMat2 = torch.zeros(100,100)
        for i in range(0,99): self.VitTransMat2[i,i+1]=1.
        # LSTM sur les 100 trames
        self.h = 50
        self.lstm = nn.LSTM(self.b,self.h,batch_first=True)
        # MLP to predict RUL
        self.d = 30
        self.mlprul = nn.Sequential(nn.Linear(self.h,self.d),nn.ReLU(),
                                nn.Linear(self.d,self.d),nn.ReLU(),
                                nn.Linear(self.d,1))
        # for training:
        self.stage = -1
        self.mseloss = nn.MSELoss()
        self.ctcloss = nn.CTCLoss(blank=100)
        self.tgtseq = torch.tensor([i for i in range(100)]).long()

    def forward(self,x):
        # x = (B,T,a)
        # assign each timestep to one of the 100 with Viterbi
        B,T = x.size(0),x.size(1)
        z = self.mlpx(x)
        # z = (B,T,b)
        # now compute distances btw every input frame and every 100 tgt frame:
        allsims = torch.matmul(z,self.c)
        # allsims = (B,T,b) * (b,100) = (B,T,100)
        # now Viterbi 
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
                states[b,t-1]= prev
        # for t in range(T):
        #     print("VIT",t,states[0,t].item())
 
        # on a ici l'alignement global optimal selon le score = \sum_t,u sim(X_t,S_u) 
        # chacun des 100 centroides est donc aligne avec un ensemble de X_t
        # on a donc une segmentation de la longue sequence (X_t)
        # on reduit cette longue seq en moyennant selon cette segmentation
        zz = torch.zeros(B,100,self.b)
        for b in range(B):
            ns = torch.bincount(states[b])
            for t in range(T):
                zz[b,states[b,t]] += z[b,t]
            for t in range(100): zz[b,t] /= ns[t].float()
        # puis on applique notre LSTM
        y,_ = self.lstm(zz)
        # y = (B,100,h)
        rul = self.mlprul(y[:,-1,:].view(B,-1))
        return rul

    def shallTrainAlign(self):
        # alternate training align and prediction
        # TODO: train both together E2E ?
        self.stage += 1
        if self.stage<=10: return True
        if self.stage<=20: return False
        self.stage = 0
        return True

    def alignLoss(self,x,length):
        # x = (B,T,inputdim)
        # length = (B,)
        B,T = x.size(0),x.size(1)
        goldy = self.tgtseq.expand(B,100)
        z = self.mlpx(x)
        allsims = torch.matmul(z,self.c)
        # allsims = (B,T,b) * (b,100) = (B,T,100)
        blanks = torch.full((B,T,1),-9999.)
        pp = torch.cat((allsims,blanks),dim=2)
        logprobs = nn.functional.log_softmax(pp,dim=2).permute(1,0,2)
        # logprobs = (T,B,101)
        loss = self.ctcloss(logprobs,goldy,length,torch.full((B,),100))
        return loss

    def rulLoss(self,x,y):
        haty = self.forward(x)
        return self.mseloss(haty,y)

    def training_step(self,batch,batch_idx):
        x,rul,length = batch
        # 1- CTC loss for a few epochs to train to align
        # 2- RUL-pred (MSE) loss after Viterbi align
        if self.shallTrainAlign(): return self.alignLoss(x,length)
        else: return self.rulLoss(x,rul)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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

class ToyData(torch.utils.data.Dataset):
    def __init__(self):
        self.data = []
        # generate a total of 20 random samples
        for i in range(20):
            x = torch.rand(130,7)
            rul = torch.rand(1)
            # 3rd item = length of the seq (useful when padding seqs)
            self.data.append((x,rul,130))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        return self.data[i]

def toytest():
    mod=TSProjector(7)
    data = ToyData()
    params = {'batch_size': 2, 'shuffle': True, 'num_workers': 1}
    trainD = torch.utils.data.DataLoader(data,**params)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(mod, trainD)

toytest()


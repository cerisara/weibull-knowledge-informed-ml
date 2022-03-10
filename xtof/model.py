import torch
import torch.nn as nn
import pytorch_lightning as pl
import data

# import logging
# loger = logging.getLogger('pytorch_lightning')
# loger.setLevel(logging.DEBUG)

# the later you wait, the better the tools
# the sooner you start, the better your tools

# pour visualiser avec tensorboard:
# tensorboard --logdir ./lightning_logs/

class TSProjector(pl.LightningModule):
    def __init__(self,a):
        super(TSProjector, self).__init__()

        #### CNN+DAN pour chaque seconde = 20k samples
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(4, 4, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(4, 4, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.dan1 = nn.AdaptiveAvgPool1d(1)   # (B,4,L) -> (B,4,1)
        self.dan2 = nn.Sequential(
                nn.Linear(4,a),nn.ReLU(),
                nn.Linear(a,a),nn.ReLU(),
                nn.Linear(a,a))

        #### Modele de projection vers un 100-LSTM
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

    def forward(self,x0):
        # TODO: prendre exemple sur ce lien pour plus rapide ? https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html

        #### ConvNet sur les 1s-segments
        # x0 = (B,Nseqs,Lseq,1) Lseq=20k
        B,T,L = x0.size(0),x0.size(1),x0.size(2)
        z = self.cnn(x0.view(B*T,L,1).permute(0,2,1))
        # z = (B*T,4,L')
        zz = self.dan1(z)
        # zz = (B*T,4,1)
        zz = self.dan2(zz.view(B*T,-1))
        # zz = (B*T,a)
        x = zz.view(B,T,-1)

        #### Projection vers 100-LSTM
        # x = (B,T,a)

        # assign each timestep to one of the 100 with Viterbi
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
                states[b,t-1]= max(0,prev)
        for b in range(B):
            for t in range(T):
                if states[b,t]<0 or states[b,t]>99: print("BUG",b,t,states[b,t])
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

        #### 100-LSTM pour predire les RULs

        y,_ = self.lstm(zz)
        # y = (B,100,h)
        ruls = self.mlprul(y)
        return ruls

    def shallTrainAlign(self):
        # alternate training align and prediction
        # TODO: train both together E2E ?
        self.stage += 1
        if self.stage<=10: return True
        if self.stage<=20: return False
        self.stage = 0
        return True

    def alignLoss(self,x0,length):
        # x0 = (B,Nseqs,Lseq,1) Lseq=20k
        # length = (B,)
        B,T,L = x0.size(0),x0.size(1),x0.size(2)
        z = self.cnn(x0.view(B*T,L,1).permute(0,2,1))
        # z = (B*T,4,L')
        zz = self.dan1(z)
        # zz = (B*T,4,1)
        zz = self.dan2(zz.view(B*T,-1))
        # zz = (B*T,a)
        x = zz.view(B,T,-1)
        # x = (B,T,inputdim)
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
        print("ctcloss",loss.item())
        return loss

    def rulLoss(self,x,y):
        haty = self.forward(x)
        loss = self.mseloss(haty,y)
        print("rulloss",loss.item())
        return loss

    def training_step(self,batch,batch_idx):
        x,rul,length = batch
        # 1- CTC loss for a few epochs to train to align
        # 2- RUL-pred (MSE) loss after Viterbi align
        if self.shallTrainAlign():
            train_loss= self.alignLoss(x,length)
            self.log("cctloss",train_loss)
        else:
            train_loss= self.rulLoss(x,rul)
            self.log("rulloss",train_loss)
        return train_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

class IMSData(torch.utils.data.Dataset):
    def __init__(self):
        allseqs, allruls = data.loadTrain()
        # allseqs = [Nseqs, Lseq=20k]
        # on veut (Nseqs,Lseq,1) Lseq=20k
        self.data = [torch.tensor(allseqs).unsqueeze(2)]
        self.ruls = [torch.tensor(allruls)]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        return self.data[i],self.ruls[i],self.data[i].size(0)

class ToyData(torch.utils.data.Dataset):
    def __init__(self):
        self.data = []
        # generate a total of 20 random samples
        for i in range(20):
            x = torch.rand(1000,7)
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
    print("data loaded")
    params = {'batch_size': 2, 'shuffle': True, 'num_workers': 1}
    trainD = torch.utils.data.DataLoader(data,**params)
    trainer = pl.Trainer(max_epochs=1000, log_every_n_steps=1)
    trainer.fit(mod, trainD)

def imstest():
    mod=TSProjector(100)
    data = IMSData()
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1}
    trainD = torch.utils.data.DataLoader(data,**params)
    trainer = pl.Trainer(max_epochs=1000, log_every_n_steps=1)
    trainer.fit(mod, trainD)


imstest()


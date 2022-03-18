import torch
import torch.nn as nn
import pytorch_lightning as pl
import data

class ourLSTM(pl.LightningModule):
    def __init__(self,a):
        super(ourLSTM, self).__init__()

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

        self.h = 50
        self.lstm = nn.LSTM(a,self.h,batch_first=True)
        # MLP to predict RUL
        self.d = 30
        self.mlprul = nn.Sequential(nn.Linear(self.h,self.d),nn.ReLU(),
                                nn.Linear(self.d,self.d),nn.ReLU(),
                                nn.Linear(self.d,1))
        # for training:
        self.trainingsteps = 0
        self.mseloss = nn.MSELoss()

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
        # x = (B,T,a)

        y,_ = self.lstm(x)
        # y = (B,T,h)
        ruls = self.mlprul(y)
        # ruls = (B,T,1)
        return ruls

    def training_step(self,batch,batch_idx):
        x,rul,length = batch
        haty = self.forward(x)
        train_loss = self.mseloss(haty,y)
        self.log("train_rul_loss",train_loss)
        return train_loss

    def validation_step(self,batch,batch_idx):
        x,rul,length = batch
        haty = self.forward(x)
        dev_loss = self.mseloss(haty,y)
        self.log("dev_rul_loss",dev_loss)
        return dev_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
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

class IMSDev(torch.utils.data.Dataset):
    def __init__(self):
        allseqs, allruls = data.loadDev()
        # allseqs = [Nseqs, Lseq=20k]
        # on veut (Nseqs,Lseq,1) Lseq=20k
        self.data = [torch.tensor(allseqs).unsqueeze(2)]
        self.ruls = [torch.tensor(allruls)]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        return self.data[i],self.ruls[i],self.data[i].size(0)


def imstest():
    mod=ourLSTM(100)
    data = IMSData()
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1}
    trainD = torch.utils.data.DataLoader(data,**params)
    dev = IMSDev()
    devD = torch.utils.data.DataLoader(dev,**params)
    trainer = pl.Trainer(max_epochs=1000000, log_every_n_steps=1)
    trainer.fit(mod, trainD, devD)


imstest()


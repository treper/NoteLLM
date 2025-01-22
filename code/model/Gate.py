import torch.nn as nn
import torch

class Gate(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.trans_all = nn.Linear(dim*2,dim,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            nn.init.xavier_normal_(module.weight)
    
    def forward(self,image_emb,text_emb):
        ori_emb = image_emb
        whole = torch.cat([image_emb,text_emb],1)
        z = self.sigmoid(self.trans_all(whole))
        emb = z*image_emb + (1-z)*text_emb
        return emb+ori_emb

class GatewLn(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.trans1 = nn.Linear(dim,dim,bias=False)
        self.trans2 = nn.Linear(dim,dim,bias=False)
        self.trans_all = nn.Linear(dim*2,dim,bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            nn.init.xavier_normal_(module.weight)
    
    def forward(self,image_emb,text_emb):
        ori_emb = image_emb
        whole = torch.cat([image_emb,text_emb],1)
        z = self.sigmoid(self.trans_all(whole))
        image_emb = self.tanh(self.trans1(image_emb))
        text_emb = self.tanh(self.trans2(text_emb))
        emb = z*image_emb + (1-z)*text_emb
        return emb+ori_emb
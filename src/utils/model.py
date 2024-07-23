import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np

from . import GlobalParameters
from . import CustomDataLoader

###########
## Model ##
###########

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()

        # 2*n + 1 + 2 parameter is sequence length (both sides of base, base, and start and end token)
        self.seq_length = (GlobalParameters.n * 2) + 1 + 2

        self.dropout = nn.Dropout(p=GlobalParameters.dropout)
        ## Embeddings
        
        # Position embedding
        self.linear = nn.Linear(len(CustomDataLoader.token_dict), GlobalParameters.d_model) # nn.Linear(4, d_model)
        self.pos_embed = Attention(GlobalParameters.d_model, GlobalParameters.nhead, max_len=self.seq_length, dropout=GlobalParameters.dropout)
        
        # Token embedding
        self.cnn = torch.nn.Conv2d(1, GlobalParameters.d_model, ( 1, GlobalParameters.d_model),  bias=False)
        
        # Normalize
        self.normalize = nn.LayerNorm(GlobalParameters.d_model, dtype=torch.float32)
        
        ## Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=GlobalParameters.d_model, nhead=GlobalParameters.nhead, dtype=torch.float32, batch_first=True, dropout = GlobalParameters.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=GlobalParameters.num_layers)
        
        ## Decoder
        self.decoder = nn.Linear(GlobalParameters.d_model, 1, dtype=torch.float32)
        self.decoder2 = nn.Linear(self.seq_length, 1, dtype=torch.float32)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        ## Embeddings 
        pos_emb = self.dropout(self.linear(x))
        # pos_emb = self.embedding(torch.argmax(x, dim=2))

        pos_emb, attn = self.pos_embed(pos_emb)

        
        np.save(f'{GlobalParameters.output_directory}/x.npy', x.cpu().detach().numpy())
        np.save(f'{GlobalParameters.output_directory}/temp.npy', attn.cpu().detach().numpy())
        
        token_emb = self.dropout(self.cnn(pos_emb.unsqueeze(1)).squeeze())
        
        # Add batch dimension if there's only one in batch
        if len(token_emb.shape) == 2:
            token_emb = token_emb.unsqueeze(0)
        token_emb = token_emb.permute(0,2,1)
        
        x = self.normalize(token_emb + pos_emb) 
        
        x = self.encoder(x)
        x = self.dropout(self.decoder(x).squeeze())
        x = self.decoder2(x).squeeze()
        
        x = self.sigmoid(x)  

        return x
    
class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1025, dropout=0):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder: raise ValueError("incompatible `d_model` and `num_heads`")
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer( "mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len: raise ValueError("sequence length exceeds model capacity")

        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1) #(batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        QEr = torch.matmul(q, Er_t)
        Srel = self.skew(QEr)
        QK_t = torch.matmul(q, k_t)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        self.attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(self.attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        return self.dropout(out)
    def skew(self, QEr):
        padded = F.pad(QEr, (1, 0))
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        Srel = reshaped[:, :, 1:, :]
        return Srel      

class Attention(RelativeGlobalAttention):
    def forward(self, x):
        result = super().forward(x)
        return result, self.attn
        
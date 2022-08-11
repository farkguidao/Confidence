import torch
import torch.nn as nn

from models.encoder import BiLSTMLayer
from utils.self_attention import ScaledDotProductAttention


class SpeakerAwareAttention(nn.Module):
    def __init__(self,d_model,nhead,dropout):
        super(SpeakerAwareAttention, self).__init__()
        dk = d_model//nhead
        self.same_attention = ScaledDotProductAttention(d_model,dk,dk,nhead,dropout=dropout)
        self.other_attention = ScaledDotProductAttention(d_model,dk,dk,nhead,dropout=dropout)

    def forward(self,x,same_mask,other_mask,A):
        return self.same_attention.forward(x,x,x,same_mask,attention_weights=A)\
               +self.other_attention.forward(x,x,x,other_mask,attention_weights=A)


def generate_edge_em(adj,len_list,pos,p):
    '''
    :param adj: bs,seq_len,seq_len
    :param len_list:
    :param pos: seq_len,seq_len
    :param p: bs,seq_len,seq_len,d
    :return:
    '''
    bs,seq_len = adj.shape[0], adj.shape[-1]

    # 自环
    index = torch.arange(seq_len)
    adj[:, index, index] = -1
    key_padding_mask = torch.zeros_like(adj).bool()
    for i, length in enumerate(len_list):
        key_padding_mask[i, :, length:] = True

    em = torch.cat([p,
                    pos.unsqueeze(-1).unsqueeze(0).expand(bs,seq_len,seq_len,1),
                    adj.unsqueeze(-1).expand(bs,seq_len,seq_len,1)],dim=-1)
    return em,key_padding_mask




class ConfidenceDecoder(nn.Module):
    def __init__(self,num_sent: int,
                 num_act: int,
                 input_dim: int,
                 hidden_dim: int,
                 nhead: int,
                 num_layer: int,
                 dropout_rate: float):
        super(ConfidenceDecoder, self).__init__()
        self.num_act = num_act
        self.num_sent = num_sent


        self.transfer_sent_linear = nn.Linear(input_dim, hidden_dim)
        self.transfer_act_linear = nn.Linear(input_dim, hidden_dim)

        self.final_sent_linear = nn.Linear(hidden_dim, num_sent)
        self.final_act_linear = nn.Linear(hidden_dim, num_act)

        dk = hidden_dim//nhead
        self.act_layers = nn.ModuleList([SpeakerAwareAttention(hidden_dim,nhead,dropout=dropout_rate) for _ in range(num_layer)])
        self.sent_layers = nn.ModuleList(
            [SpeakerAwareAttention(hidden_dim, nhead, dropout=dropout_rate) for _ in range(num_layer)])

        # self.act_layers = nn.ModuleList(
        #     [ScaledDotProductAttention(hidden_dim,dk,dk,nhead,dropout=dropout_rate) for _ in range(num_layer)])
        # self.sent_layers = nn.ModuleList(
        #     [ScaledDotProductAttention(hidden_dim, dk, dk, nhead, dropout=dropout_rate) for _ in range(num_layer)])

        self.sent_A_generator = nn.Sequential(nn.Linear(2*(num_sent+num_act)+hidden_dim, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim,nhead),
                                              nn.Sigmoid())

        self.act_A_generator = nn.Sequential(nn.Linear(2*(num_sent+num_act)+hidden_dim, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, nhead),
                                              nn.Sigmoid())

        self.max_len = 100
        self.postion_em = nn.Embedding(self.max_len*2,hidden_dim)
        # self.type_em = nn.Embedding(2,hidden_dim)

        a = torch.arange(self.max_len).expand(self.max_len,self.max_len)
        self.register_buffer('pos_m',a-a.T+self.max_len)
        # self.register_buffer('pos_m',a-a.T)

        # self.act_lstm = BiLSTMLayer(hidden_dim,dropout_rate)
        # self.sent_lstm = BiLSTMLayer(hidden_dim,dropout_rate)
        self.nhead = nhead

    def forward(self,sent_x,act_x,sent_logits,act_logits,len_list,adj):
        '''
        :param sent_x:
        :param act_x:
        :param sent_logits:
        :param act_logits:
        :param len_list:
        :param adj:
        :return:
        '''

        # step one

        # p = torch.cat([torch.softmax(sent_logits, -1), torch.softmax(act_logits, -1)],-1)  # bs,seq_len,num_sent+num_act

        # 测试去除softmax
        p = torch.cat([sent_logits, act_logits], -1)  # bs,seq_len,num_sent+num_act
        #step two

        # initial inputs
        sent_x, act_x = self.transfer_sent_linear(sent_x), self.transfer_act_linear(act_x)

        sent_x_,act_x_ = sent_x,act_x
        # generate A
        bs, seq_len, hidden_dim = sent_x.size()
        p = torch.cat([p.reshape(bs, seq_len, 1, -1).expand(bs, seq_len, seq_len,self.num_act+self.num_sent),
                       p.reshape(bs, 1, seq_len, -1).expand(bs, seq_len, seq_len, self.num_act+self.num_sent)],-1)  # bs,seq_len,seq_len,2*(num_sent+num_act)




        pos_em = self.postion_em(self.pos_m[:seq_len, :seq_len]).expand(bs, seq_len,
                                                                        seq_len,hidden_dim)  # bs,seq_len,seq_len,hidden_dim
        em = torch.cat([p,pos_em],-1) # 2*(num_sent+num_act) + hidden_dim

        sent_A = self.sent_A_generator(em).reshape(bs,seq_len,seq_len,self.nhead,) # bs,seq_len,seq_len,nhead,hidden_dim//nhead
        act_A = self.act_A_generator(em).reshape(bs,seq_len,seq_len,self.nhead,)  # bs,seq_len,seq_len,nhead
        sent_A = sent_A.transpose(2,3).transpose(1,2) # bs,nhead, ,seq_len
        act_A = act_A.transpose(2, 3).transpose(1, 2)  # bs,nhead,seq_len,seq_len,hidden_dim//nhead

        # generate same edge mask and other edge mask
        # padding_mask

        same_mask = adj==0
        other_mask = ~same_mask
        for i,length in enumerate(len_list):
            other_mask[i,:,length:] = True
            same_mask[i,:,length:] = True
        # 添加自环
        index = torch.arange(seq_len)
        other_mask[:,index,index]=False
        same_mask[:,index,index]=False
        same_mask,other_mask = same_mask.unsqueeze(1), other_mask.unsqueeze(1) # bs,nhead,seq_len,seq_len

        # sent_x, act_x = sent_x + self.sent_lstm(sent_x), act_x + self.act_lstm(act_x)

        # graph convolution
        for sent_layer,act_layer in zip(self.sent_layers,self.act_layers):
            sent_x = sent_x + torch.relu(sent_layer(x=sent_x,same_mask=same_mask,other_mask=other_mask,A=sent_A))
            act_x = act_x + torch.relu(act_layer(x=act_x, same_mask=same_mask, other_mask=other_mask, A=act_A))

        # one = torch.ones_like(adj[0],dtype=torch.bool)
        # windows_mask = one.triu(5 + 1) + one.triu(5 + 1).T
        # mask = torch.zeros_like(adj,dtype=torch.bool)
        # for i,length in enumerate(len_list):
        #     mask[i,:,length:]=True
        # # mask = mask | windows_mask
        # mask=mask.unsqueeze(1)
        # sent_x_, act_x_ = sent_x, act_x
        # for sent_layer,act_layer in zip(self.sent_layers,self.act_layers):
        #     sent_x = sent_x + torch.relu(sent_layer(sent_x,sent_x,sent_x,mask,attention_weights=sent_A))
        #     act_x = act_x + torch.relu(act_layer(act_x,act_x,act_x,mask,attention_weights=act_A))


        final_sent_logits, final_act_logits = self.final_sent_linear(sent_x_+sent_x),self.final_act_linear(act_x_+act_x)


        return final_sent_logits, final_act_logits






if __name__ == '__main__':
    x = torch.randn(32,50,128)
    adj = torch.randn(32,50,50)
    adj[adj<0]=0
    len_list = torch.randint(10,50,(32,))
    decoder = ConfidenceDecoder(3,5,128,64,4,3,0.1)

    result = decoder(x,x,len_list,adj)











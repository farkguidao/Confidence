from typing import Union, List, Optional, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch.nn.functional as F
from torchmetrics import F1Score,Precision,Recall,MetricCollection

from models.decoder import ConfidenceDecoder
from models.encoder import BiEncoder


class Model(pl.LightningModule):
    def __init__(self,dataset, encoder_hidden_dim, decoder_hidden_dim, pretrained_model,
                 num_sent, num_act, nhead, num_layer,garma,lr,wd,dropout, embedding_matrix=None):
        '''
        :param dataset:
        :param encoder_hidden_dim:
        :param decoder_hidden_dim:
        :param pretrained_model:
        :param num_sent:
        :param num_act:
        :param nhead:
        :param num_layer:
        :param lr:
        :param wd:
        :param dropout:
        :param embedding_matrix:
        '''
        super(Model, self).__init__()
        self.save_hyperparameters()

        # 评价器与glove
        if dataset=='dailydialogue':
            # macro
            self.act_metric = MetricCollection([F1Score(num_classes=num_act, average='macro'),
                                                Precision(num_classes=num_act, average='macro'),
                                                Recall(num_classes=num_act, average='macro')],prefix='act')
            self.sent_metric = MetricCollection([F1Score(num_classes=num_sent, average='macro'),
                                                Precision(num_classes=num_sent, average='macro'),
                                                Recall(num_classes=num_sent, average='macro')],prefix='sent')

        if dataset=='mastodon':
            # macro
            self.act_metric = MetricCollection([F1Score(num_classes=num_act, average='weighted'),
                                                Precision(num_classes=num_act, average='weighted'),
                                                Recall(num_classes=num_act, average='weighted')],prefix='act')
            self.sent_metric = MetricCollection([F1Score(num_classes=num_sent,ignore_index=2, average='macro'),
                                                Precision(num_classes=num_sent,ignore_index=2, average='macro'),
                                                Recall(num_classes=num_sent,ignore_index=2, average='macro')],prefix='sent')


        word_embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float),
                                                             freeze=False)
        self.encoder = BiEncoder(word_embedding, encoder_hidden_dim,num_sent,num_act,nhead,dropout,pretrained_model)
        self.decoder = ConfidenceDecoder(num_sent,num_act,encoder_hidden_dim,decoder_hidden_dim,nhead,num_layer,dropout)

        self.act_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.sent_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.margin_loss = MarginLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.wd)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30],
                                                            gamma=0.1)
        return [optimizer],[lr_scheduler]
        # return optimizer



    def forward(self, *args, **kwargs) -> Any:
        var_uttn, var_p, len_list,mask,adj= args
        if self.hparams.pretrained_model != 'none':
            sent_x, act_x, sl1,al1 = self.encoder(var_p,adj,mask)
        else:
            sent_x, act_x, sl1,al1 = self.encoder(var_uttn,adj,mask)
        sl2,al2 = self.decoder(sent_x,act_x,sl1.detach(),al1.detach(),len_list,adj)

        sl1,al1 = self.flaten(sl1,len_list),self.flaten(al1,len_list)
        sl2,al2 = self.flaten(sl2,len_list),self.flaten(al2,len_list)
        # bs,d_len,num

        return sl1,al1,sl2,al2

    # 加padding和去padding的问题，
    def flaten(self,logits,len_list):
        logits_list = [ i[:j] for i,j in zip(logits,len_list)]
        return torch.cat(logits_list,dim=0)


# 加参数 loss1 loss2

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        batch=args[0]
        var_uttn, var_p, len_list, mask, adj, sent_label,act_label = batch
        sl1, al1, sl2, al2 = self(var_uttn, var_p, len_list, mask, adj)

        loss1 = self.sent_criterion(sl1,sent_label)+self.act_criterion(al1,act_label)
        loss2 = self.sent_criterion(sl2, sent_label) + self.act_criterion(al2, act_label)
        # margen_loss = self.margin_loss(sl1,sl2,sent_label) + self.margin_loss(al1,al2,act_label)
        # loss = loss1+loss2+self.hparams.garma*margen_loss
        loss = self.hparams.garma*loss1+loss2
        self.log('loss',loss)
        return loss

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        batch = args[0]
        var_uttn, var_p, len_list, mask, adj, sent_label, act_label = batch
        sl1, al1, sl2, al2 = self(var_uttn, var_p, len_list, mask, adj)

        sent_pre,act_pre=sl2.max(-1)[1],al2.max(-1)[1]
        self.sent_metric.update(sent_pre,sent_label)
        self.act_metric.update(act_pre, act_label)
        return super().validation_step(*args, **kwargs)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.log_dict(self.sent_metric.compute())
        self.log_dict(self.act_metric.compute())
        self.sent_metric.reset()
        self.act_metric.reset()
        super().validation_epoch_end(outputs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.validation_epoch_end(outputs)


class MarginLoss(nn.Module):
    def forward(self,initial,final,lable):
        return torch.sum(torch.index_select(F.relu(F.log_softmax(initial, dim=-1) \
                                                           - F.log_softmax(final, dim=-1)), 1,
                                                    lable))
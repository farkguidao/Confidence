import argparse
import os
from typing import Union, List, Optional, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from torchmetrics import F1Score,Precision,Recall,MetricCollection

from dataloader.dataloader import Dataloader
from models.decoder import ConfidenceDecoder
from models.encoder import BiEncoder
from models.model import Model

def test(model, trainer, dataloader, ckpt_path):
    # model.set_final_step()
    ckpt_base_path = ckpt_path + '/checkpoints/'

    print('------------------------------act_score--------------------------------------------------')
    res_act = trainer.test(model, dataloader, ckpt_path=ckpt_base_path + 'act.ckpt', verbose=False)[0]

    print('------------------------------sent_score-------------------------------------------------')
    res_sent = trainer.test(model, dataloader, ckpt_path=ckpt_base_path + 'sent.ckpt', verbose=False)[0]

    res = {'actF1Score': res_act['actF1Score'],
           'actPrecision': res_act['actPrecision'],
           'actRecall': res_act['actRecall'],
           'sentF1Score': res_sent['sentF1Score'],
           'sentPrecision': res_sent['sentPrecision'],
           'sentRecall': res_sent['sentRecall'],
           }
    return res

if __name__ == '__main__':
    # args = {
    #     'dataset': 'dailydialogue',
    #     'data_path': 'data/dailydialogue',
    #     'batch_size': 32,
    #     'pretrained_model': 'none',
    #
    #     'encoder_hidden_dim': 128,
    #     'decoder_hidden_dim': 64,
    #     'nhead': 4,
    #     'num_layer': 3,
    #     'garma': 1.,
    #     'lr': 1e-3,
    #     'wd': 1e-8,
    #     'dropout': 0.2,
    #     'max_epochs': 30,
    #     'gpus': 1
    #
    # }

    args={
        'dataset': 'mastodon',
        'data_path': 'data/mastodon',
        'batch_size': 16,
        'pretrained_model': 'none',

        'encoder_hidden_dim': 256,
        'decoder_hidden_dim': 64,
        'nhead': 4,
        'num_layer': 2,
        'garma': 1.,
        'lr': 1e-3,
        'wd': 1e-8,
        'dropout': 0.2,
        'max_epochs': 30,
        'gpus': 1
    }
    args = argparse.Namespace(**args)

    dataloader = Dataloader(dataset=args.dataset,
                            data_path=args.data_path,
                            batch_size=args.batch_size,
                            pretrained_model=args.pretrained_model)
    model = Model(dataset=args.dataset,
                  encoder_hidden_dim=args.encoder_hidden_dim,
                  decoder_hidden_dim=args.decoder_hidden_dim,
                  pretrained_model=args.pretrained_model,
                  num_sent=dataloader.num_sent,
                  num_act=dataloader.num_act,
                  nhead=args.nhead,
                  num_layer=args.num_layer,
                  garma=args.garma,
                  lr=args.lr,
                  wd=args.wd,
                  dropout=args.dropout,
                  embedding_matrix=dataloader.initial_embedding_matrix)

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         gpus=args.gpus,
                         log_every_n_steps=1)
    # res = test(model, trainer, dataloader.test_dataloader(), 'lightning_logs/best/version_120')
    res = test(model, trainer, dataloader.test_dataloader(), 'lightning_logs/best/version_269')

    print('--------test result------')
    print(res)






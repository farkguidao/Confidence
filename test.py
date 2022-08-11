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

if __name__ == '__main__':
    args = {
        'dataset': 'mastodon',
        'data_path': 'data/mastodon',
        'batch_size': 16,
        'pretrained_model': 'none',

        'encoder_hidden_dim': 128,
        'decoder_hidden_dim': 128,
        'nhead': 8,
        'num_layer': 2,
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
                  lr=args.lr,
                  wd=args.wd,
                  dropout=args.dropout,
                  embedding_matrix=dataloader.initial_embedding_matrix)

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         gpus=args.gpus,
                         log_every_n_steps=1)
    # trainer.fit(model,dataloader)
    # ckpt_path = trainer.log_dir + '/checkpoints/' + os.listdir(trainer.log_dir + '/checkpoints')[0]
    ckpt_path  = 'lightning_logs/version_0/checkpoints/epoch=9-step=160.ckpt'
    # trainer.validate(model, dataloader.test_dataloader(), ckpt_path=ckpt_path)
    dl=dataloader.test_dataloader()
    model.set_final_step()
    res=trainer.test(model, dl, ckpt_path='lightning_logs/version_43/checkpoints/act.ckpt')


    # trainer.test(model, dl, ckpt_path=ckpt_path)





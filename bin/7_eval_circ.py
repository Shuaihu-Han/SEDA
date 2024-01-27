#! /usr/bin/env python3
"""
dataset_4_explaination is for generating quantitative analysis file
for test saved model you can replace dataset_4_explaination with dataset
"""

import os
from dataclasses import dataclass

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataclass_parser import parse_into_dataclass
from circumst_event.narrative_cloze.dataset_4_explaination import MCNCDataset
from circumst_event.narrative_cloze.models import CircEventLightningModule
import json
import yaml
from circumst_event.narrative_cloze.dataset_4_explaination import MCNCDataModule
from pytorch_lightning.loggers import CSVLogger

def barray_add_1(array):
    length = len(array)
    extra = 1
    for i in range(length-1, -1, -1):
        if extra == 0:
            return array
        else:
            if array[i] == 0:
                array[i] = 1
                extra = 0
            else:
                array[i] = 0
                extra = 1
    return array

@dataclass
class EvaluationArguments:
    test_set_path: str = ''
    check_point_path: str = ''
    context_chain_length: int = 8
    seed: int = 1234

def main():
    args = EvaluationArguments()
    #users need to set their own model path 
    args.check_point_path = r"your.ckpt"

    hdir = os.path.join(
            os.path.dirname(os.path.dirname(args.check_point_path)),
            "hparams.yaml",
        )
    with open(hdir, 'r', encoding='utf-8') as f:
        hfile = yaml.load(f.read(), Loader=yaml.FullLoader)

    model = CircEventLightningModule.load_from_checkpoint(
        args.check_point_path,

        pretrained_path=hfile['pretrained_path'],
        freeze_embedding=hfile['freeze_embedding'],
        dropout=hfile['dropout'],
        hidden_size=hfile['hidden_size'],
        num_layers=hfile['num_layers'],
        num_heads=hfile['num_heads'],
        learning_rate=hfile['learning_rate'],
        weight_decay=hfile['weight_decay'],
    )

    logger = CSVLogger('logs', name='my_model')

    trainer = pl.Trainer(
        logger=False,
        gpus=[0],
        auto_select_gpus=True,
        callbacks=[]
    )

    indicator_len = hfile['context_events_length']
    indicator = [0] * indicator_len
    for i in range(2**indicator_len):
        pl.seed_everything(1234)
        dm = MCNCDataModule(
            dataset_path=hfile['dataset_path'],
            batch_size=hfile['batch_size'],
            num_choices=hfile['num_choices'],
            argument_length=hfile['argument_length'],
            context_events_length=hfile['context_events_length'],
            num_workers=1,
            pin_memory=False,
            chain_indicator=indicator
        )
        print(f'indicator:{indicator}')
        metrics = trainer.test(
            model=model,
            datamodule=dm,
        )
        barray_add_1(indicator)

if __name__ == "__main__":
    main()

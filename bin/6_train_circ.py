#! /usr/bin/env python3

from dataclasses import asdict, dataclass

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from circumst_event.narrative_cloze.models import CircEventLightningModule
from dataclass_parser import parse_into_dataclass
from circumst_event.narrative_cloze.dataset import MCNCDataModule

@dataclass
class _SemEventTrainingArguments:
    __program__ = __file__
    logdir: str = ''
    dataset_path: str = ''
    pretrained_path: str = ''
    num_choices: int = 5
    argument_length: int = 5
    context_events_length: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    freeze_embedding: bool = False
    hidden_size: int = 128
    num_layers: int = 4
    dropout: float = 0.0
    num_heads: int = 1
    segmentation_num: int = 3


def main():
    args = _SemEventTrainingArguments()
    args.logdir = 'log'
    #users need to set their own vocab path 
    args.pretrained_path = r'/root/autodl-tmp/emnlp/vocab/vocab.txt'
    #users need to set their own data path 
    args.dataset_path = r'/root/autodl-tmp/emnlp/reduced_split' 
    pl.seed_everything(1234)
    module = CircEventLightningModule(
        pretrained_path=args.pretrained_path,
        freeze_embedding=args.freeze_embedding,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for i in range(0, args.segmentation_num):
        trainer = pl.Trainer(
            logger=TensorBoardLogger(
                save_dir=args.logdir, name="model", default_hp_metric=False
            ),
            callbacks=[
                ModelCheckpoint(
                    filename="{epoch}-{acc_val:.4f}-{acc_test:.4f}",
                    monitor="acc_val",
                    save_top_k=3,
                    mode='max'
                ),
                EarlyStopping(monitor="acc_val", patience=5, mode="max",),
            ],
            auto_select_gpus=True,
            gpus=[0],
            terminate_on_nan=True,
            max_epochs=20,
        )
        trainer.logger.log_hyperparams(asdict(args))
        dm = MCNCDataModule(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            num_choices=args.num_choices,
            argument_length=args.argument_length,
            context_events_length=args.context_events_length,
            num_workers=4,
            data_segmentation_num=args.segmentation_num,
            data_current_segmentation=i,
            pin_memory=False
        )
        trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm)


if __name__ == "__main__":
    main()

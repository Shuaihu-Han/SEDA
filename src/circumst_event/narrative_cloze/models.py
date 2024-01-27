from typing import Any, List, NamedTuple, Optional

import numpy
import pytorch_lightning as pl
import torch
from torch import BoolTensor, LongTensor, Tensor, nn
from torch.nn import functional as f
from torch.optim import Adam
from torchmetrics import Accuracy, MetricCollection

from torch_utils import load_glove_into_dict, mh_attention_forward, rnn_forward
from math import sqrt
from circumst_event.narrative_cloze.event_trans import event_trans, choice_trans
from d2l import torch as d2l
import warnings
from pytorch_lightning.loggers import CSVLogger
warnings.filterwarnings("ignore")

class MCNarrativeCloze(NamedTuple):
    events_tensor: Tensor  # (bs, event_length, pred+args, length)
    events_mask_tensor: BoolTensor  # (bs, event_length)

    choices_tensor: Tensor  # (bs, num_choices, pred+args, length)
    label_tensor: LongTensor  # (bs,)


class MCNCWithSentence(NamedTuple):
    events_tensor: Tensor  # (bs, event_length, pred+args, length)
    events_mask_tensor: Tensor  # (bs, event_length)
    choices_tensor: Tensor  # (bs, num_choices, pred+args, length)
    label_tensor: Tensor  # (bs,)

    sentences_tensor: Tensor  # (bs, num_events, seq_len)
    sentences_ids_mask: Tensor  # (bs, num_events, seq_len)
    event_sent_mat: Tensor  # (bs, num_events)

class MCNCWithSentenceBatch(NamedTuple):
    events_tensor: Tensor  # (bs, event_length, pred+args, length)
    events_mask_tensor: Tensor  # (bs, event_length)
    choices_tensor: Tensor  # (bs, num_choices, pred+args, length)
    label_tensor: Tensor  # (bs,)

    sentences_tensor: Tensor  # (bs, num_events, seq_len)
    sentences_ids_mask: Tensor  # (bs, num_events, seq_len)
    event_sent_mat: Tensor  # (bs, num_events)

class MCNCOutput(NamedTuple):
    logits: Tensor
    event_sent_weights: Optional[Tensor] = None

class CircEventLightningModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_path: str,
        freeze_embedding: bool,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()
        # embedding
        glove_dict = load_glove_into_dict(pretrained_path)
        weights = torch.from_numpy(numpy.row_stack(list(glove_dict.values())))
        self._embedding = nn.Embedding.from_pretrained(
            embeddings=torch.as_tensor(weights, dtype=torch.float),
            freeze=freeze_embedding,
        )
        # event embedding
        self.embed_size = self._embedding.embedding_dim

        self.hidden_size = hidden_size

        self._transformer = choice_trans(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
            # dim_feedforward=hidden_size
        )

        self._event_emb_transformer = event_trans(
            d_model=hidden_size,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            nhead=num_heads,
            dim_feedforward=hidden_size
        )
        
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._train_metric = MetricCollection({"acc_train": Accuracy(top_k=1)})
        self._dev_metric = MetricCollection({"acc_val": Accuracy(top_k=1)})
        self._test_metric = MetricCollection({"acc_test": Accuracy(top_k=1)})

    def configure_optimizers(self):
        params_1x = [param for name, param in self.named_parameters() if name not in ["_embedding.weight"]]
        return Adam(
            [{'params': params_1x}, {'params': self._embedding.parameters(), 'lr': self._learning_rate / 10}],
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

    def forward(self, mcnc: MCNCWithSentenceBatch):
        batch_size, num_events, sequence_length = mcnc.sentences_tensor.shape

        events_tensor, events_tensor_mask = mcnc.events_tensor[0], mcnc.events_tensor[1]
        choice_tensor, choice_tensor_mask = mcnc.choices_tensor[0], mcnc.choices_tensor[1]

        choice_num = choice_tensor.shape[1]

        all_events_tensor = torch.cat([events_tensor, choice_tensor], dim=1)

        arguments_embedding = self._embedding(all_events_tensor)
        all_events_mask_tensor = torch.cat([events_tensor_mask, choice_tensor_mask], dim=1) 

        sq_tensor = self._embedding(mcnc.sentences_tensor)

        trans_src, choice_events_embedding = self._event_emb_transformer(
            tgt=[arguments_embedding, all_events_mask_tensor],
            src=sq_tensor,
            src_key_padding_mask=~torch.flatten(mcnc.sentences_ids_mask, 0, 1).bool()
        )

        logits = self._transformer(
            src=trans_src,
            tgt=choice_events_embedding
        )

        return MCNCOutput(
            logits=logits,
        )

    def compute_loss(
        self,
        output_dict: MCNCOutput,
        input_dict: MCNCWithSentenceBatch,
    ) -> Tensor:
        loss = f.cross_entropy(output_dict.logits, input_dict.label_tensor)
        return loss

    def training_step(self, batch: MCNCWithSentenceBatch, *args, **kwargs):
        output: MCNCOutput = self(batch)
        loss = self.compute_loss(output, batch)
        self._train_metric(torch.softmax(output.logits, dim=-1), batch.label_tensor)
        self.log("loss", loss.item())
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log_dict(self._train_metric.compute())

    def validation_step(
        self,
        batch: MCNCWithSentenceBatch,
        batch_idx: int,
        dataloader_idx: int,
        *args,
        **kwargs,
    ):
        output: MCNCOutput = self(batch)
        if dataloader_idx == 0:
            metric = self._dev_metric
        elif dataloader_idx == 1:
            metric = self._test_metric
        else:
            raise ValueError()

        batch_metrics = metric(torch.softmax(output.logits, dim=-1), batch.label_tensor)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self._dev_metric.compute())
        self.log_dict(self._test_metric.compute())

    def test_step(self, batch: MCNCWithSentenceBatch, batch_id, *args, **kwargs):
        output: MCNCOutput = self(batch)
        prob = torch.softmax(output.logits, dim=-1)

        # this code is to generate explantion data
        # label_prob = prob[torch.arange(len(batch.label_tensor)), batch.label_tensor]
        # with open('max_prob.txt', 'a') as f:
        #     numpy.savetxt(f, label_prob.cpu().numpy().reshape(1, -1), fmt='%.4f')

        self._test_metric(prob, batch.label_tensor)
        lengths = torch.sum(batch.events_mask_tensor, dim=-1)
        preds = torch.argmax(output.logits, dim=-1) == batch.label_tensor
        self.write_prediction(str(batch_id), [lengths, preds])

    def test_epoch_end(self, outputs) -> None:
        self.log_dict(self._test_metric.compute())
        self.trainer.evaluation_loop.predictions.to_disk()

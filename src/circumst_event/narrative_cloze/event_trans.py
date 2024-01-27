from torch import nn
import torch
from d2l import torch as d2l
from typing import Optional, Any, Union, Callable
import torch
from torch import Tensor
from torch.nn import functional as F
from math import sqrt

class choice_trans(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None):
        super(choice_trans, self).__init__()
        self.trans = nn.Transformer(
            d_model=d_model * 3,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype
        )
        self.output = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        
        trans_output = self.trans(
            src=src,
            tgt=torch.repeat_interleave(tgt, 3, dim=2),
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        logits = torch.squeeze(self.output(trans_output), dim=-1)

        return logits

class event_trans(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, embed_size=100, seq_len=60):
        super(event_trans, self).__init__()
        self.sa = event_self_attention(num_layers=num_encoder_layers)
        self.trans = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype
        )

        self.global_trans = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype
        )
        self.shape_input = nn.Sequential(
            nn.Linear(embed_size, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        self.sq_pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.d_model = d_model

    def forward(self, src: Tensor, tgt, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        
        arguments_embedding, all_events_mask_tensor = tgt[0], tgt[1]
        batch_size = arguments_embedding.shape[0]
        sq_tensor = src

        k_v_embedding = torch.flatten(arguments_embedding, 0, 2)
        k_v_embedding_mask = torch.flatten(all_events_mask_tensor, 0, 2)
        flattened_sq_tensor = torch.flatten(sq_tensor, 0, 1)
        
        context_events_embedding, choice_events_embedding = self.sa(k_v_embedding, k_v_embedding_mask)
        hidden_size_sq_tensor = self.shape_input(flattened_sq_tensor)
        hidden_size_sq_tensor = hidden_size_sq_tensor + self.sq_pos_encoding[:, :hidden_size_sq_tensor.shape[1], :]

        event_emb_tgt = torch.unsqueeze(torch.flatten(context_events_embedding, 0, 1), dim=1)
        flattened_decoder_embedding = self.trans(
            tgt=event_emb_tgt,
            src=hidden_size_sq_tensor,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        decoder_embedding = torch.squeeze(
            flattened_decoder_embedding, dim=1
        ).unflatten(0, (batch_size, -1))

        global_emb = self.global_trans(
            tgt=context_events_embedding,
            src=decoder_embedding
        )

        next_level_trans_src = torch.cat(
            [
                context_events_embedding,
                decoder_embedding,
                global_emb
            ],
            dim=-1,
        )

        return next_level_trans_src, choice_events_embedding

class event_self_attention(nn.Module):
    def __init__(self, embed_size=100, num_heads=1, hidden_size=128, num_layers=1, dropout=0, max_len=5, 
    num_events=8, choice_num=5,):
        super(event_self_attention, self).__init__()
        self.num_events = num_events
        self.choice_num = choice_num
        self._composition = nn.Sequential(
            nn.Linear(5 * 3 * embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self._event_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True, dropout=dropout)
        self._event_trans = nn.TransformerEncoder(self._event_layer, num_layers=num_layers)
        self._event_pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_size))

    def forward(self, k_v_embedding, k_v_embedding_mask):
        batch_size = k_v_embedding.shape[0] // ((self.num_events + self.choice_num) * 3)
        weighted_event = self._event_trans(src=k_v_embedding, src_key_padding_mask=~k_v_embedding_mask.bool())
        events_argument_embedding = weighted_event.unflatten(0, (batch_size, self.num_events + self.choice_num, 3))
        all_events_embedding = self._composition(
            torch.flatten(events_argument_embedding, 2, -1)
        )
        context_events_embedding, choice_events_embedding = torch.tensor_split(
            all_events_embedding, (-self.choice_num,), dim=1
        )
        return context_events_embedding, choice_events_embedding

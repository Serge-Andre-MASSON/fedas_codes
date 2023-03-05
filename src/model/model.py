from os.path import dirname
import torch
from yaml import load, Loader
from torch.nn import Module, Embedding, Transformer, Linear

from model.utils.utils import Mask, PositionalEncoding

model_conf_path = f"{dirname(__file__)}/model.yaml"


class Model(Module):
    def __init__(
            self,
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            masks,
            dim_feedforward,
            len_encoder_vocab,
            len_decoder_vocab,
            batch_first) -> None:
        super().__init__()

        self.d_model = d_model

        self.padding_mask = masks.padding
        self.subsequent_mask = masks.subsequent

        self.hints_embeddings = Embedding(
            num_embeddings=len_encoder_vocab,
            embedding_dim=d_model
        )

        self.codes_embeddings = Embedding(
            num_embeddings=len_decoder_vocab,
            embedding_dim=d_model)

        self.x_pe = PositionalEncoding(d_model)
        self.y_pe = PositionalEncoding(d_model)

        self.transformer = Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            batch_first=batch_first
        )

        self.generator = Linear(d_model, len_decoder_vocab)

    def forward(self, X, y):
        src_key_padding_mask = self.padding_mask(X)
        tgt_mask = self.subsequent_mask(y)

        X = self.hints_embeddings(X)
        X = self.x_pe(X)

        y = self.codes_embeddings(y)
        y = self.y_pe(y)

        transformed = self.transformer(
            X, y,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask
        )

        return self.generator(transformed)


def get_model(len_encoder_vocab, len_decoder_vocab, model_conf=model_conf_path):
    with open(model_conf) as f:
        yaml_conf = load(f, Loader=Loader)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    masks: Mask = Mask(device)

    model = Model(
        masks=masks,
        len_encoder_vocab=len_encoder_vocab,
        len_decoder_vocab=len_decoder_vocab,
        **yaml_conf
    ).to(device)

    return model

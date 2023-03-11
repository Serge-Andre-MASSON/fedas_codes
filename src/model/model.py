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
            len_decoder_vocab) -> None:
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
            batch_first=True
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

    def predict(self, X: torch.Tensor, y_in: torch.Tensor = None):
        self.eval()  # For results consistency
        if y_in is None:
            y_in = torch.zeros(X.size(0), 1)\
                .long()\
                .to("cuda")  # Assuming 0 is the <sos> token id

        y = None
        p = None

        for _ in range(4):
            probs, preds = self(X, y_in)[:, -1, :].softmax(-1).max(-1)
            preds = preds.view(-1, 1)
            probs = probs.view(-1, 1)

            if y is None:
                y = preds
                p = probs

            else:
                y = torch.cat([y, preds], axis=-1)
                p = torch.cat([p, probs], axis=-1)

            y_in = torch.cat([y_in, preds], axis=-1)

        return y, p


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

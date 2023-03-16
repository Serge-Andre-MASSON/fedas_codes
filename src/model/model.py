import torch
from torch.nn import Module, Embedding, Transformer, Linear

from model.utils.utils import Mask, PositionalEncoding
from conf.conf import get_conf


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

    def forward(self, X, y) -> torch.Tensor:
        padding_mask = self.padding_mask(X)
        subsequent_mask = self.subsequent_mask(y)

        X = self.hints_embeddings(X)

        y = self.codes_embeddings(y)
        y = self.y_pe(y)

        transformed = self.transformer(
            X, y,
            src_key_padding_mask=padding_mask,
            tgt_mask=subsequent_mask
        )

        return self.generator(transformed)

    @torch.no_grad()
    def predict(self, X: torch.Tensor):
        self.eval()  # For results consistency

        y_in = torch.zeros(X.size(0), 1)\
                    .long()\
                    .to("cuda")  # Assuming 0 is the <sos> token id

        y = None
        p = None

        for _ in range(4):
            probs, preds = self(X, y_in)[:, -1, :].softmax(-1).max(-1)
            preds = preds.view(-1, 1)
            probs = probs.view(-1, 1)

            y = preds if y is None else torch.cat([y, preds], axis=-1)
            p = probs if p is None else torch.cat([p, probs], axis=-1)

            y_in = torch.cat([y_in, preds], axis=-1)

        return p, y


def get_model(len_encoder_vocab: int, len_decoder_vocab: int):
    conf = get_conf("model")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        masks=Mask(device),
        len_encoder_vocab=len_encoder_vocab,
        len_decoder_vocab=len_decoder_vocab,
        **conf
    ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    return model

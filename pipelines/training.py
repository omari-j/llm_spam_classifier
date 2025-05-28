import logging
import os
from pathlib import Path
import numpy as np
from common import (
    PYTHON,
    FlowMixin,
    configure_logging,
    packages,
)
import pandas as pd
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    environment,
    project,
    pypi_base,
    resources,
    step,
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset

configure_logging()

TRAINING_EPOCHS = 5
TRAINING_BATCH_SIZE = 8
TRAINING_FRACTION = 0.7
VALIDATION_FRACTION = 0.1
NUMBER_OF_WORKERS = 0

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})


@project(name="penguins")
@pypi_base(
    python=PYTHON,
    packages=packages(
        "pandas",
        "numpy",
        "keras",
        "torch"
        "boto3",
        "packaging",
        "mlflow",
        "setuptools",
        "python-dotenv",
    ),
)

class Training(FlowSpec, FlowMixin):
    """Training pipeline.

    This pipeline trains, evaluates, and registers a model to classify a
    message as spam or not spam
    """
    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help=(
            "Minimum accuracy threshold required to register the model at the end of "
            "the pipeline. The model will not be registered if its accuracy is below "
            "this threshold."
        ),
        default=0.7,
    )
    @card
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            ),
        },
    )
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow
        import tiktoken

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        self.data = self.load_dataset()

        try:
            # Let's start a new MLFlow run to track everything that happens during the
            # execution of this flow. We want to set the name of the MLFlow
            # experiment to the Metaflow run identifier so we can easily
            # recognize which experiment corresponds with each run.
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        # This is the configuration we'll use to train the model. We want to set it up
        # at this point so we can reuse it later throughout the flow.
        self.training_parameters = {
            "epochs": TRAINING_EPOCHS,
            "batch_size": TRAINING_BATCH_SIZE,
        }
        self.tokenizer = tiktoken.get_encoding("gpt2")
        # Next we need to build the pytorch dataloaders we need to fine-tune the
        # models
        self.next(self.build_dataloaders)

    @card
    @step
    def build_dataloaders(self):
        """
        This step use the loaded datasets to construct the pytorch dataloaders
        used fine-tune the GPT-model for classification
        """
        from torch.utils.data import DataLoader

        train_df, validation_df, test_df = self._random_split(
            self.data,
            TRAINING_FRACTION,
            VALIDATION_FRACTION
        )

        train_dataset = SpamDataset(
            df=train_df,
            max_length=None,
            tokenizer=self.tokenizer
        )

        val_dataset = SpamDataset(
            df=validation_df,
            max_length=train_dataset.max_length,
            tokenizer=self.tokenizer
        )
        test_dataset = SpamDataset(
            df=test_df,
            max_length=train_dataset.max_length,
            tokenizer=self.tokenizer
        )

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=TRAINING_BATCH_SIZE,
            shuffle=True,
            num_workers=NUMBER_OF_WORKERS,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=TRAINING_BATCH_SIZE,
            num_workers=NUMBER_OF_WORKERS,
            drop_last=False,
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=TRAINING_BATCH_SIZE,
            num_workers=NUMBER_OF_WORKERS,
            drop_last=False,
        )

        self.next(self.build_model)

    @step
    @card
    def build_model(self):
        import torch
        import torch.nn as nn

        self.model = GPTModel(NEW_CONFIG)







    def _random_split(self, df, train_frac, validation_frac):
        # Shuffle the entire DataFrame
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)

        # Calculate split indices
        train_end = int(len(df) * train_frac)
        validation_end = train_end + int(len(df) * validation_frac)

        # Split the DataFrame
        train_df = df[:train_end]
        validation_df = df[train_end:validation_end]
        test_df = df[validation_end:]

        return train_df, validation_df, test_df



    def _assign(self, left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))

    def _load_weights_into_gpt(self, gpt, params):
        gpt.pos_emb.weight = self._assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = self._assign(gpt.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = self._assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = self._assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = self._assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = self._assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = self._assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = self._assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = self._assign(
                gpt.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = self._assign(
                gpt.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = self._assign(
                gpt.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = self._assign(
                gpt.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = self._assign(
                gpt.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = self._assign(
                gpt.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = self._assign(
                gpt.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = self._assign(
                gpt.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = self._assign(
                gpt.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = self._assign(
                gpt.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = self._assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = self._assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = self._assign(gpt.out_head.weight, params["wte"])







class SpamDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=None, pad_token_id=50256):
        self._data = df

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self._data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self._data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self._data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

if __name__ == "__main__":
    Training()
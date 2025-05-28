import logging
import os
from pathlib import Path
import numpy as np
from common import (
    PYTHON,
    FlowMixin,
    configure_logging,
    packages,
    download_and_load_gpt2,
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
from inference import Model
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
        "boto3",
        "packaging",
        "mlflow",
        "setuptools",
        "python-dotenv",
        "torch",
        "tiktoken",
        "tensorflow",
        "tqdm"
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
        import tempfile

        model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")

        self.model = GPTModel(NEW_CONFIG)
        self.model.eval()
        self.device = torch.device("cpu")



        with tempfile.TemporaryDirectory() as directory:
            logging.info(f"Created temporary directory for GPT-2 weights: {directory}")

            logging.info(f"Downloading GPT-2 {model_size} to temporary directory...")

            settings, params = download_and_load_gpt2(model_size, directory)

            logging.info("Loading weights into model...")

            self._load_weights_into_gpt(self.model, params)
            self.model = self.model.to(self.device)
            logging.info(msg=f"Model loaded successfully and moved to {self.device}")
            logging.info(msg="Temporary files will be cleaned up when exiting this block")

        for param in self.model.parameters():
            param.requires_grad = False

        num_classes = 2

        # new output head has its requires_grad attribute set to True by default
        # which means that it will be updated during training
        self.model.out_head = torch.nn.Linear(
            in_features=NEW_CONFIG["emb_dim"],
            out_features=num_classes
        )

        # setting the
        for param in self.model.trf_blocks[-1].parameters():
            param.requires_grad = True

        for param in self.model.final_norm.parameters():
            param.requires_grad = True

        self.next()

    @step
    @card
    def train_model(self):
        import mlflow
        import time



        torch.manual_seed(123)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.1)

        self.training_parameters["optimizer"] = self.optimizer

        start_time = time.time()

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Let's disable the automatic logging of models during training so we
            # can log the model manually during the registration step.
            mlflow.autolog(log_models=False)

            self.train_losses, self.val_losses, self.train_accs, self.val_accs, self.examples_seen = self._train_classifier_simple(
                self.model,
                self.train_loader,
                self.val_loader,
                self.optimizer,
                self.device,
                num_epochs=TRAINING_EPOCHS,
                eval_freq=50,
                eval_iter=5,
            )
            mlflow.log_params(self.training_parameters)

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        self.model.eval()
        logging.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

        self.next(self.evaluate)

    @step
    @card
    def evaluate(self):
        """Evaluate the model we trained in the train models step
        """
        import mlflow

        logging.info("Evaluating the model")

        # Let's evaluate the model using the test data we processed and stored as
        # artifacts during the `transform` step.
        self.train_accuracy = self._calc_accuracy_loader(
            self.train_loader, self.model, self.device
        )
        self.val_accuracy = self._calc_accuracy_loader(
            self.val_loader, self.model, self.device
        )
        self.test_accuracy = self._calc_accuracy_loader(
            self.test_loader, self.model, self.device
        )

        logging.info(
            "Training accuracy: %f - Validation accuracy: %f - Test accuracy: %f",
            self.train_accuracy,
            self.val_accuracy,
            self.test_accuracy
        )

        # Let's log everything under the same nested run we created when training the
        # current fold's model.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(
                {
                    "test_accuracy": self.test_accuracy,
                    "validation_accuracy": self.val_accuracy,
                    "train_accuracy": self.train_accuracy,
                    "execution_time": self.execution_time,
                    "train_losses": self.train_losses,
                    "validation_losses": self.val_losses

                },
            )

        # When we finish evaluating every fold in the cross-validation process, we want
        # to evaluate the overall performance of the model by averaging the scores from
        # each fold.
        self.next(self.register_model)

    @step
    def register_model(self):
        """Register the model in the Model Registry.

        This function will prepare and register the final model in the Model Registry

        We'll only register the model if its accuracy is above a predefined threshold.
        """
        import tempfile

        import mlflow

        if self.test_accuracy >= self.accuracy_threshold:
            logging.info("Registering model...")

            # We'll register the model under the experiment we started at the beginning
            # of the flow. We also need to create a temporary directory to store the
            # model artifacts.
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                # We can now register the model using the name "penguins" in the Model
                # Registry. This will automatically create a new version of the model.
                mlflow.pyfunc.log_model(
                    python_model=Model(data_capture=False),
                    registered_model_name="spam",
                    artifact_path="model",
                    code_path=[(Path(__file__).parent / "inference.py").as_posix()],
                    artifacts=self._get_model_artifacts(directory),
                    pip_requirements=self._get_model_pip_requirements(),
                    signature=self._get_model_signature(),
                    # Our model expects a Python dictionary, so we want to save the
                    # input example directly as it is by setting`example_no_conversion`
                    # to `True`.
                )
        else:
            logging.info(
                "The accuracy of the model (%.2f) is lower than the accuracy threshold "
                "(%.2f). Skipping model registration.",
                self.test_accuracy,
                self.accuracy_threshold,
            )

            # Let's now move to the final step of the pipeline.
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")

    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.

        """
        # Let's save the model inside the supplied directory.

        model_path = (Path(directory) / "model.pth").as_posix()
        torch.save(self.model, model_path)

        return {
            "model": model_path
        }

    def _get_model_signature(self):
        """Return the model's signature.

        The signature defines the expected format for model inputs and outputs. This
        definition serves as a uniform interface for appropriate and accurate use of
        a model.
        """
        from mlflow.models import infer_signature

        return infer_signature(
            model_input={
                "text": "This is a sample email for demonstration purposes."
            },
            model_output={"classification": "not spam"},
            params={"data_capture": False},
        )

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        return [
            f"{package}=={version}"
            for package, version in packages(
                "scikit-learn",
                "tiktoken",
                "torch",
                "pandas",
                "numpy",
            ).items()
        ]

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


    def _calc_accuracy_loader(self, data_loader, model, device, num_batches=None):
        model.eval()
        correct_predictions, num_examples = 0, 0

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                with torch.no_grad():
                    logits = model(input_batch)[:, -1, :]  # Logits of last output token
                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
        return correct_predictions / num_examples

    def _calc_loss_batch(self, input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)[:, -1, :]  # Logits of last output token
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss

    def _calc_loss_loader(self, data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self._calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def _evaluate_model(self, model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = self._calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = self._calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss

    def _train_classifier_simple(self, model, train_loader, val_loader, optimizer, device, num_epochs,
                                eval_freq, eval_iter):
        # Initialize lists to track losses and examples seen
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        examples_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                loss = self._calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()  # Calculate loss gradients
                optimizer.step()  # Update model weights using loss gradients
                examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self._evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Calculate accuracy after each epoch
            train_accuracy = self._calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
            val_accuracy = self._calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
            print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
            print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)

        return train_losses, val_losses, train_accs, val_accs, examples_seen






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
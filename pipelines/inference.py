import logging
import logging.config
import os
import sqlite3
import uuid
from common import GPTModel
from datetime import datetime, timezone
from typing import Any, Optional, Dict
import tiktoken
import joblib
import mlflow
import torch
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModelContext
import torch.nn as nn


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



class Model(mlflow.pyfunc.PythonModel):
    """A custom model that can be used to make predictions.

    This model implements an inference pipeline with three phases: preprocessing,
    prediction, and postprocessing. The model will optionally store the input requests
    and predictions in a SQLite database.
    """

    def __init__(
        self,
        data_collection_uri: Optional[str] = "spam.db",
        *,
        data_capture: bool = False,
        encoding_name: str = "gpt2",
    ) -> None:
        """Initialize the model.

        By default, the model will not collect the input requests and predictions. This
        behavior can be overwritten on individual requests.

        This constructor expects the connection URI to the storage medium where the data
        will be collected. By default, the data will be stored in a SQLite database
        named "spam.db" and located in the root directory from where the model runs.
        You can override the location by using the 'DATA_COLLECTION_URI' environment
        variable.
        """
        self.data_capture = data_capture
        self.data_collection_uri = data_collection_uri
        self.encoding_name = encoding_name

    def load_context(self, context: PythonModelContext) -> None:
        """Laod the fine-tuned weights into the model architecture.

        This function is called only once as soon as the model is constructed.
        """

        self._configure_logging()
        logging.info("Loading model context...")

        self.device = "cpu"
        # assign model architecture class attribute
        self.model = GPTModel(NEW_CONFIG)
        # define the binary classification output head
        out_head = torch.nn.Linear(
            in_features=NEW_CONFIG["emb_dim"],
            out_features=2,
        )
        # replace the existing output head with the new head
        self.model.out_head = out_head
        self.model.eval()

        self.model.to(self.device)
        # assign the tokenizer to a class attribute
        self.tokenizer = tiktoken.get_encoding(self.encoding_name)
        # If the DATA_COLLECTION_URI environment variable is set, we should use it
        # to specify the database filename. Otherwise, we'll use the default filename
        # specified when the model was instantiated.
        self.data_collection_uri = os.environ.get(
            "DATA_COLLECTION_URI",
            self.data_collection_uri,
        )


        logging.info("Data collection URI: %s", self.data_collection_uri)

        logging.info(f"Loading model weights from {context.artifacts['model']}")
        try:
            # Try loading as a dictionary first (recommended format)
            checkpoint = torch.load(
                context.artifacts["model"],
                map_location=self.device
            )

            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # If saved as a dictionary with state_dict key
                self.model.load_state_dict(checkpoint['state_dict'])
                logging.info("Loaded model weights from checkpoint dictionary")
            else:
                # If saved as a direct state_dict (your current approach)
                self.model.load_state_dict(checkpoint)
                logging.info("Loaded model weights from direct state_dict")

        except Exception as e:
            logging.error(f"Error loading model weights: {str(e)}")
            raise

        # Keep model in eval mode after loading
        self.model.eval()

        # Log some information about the model for debugging

        logging.info("Model is ready to receive requests")

    def predict(
        self,
        context: Optional[Any],
        model_input: pd.DataFrame, # Explicitly expect a DataFrame based on MLflow behavior
        params: Optional[Dict[str, Any]] = None,
    ) -> list: # Return type is a list (likely of dictionaries)
        """Handle the request received from the client.
        This method is responsible for processing the input data received from the
        client, making a prediction using the model, and returning a readable response
        to the client.
        """
        logging.info(
            "Received prediction request with %d %s",
            len(model_input),
            "samples" if len(model_input) > 1 else "sample",
        )

        # verify that model_input DataFrame has a "text" column
        if "text" not in model_input.columns:
            logging.error("Input DataFrame is missing the 'text' column.")
            # Return a list of error dicts, matching the expected output structure count
            return [{"error": "Input DataFrame missing 'text' column", "classification": None}] * len(model_input)

        texts_to_classify = model_input["text"].tolist()
        results = []

        for text_value in texts_to_classify:
            result = self.classify(
                text=text_value,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                # max_length can be passed from params if needed, e.g., params.get("max_length")
                # Otherwise, classify will use its internal default (supported_context_length)
            )
            results.append(result) # result is "spam" or "not spam"

        model_output = self.process_output(results) # process_output should format this list of strings

        if (
            params
            and params.get("data_capture", False) is True
            or not params # if params is None, this becomes True
            and self.data_capture # Then this is evaluated
        ):
            self.capture(model_input, model_output) # Capture the input DataFrame and the final output

        logging.info("Returning prediction to the client")
        logging.debug("%s", model_output)

        return model_output


    def process_output(self, output: [str]) -> list:
        """Process the prediction received from the model.

        This method is responsible for transforming the prediction received from the
        model into a readable format that will be returned to the client.
        """
        logging.info("Processing prediction received from the model...")

        result = []
        if output is not None:
            classifications = output

            # Return the prediction from the model output as JSON.
            result = [
                {"classification": classification} for classification in classifications
            ]

        return result


    def classify(self, text, model, tokenizer, device, max_length=1024, pad_token_id=50256):

        model.eval()

        # Prepare inputs to the model
        input_ids = tokenizer.encode(text)
        supported_context_length = model.pos_emb.weight.shape[0]
        # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
        # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

        # Truncate sequences if they too long
        input_ids = input_ids[:min(max_length, supported_context_length)]

        # Pad sequences to the longest sequence
        input_ids += [pad_token_id] * (max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # add batch dimension

        # Model inference
        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
        predicted_label = torch.argmax(logits, dim=-1).item()

        # Return the classified result
        return "spam" if predicted_label == 1 else "not spam"


    def capture(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Save the input request and output prediction to the database.

        This method will save the input request and output prediction to a SQLite
        database. If the database doesn't exist, this function will create it.
        """
        logging.info("Storing input payload and predictions in the database...")

        connection = None
        try:
            connection = sqlite3.connect(self.data_collection_uri)

            # Let's create a copy from the model input so we can modify the DataFrame
            # before storing it in the database.
            data = model_input.copy()

            # We need to add the current time and the classification
            # to the DataFrame to store everything together.
            data["date"] = datetime.now(timezone.utc)

            # Initialize the classification with None. We'll
            # overwrite it later if the model output is not empty.
            data["classification"] = None

            # add a column to store the ground truth. This column can be
            # used by the labeling team to provide the actual species for the data.
            data["ground_truth"] = None

            # If the model output is not empty, update the classification
            # columns with the corresponding values.
            if model_output is not None and len(model_output) > 0:
                data["classification"] = [item["classification"] for item in model_output]

            # Let's automatically generate a unique identified for each row in the
            # DataFrame. This will be helpful later when labeling the data.
            data["uuid"] = [str(uuid.uuid4()) for _ in range(len(data))]

            # Finally, we can save the data to the database.
            data.to_sql("data", connection, if_exists="append", index=False)

        except sqlite3.Error:
            logging.exception(
                "There was an error saving the input request and output prediction "
                "in the database.",
            )
        finally:
            if connection:
                connection.close()

    def _configure_logging(self):
        """Configure how the logging system will behave."""
        import sys
        from pathlib import Path

        if Path("logging.conf").exists():
            logging.config.fileConfig("logging.conf")
        else:
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                level=logging.INFO,
            )

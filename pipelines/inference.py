import logging
import logging.config
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Dict
import tiktoken
import joblib
import mlflow
import torch
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModelContext


class Model(mlflow.pyfunc.PythonModel):
    """A custom model that can be used to make predictions.

    This model implements an inference pipeline with three phases: preprocessing,
    prediction, and postprocessing. The model will optionally store the input requests
    and predictions in a SQLite database.

    The [Custom MLflow Models with mlflow.pyfunc](https://mlflow.org/blog/custom-pyfunc)
    blog post is a great reference to understand how to use custom Python models in
    MLflow.
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
        named "penguins" and located in the root directory from where the model runs.
        You can override the location by using the 'DATA_COLLECTION_URI' environment
        variable.
        """
        self.data_capture = data_capture
        self.data_collection_uri = data_collection_uri
        self.encoding_name = encoding_name

    def load_context(self, context: PythonModelContext) -> None:
        """Load the transformers and the scikit-learn model specified as artifacts.

        This function is called only once as soon as the model is constructed.
        """
        # By default, we want to use the JAX backend for Keras. You can use a different
        # backend by setting the `KERAS_BACKEND` environment variable.

        self._configure_logging()


        self.device = "cpu"

        logging.info("Loading model context...")

        self.model = torch.load(context.artifacts["model"], weights_only=False)

        logging.info("Model context successfully loaded")

        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = tiktoken.get_encoding(self.encoding_name)
        # If the DATA_COLLECTION_URI environment variable is set, we should use it
        # to specify the database filename. Otherwise, we'll use the default filename
        # specified when the model was instantiated.
        self.data_collection_uri = os.environ.get(
            "DATA_COLLECTION_URI",
            self.data_collection_uri,
        )


        logging.info("Data collection URI: %s", self.data_collection_uri)

        logging.info("Model is ready to receive requests")

    def predict(
        self,
        context: Optional[Any], # PythonModelContext, but Optional[Any] to avoid import error in snippet
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
            len(model_input), # len() on a DataFrame gives number of rows
            "samples" if len(model_input) > 1 else "sample",
        )

        # model_input is expected to be a DataFrame with a "text" column
        # based on the signature and the curl command structure.
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
            predictions = output

            # Let's transform the prediction index back to the
            # original species. We can use the target transformer
            # to access the list of classes.

            # We can now return the prediction and the confidence from the model.
            # Notice that we need to unwrap the numpy values so we can serialize the
            # output as JSON.
            result = [
                {"classification": prediction} for prediction in predictions
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

            # We need to add the current time and the prediction
            # to the DataFrame to store everything together.
            data["date"] = datetime.now(timezone.utc)

            # Let's initialize the prediction and confidence columns with None. We'll
            # overwrite them later if the model output is not empty.
            data["classification"] = None

            # Let's also add a column to store the ground truth. This column can be
            # used by the labeling team to provide the actual species for the data.
            data["ground_truth"] = None

            # If the model output is not empty, we should update the prediction
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

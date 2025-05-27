import logging
import logging.config
import os
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
from metaflow import S3, IncludeFile, current

PYTHON = "3.9"

PACKAGES = {
    "pandas": "2.2.3",
    "torch": "2.1.0",
    "numpy": "2.1.1",
    "keras": "3.6.0",
    "boto3": "1.35.32",
    "packaging": "24.1",
    "mlflow": "2.17.1",
    "setuptools": "75.1.0",
    "requests": "2.32.3",
    "evidently": "0.4.33",
    "python-dotenv": "1.0.1",
}



class FlowMixin:
    """Base class used to share code across multiple pipelines."""

    dataset = IncludeFile(
        "spam",
        is_text=True,
        help=(
            "Local copy of the spam dataset. This file will be included in the "
            "flow and will be used whenever the flow is executed in development mode."
        ),
        default="data/spam.tsv",
    )

    def load_dataset(self):
        """Load and prepare the dataset.

        When running in production mode, this function reads every CSV file available in
        the supplied S3 location and concatenates them into a single dataframe. When
        running in development mode, this function reads the dataset from the supplied
        string parameter.
        """
        import numpy as np

        if current.is_production:
            dataset = os.environ.get("DATASET", self.dataset)

            with S3(s3root=dataset) as s3:
                files = s3.get_all()

                logging.info("Found %d file(s) in remote location", len(files))

                raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
                data = pd.concat(raw_data)
        else:
            # When running in development mode, the raw data is passed as a string,
            # so we can convert it to a DataFrame.
            data = pd.read_csv(StringIO(self.dataset), sep="\t", header=None, names=["Label", "Text"])

        data = create_balanced_dataset(data)
        data = data["Label"].map({"ham": 0, "spam": 1})
        seed = int(time.time() * 1000) if current.is_production else 42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        logging.info("Loaded dataset with %d samples", len(data))

        return data


def create_balanced_dataset(df):

    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def packages(*names: str):
    """Return a dictionary of the specified packages and their version.

    This function is useful to set up the different pipelines while keeping the
    package versions consistent and centralized in a single location.
    """
    return {name: PACKAGES[name] for name in names if name in PACKAGES}


def configure_logging():
    """Configure logging handlers and return a logger instance."""
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )






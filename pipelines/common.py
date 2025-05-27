import logging
import logging.config
import os
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
from metaflow import S3, IncludeFile, current


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






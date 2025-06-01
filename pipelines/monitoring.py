import logging
import sqlite3

from metaflow import (
    FlowSpec,
    Parameter,
    card,
    project,
    pypi_base,
    step,
)

from common import PYTHON, FlowMixin, configure_logging, packages

configure_logging()

@project(name="salary")
@pypi_base(
    python=PYTHON,
    packages=packages("evidently","pandas","boto3"),
)
class Monitoring (FlowSpec, FlowMixin):
    """A monitoring pipeline to monitor the performance of the hosted model

    This pipeline runs a series of test and generates several reports using the
    data captured by the hosted model and a reference dataset
    """

    datastore_uri = Parameter(
        "datastore-uri",
        help=(
            "The location where the production data is stored. The pipeline supports"
            "loading the data from the SQLite database or from an S3 location that"
            "follows SagemMaker's format for capturing the data."
        ),
        required=True,
    )

    ground_truth_uri = Parameter(
        "ground-truth-uri",
        help=(
            "The S3 location where the ground truth labels associated with the"
            "endpoint's collected data is stored. The content of this S3 location must"
            "follow SageMaker's format for storing ground truth data."
        )
    )

    assume_role = Parameter(
        "assume-role",
        help=(
            "The role the pipeline will assume to access the production data in S3"
            "This parameter is required when the pipeline is running under a set of"
            "credentials that don't have access to the S3 location where the"
            "production data is stored."
        ),
        required=False,
    )

    samples = Parameter(
        "samples",
        help=(
            "The maximum number of samples that will be loaded from the production"
            "datastore to the monitoring tests and reports. The flow will load"
            ""
        ),
        default=200,
    )

    @card
    @step
    def start(self):
        """Start the monitoring pipeline"""
        from evidently import DataDefinition, Dataset, BinaryClassification

        self.reference_data = self.load_dataset()
        # When running some of the tests and reports, we need to have a prediction
        # column in the reference data to match the production dataset
        self.reference_data["classification"] = self.reference_data["ground_truth"]
        self.current_data = self._load_production_datastore()

        # Some of the test and reports require labelled data,s o we need to filter out
        # the samples that don't have ground truth labels.
        logging.info(f"Current Data Columns: {self.current_data.columns}")
        self.current_data_labeled = self.current_data[
            self.current_data["ground_truth"].notna()
        ]


        logging.info(f"Current Data Labeled Columns: {self.current_data_labeled.columns}")

        self.definition = DataDefinition(
            classification=[BinaryClassification(
                target="ground_truth",
                prediction_labels="classification"
            )],
            text_columns=["text"],
            categorical_columns=["ground_truth", "classification"],

        )

        self.reference_data = Dataset.from_pandas(self.reference_data, data_definition=self.definition)

        self.current_data = Dataset.from_pandas(
            self.current_data,
            data_definition=self.definition
        )

        self.next(self.test_suite)

    @card(type="html")
    @step
    def test_suite(self):
        """Run a test suite of pre-built tests.

        This test suite will run a group of pre-built tests to perform structured data
        and model checks.
        """
        from evidently import Report
        from evidently.metrics import (
        RowCount,
        ColumnCount,
        EmptyRowsCount,
        EmptyColumnsCount,
        DuplicatedColumnsCount,
        DuplicatedRowCount,
        DatasetMissingValueCount,
        InListValueCount
        )

        report = Report([
            RowCount(),
            EmptyRowsCount(),
            DuplicatedRowCount(),
            ColumnCount(),
            EmptyColumnsCount(),
            DuplicatedColumnsCount(),
            DatasetMissingValueCount(),
            InListValueCount(
                column=["ground_truth", "classification"],
                values=["spam", "ham"]
            )
        ])

        test_report = report.run(
            reference_data=self.reference_data,
            current_data=self.current_data)

        self.html = self._get_evidently_html(test_report)


        self.next(self.data_quality_report)

    @card(type="html")
    @step
    def data_quality_report(self):
        """Generate a report about the quality of the data and any data drift.

        This report will provide detailed feature statistics, feature behavior
        overview of the data, and an evaluation of data drift with respect to the
        reference data. It will perform a side-by-side comparison between the
        reference and the production data.
        """
        from evidently.presets import DataDriftPreset
        from evidently import Report



        report = Report(
            metrics=[
                # We want to report dataset drift as long as one of the columns has
                # drifted. We can accomplish this by specifying that the share of
                # drifting columns in the production dataset must stay under 10% (one
                # column drifting out of 8 columns represents 12.5%).
                DataDriftPreset(threshold=0.1),
            ],
        )

        # We don't want to compute data drift in the ground truth column, so we need to
        # remove it from the reference and production datasets.
        logging.info(f"Reference Data Columns: {self.reference_data.columns}")
        logging.info(f"Current Data Columns: {self.current_data.columns}")

        feature_drift_report = report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )

        self.html = self._get_evidently_html(feature_drift_report)
        self.next(self.target_drift_report)

    @card(type="html")
    @step
    def test_accuracy_score(self):
        """Generate a Target Drift report.

        This report will explore any changes in model predictions with respect to the
        reference data. This will help us understand if the distribution of model
        predictions is different from the distribution of the target in the reference
        dataset.
        """
        from evidently.metrics import Accuracy
        from evidently import Report

        report = Report(
            metrics=[
                Accuracy(gte=0.9),
            ],
        )

        if not self.current_data_labeled.empty:
            accuracy_report = report.run(
                reference_data=self.reference_data,
                current_data=self.current_data_labeled,
                # We only want to compute drift for the prediction column, so we need to
                # specify a column mapping without the target column.
            )

            self.html = self._get_evidently_html(accuracy_report)
        else:
            self._message("No labeled production data.")

        self.next(self.classification_report)

    @card(type="html")
    @step
    def classification_report(self):
        """Generate a Classification report.

        This report will evaluate the quality of a classification model.
        """
        from evidently.presets import ClassificationPreset
        from evidently import Report


        report = Report(
            metrics=[ClassificationPreset()],
        )

        if not self.current_data.empty:
            classification_report = report.run(
                # The reference data is using the same target column as the prediction, so
                # we don't want to compute the metrics for the reference data to compare
                # them with the production data.
                reference_data=self.reference_data,
                current_data=self.current_data,
            )
            try:
                self.html = self._get_evidently_html(classification_report)
            except Exception:
                logging.exception("Error generating report.")
        else:
            self._message("No labeled production data.")

        self.next(self.end)

    @step
    def end(self):
        """Finish the monitoring flow."""
        logging.info("Finishing monitoring flow.")

    def _load_production_datastore(self):
        """Load the production data from the specified datastore location."""
        data = None
        if self.datastore_uri.startswith("s3://"):
            data = self._load_production_data_from_s3()
        else:
            data = self._load_production_data_from_sqlite()

        logging.info("Loaded %d samples from the production dataset.", len(data))

        return data

    def _load_production_data_from_sqlite(self):
        """Load the production data from a SQLite database."""
        import pandas as pd

        connection = sqlite3.connect(self.datastore_uri)

        query = (
            "SELECT text, classification, ground_truth FROM data ORDER BY date DESC LIMIT ?;"
        )

        # Notice that we are using the `samples` parameter to limit the number of
        # samples we are loading from the database.
        data = pd.read_sql_query(query, connection, params=(self.samples,))

        connection.close()

        return data

    def _get_evidently_html(self, evidently_object) -> str:
        """Returns the rendered EvidentlyAI report/metric as HTML

        Should be assigned to `self.html`, installing `metaflow-card-html` to be rendered
        """
        import tempfile

        with tempfile.NamedTemporaryFile() as tmp:
            evidently_object.save_html(tmp.name)
            with open(tmp.name) as fh:
                return fh.read()

    def _message(self, message):
        """Display a message in the HTML card associated to a step."""
        self.html = message
        logging.info(message)


if __name__ == "__main__":
    Monitoring()


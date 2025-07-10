# LLM-Based Spam Classification Pipeline with Orchestration, Monitoring and Deployment
This project is an end-to-end machine learning pipeline that:

- Fine-tunes a GPT-2 large large language model to classify spam messages.
- Tracks, versions and deploys the models with MLflow
- Monitors the pipeline with Evidently
- Uses Metaflow to orchestrate the entire pipeline

The full code for this project can be reviewed in the github repo

## Getting started

This project uses uv to manage dependencies and and environments. To run the project you will need to download and install uv if it isn’t on your system.

You can follow the instructions to download and install uv [here](https://docs.astral.sh/uv/getting-started/installation/)

You can read more about uv

https://docs.astral.sh/uv/guides/

Once you have cloned the repo, you need  to navigate to the project directory:

```bash
cd llm_spam_classifier
```

create the virtual environment and install dependencies:

```bash
uv sync
```

Next, activate the virtual environment:

```bash
source .venv/bin/activate
```

## **Running MLflow**

[MLflow](https://mlflow.org/) is a platform-agnostic machine learning lifecycle management tool that will help you track experiments and share and deploy models.

To run an MLflow server locally, open a terminal window, activate the virtual environment you created earlier, and run the following command:

```
mlflow server --host 127.0.0.1 --port 5001
```

Once running, you can navigate to [`http://127.0.0.1:500](http://127.0.0.1:5000/)1` in your web browser to open MLflow's user interface.

By default, MLflow tracks experiments and stores data in files inside a local `./mlruns` directory. You can change the location of the tracking directory or use a SQLite database using the parameter `--backend-store-uri`. The following example uses a SQLite database to store the tracking data:

```
mlflow server --host 127.0.0.1 --port 5001 \
   --backend-store-uri sqlite:///mlflow.db
```

For more information, check some of the [common ways to set up MLflow](https://mlflow.org/docs/latest/tracking.html#common-setups). You can also run the following command to get more information about the server:

```
mlflow server --help
```

After the server is running, modify the `.env` file inside the repository's root directory to add the `MLFLOW_TRACKING_URI` environment variable pointing to the tracking URI of the MLflow server. The following command will append the variable to the file and export it in your current shell:

```
export $((echo "MLFLOW_TRACKING_URI=http://127.0.0.1:5001" >> .env; cat .env) | xargs)
```

## **Visualizing Pipeline Results**

We can observe the execution of each pipeline and visualize their results live using [Metaflow Cards](https://docs.metaflow.org/metaflow/visualizing-results). Metaflow will set up a local server for viewing these cards as the pipeline runs.

To open the built-in card server for the Training pipeline, navigate to your repository's root directory in a new terminal window and run this command:

```
uv run pipelines/training.py --environment=pypi card server
```

Open your browser and navigate to [localhost:8324](http://localhost:8324/). Every time you run the Training pipeline, the viewer will automatically update to show the cards related to the latest pipeline execution.

Check [Using Local Card Viewer](https://docs.metaflow.org/metaflow/visualizing-results/effortless-task-inspection-with-default-cards#using-local-card-viewer) for more information about the local card viewer.

## Running the training pipeline

To run the script that finetunes the GPT model and sets the accuracy_threshold use the following command:

```bash
uv run pipelines/training.py --environment=pypi run \
--accuracy-threshold 0.7
```

Where 0.7 is the accuracy threshold of your choice. Models that don’t meet the threshold will not be logged to the MLflow model registry.

## How MLflow deployments work

You must package a model as an [MLflow Model](https://www.notion.so/Introduction-to-MLflow-1859c5b1995280f29a80d02b77c1cb07?pvs=21) to deploy it. MLflow Models are directories that contain all the metadata needed to use the model for prediction. MLflow can containerise models (when deployed in production) or create a virtual environment (for local deployment)

Below is the command used to deploy the model locally

```bash
uv run mlflow models serve \
    -m models:/spam/$( \
        curl -s -X GET "$MLFLOW_TRACKING_URI""/api/2.0/"\
"mlflow/registered-models/"\
"get-latest-versions" \
            -H "Content-Type: application/json" \
            -d '{"name": "spam"}' | \
        jq -r '.model_versions[0].version'\
    ) -h 0.0.0.0 -p 8080 --no-conda
```

## Generating predictions from the local inference server

After the server is up and running, it is now time to generate predictions:

```bash
curl -X POST http://0.0.0.0:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [{
            "text": "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
        }]}'
```

Above is an example, the user can enter the message of their choice.

## Running the synthetic traffic pipeline locally

To simulate production conditions, you can run the traffic pipeline with an example command:

```bash
uv run pipelines/traffic.py --environment=pypi run \
    --action traffic \
    --target local \
    --drift-proportion 0.5 \
    --target-uri http://127.0.0.1:8080/invocations \
    --samples 20 \
    --drift-type vocabulary
    
```

Command line options can be used to set class variables within the flows. The options available are:

### action

```bash
--action
```

The action you want to perform. The supported actions are 'traffic' for sending traffic to the endpoint and 'labeling' for labeling the data captured by the endpoint.

### target

```bash
--target
```

The target platform hosting the model. The value for running a local inference service is 'local'.

### target uri

```bash
--target-uri
```

"The location where the pipeline will send the fake traffic or generate ground truth labels. If generating traffic, this parameter will point to the hosted model. If generating labels, this parameter will point to the location of the data captured by the model.

### samples

```bash
--samples
```

The number of samples that will be sent to the hosted model. Samples will be sent in batches of 10, so you might end up with a few more samples than the value you set for this parameter.

### drift proportion

```bash
--drift-proportion
```

A float between 0 and 1 representing the proportion of the traffic that will synthetically generated.The rest of the data will be from the original dataset.

### drift type

```bash
--drift-type
```

The type of drift to introduce in the text samples. Options:'none': No drift, 'vocabulary': Introduce new vocabulary/slang terms.

### ground truth quality

```bash
--ground-truth-quality
```

"This parameter represents how similar the ground truth labels will be to the predictions generated by the model. Setting this parameter to a value less than 1.0 will introduce noise in the labels to simulate inaccurate model predictions.

Once we run the traffic pipeline, we verify the database has stored the data.

```bash
sqlite3 spam.db "SELECT COUNT(*) FROM data;"
```

After generating some traffic, you can run the pipeline to generate fake ground truth labels for the data captured by the model. The `--target-uri` parameter points to the SQLite database, where the model captured the input data and predictions:

```bash
uv run pipelines/traffic.py --environment=pypi run \
    --action traffic \
    --target local \
    --target-uri spam.db \
    -- ground-truth-quality 0.8

```

### Traffic

```bash
python3 pipelines/traffic_labeller.py --environment=pypi run \
    --action labeling \
    --target local \
    --drift-proportion 0 \
    --target-uri http://127.0.0.1:8080/invocations \
    --samples 20 \
    --drift-type none
    
```

## Monitoring pipeline

To monitor the performance of our deployed model, the monitoring pipeline executes tests and produces insightful graphics. It compiles a comprehensive report by leveraging three key data sources: the original dataset, synthetic data generated by the `traffic.py` module's traffic action, and the labels derived from the same module's labeling action.

To view the graphics and report results you can set up Metaflow's built-in viewer for the Monitoring pipeline with the following command:

```bash
uv run pipelines/monitoring.py --environment=pypi card server
```

The navigate in your browser to [localhost:8324](http://localhost:8324/) .

Finally, run the Monitoring pipeline using the command below. The `--datastore-uri` parameter should point to the SQLite database where the model stores the input data and classifications.

```bash
uv run pipelines/monitoring.py --environment=pypi run \
    --datastore-uri spam.db
```

You will see every report generated by the pipeline in the built-in viewer opened in your browser.

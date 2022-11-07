import os
import time
from pathlib import Path
from pprint import pprint
from typing import Optional

import gcsfs
import typer
from google.cloud import aiplatform, storage

BUCKET_NAME = os.getenv("BUCKET")
PROJECT = os.getenv("PROJECT")
CONTAINER_URI = f"gcr.io/{PROJECT}/nlp-sample:latest"
REGION = "us-east4"


def list_of_dataset_version(dataset_name: str):
    fs = gcsfs.GCSFileSystem(project=PROJECT)
    dataset_path = f"{BUCKET_NAME}/{dataset_name}/"
    pprint(fs.ls(dataset_path))


def upload_dataset(path_to_data: Path, dataset_name: str):
    fs = gcsfs.GCSFileSystem(project=PROJECT)
    dataset_path = f"{BUCKET_NAME}/{dataset_name}/{time.time()}"
    print(f"Upload data to {dataset_path}")
    fs.put(str(path_to_data), dataset_path, recursive=True)


def download_dataset(dataset_name: str, path_to_data: Path, version: Optional[str] = None):
    fs = gcsfs.GCSFileSystem(project=PROJECT)
    if version is None:
        dataset_path = f"{BUCKET_NAME}/{dataset_name}/"
        versions = fs.ls(dataset_path)
        last_version = sorted([float(x.split("/")[-1]) for x in versions])[-1]
        print(f"using latest version {last_version}")
        version = last_version
    dataset_path = f"{BUCKET_NAME}/{dataset_name}/{version}"
    fs.get(dataset_path, str(path_to_data), recursive=True)


# Based on https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/community-content/tf_keras_image_classification_distributed_multi_worker_with_vertex_sdk
def run_traininig_job(display_name: str):

    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET_NAME)

    command = f"python nlp_sample/train.py --model_name_or_path prajjwal1/bert-tiny --train_file /gcs/{BUCKET_NAME}/nlp-dataset/1667623178.0768118/train.csv --validation_file /gcs/{BUCKET_NAME}/nlp-dataset/1667623178.0768118/val.csv --output_dir /gcs/{BUCKET_NAME}/models/{display_name}/result".split()
    job = aiplatform.CustomContainerTrainingJob(display_name=display_name, container_uri=CONTAINER_URI, command=command)

    model = job.run(replica_count=1, machine_type="n1-standard-4", base_output_dir="gs://skjdfhglsdkjfhglskdjhglskdjfg")


def list_traininig_job():
    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET_NAME)
    custom_jobs = aiplatform.CustomJob.list()
    pprint([x.to_dict()["displayName"] for x in custom_jobs])


def create_cli():
    app = typer.Typer()
    app.command()(upload_dataset)
    app.command()(list_of_dataset_version)
    app.command()(download_dataset)

    app.command()(run_traininig_job)
    app.command()(list_traininig_job)
    return app


if __name__ == "__main__":
    app = create_cli()
    app()

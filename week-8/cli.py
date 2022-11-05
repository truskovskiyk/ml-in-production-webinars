import typer
from google.cloud import storage
import gcsfs


BUCKET_NAME = "65e19660-bb7a-4c31-8838-0a6bc072e9cd"

def list_buckets():
    storage_client = storage.Client()
    buckets = storage_client.list_buckets()
    print(buckets)
    for bucket in buckets:
        print(bucket.name)

def list_of_datasets():
    fs = gcsfs.GCSFileSystem(project='my-google-project')
    fs.ls(BUCKET_NAME)
    print("TEST")
    list_buckets()

def upload_dataset():
    pass

def download_dataset():
    pass

def run_notebook():
    pass

def run_traininig_job():
    pass

def create_pipeline():
    pass

def create_endpoint():
    pass


def create_cli():
    app = typer.Typer()
    app.command()(list_of_datasets)
    app.command()(upload_dataset)
    return app

if __name__ == "__main__":
    app = create_cli()
    app()

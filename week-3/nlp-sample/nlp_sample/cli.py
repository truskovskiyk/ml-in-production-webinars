import typer

from nlp_sample.data import load_cola_data
from nlp_sample.train import train
from nlp_sample.utils import load_from_registry, upload_to_registry

app = typer.Typer()
app.command()(train)
app.command()(load_cola_data)
app.command()(upload_to_registry)
app.command()(load_from_registry)

if __name__ == "__main__":
    app()

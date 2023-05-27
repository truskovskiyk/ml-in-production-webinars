import argparse

import kserve

from .image_transformer import ImageTransformer

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="The name that the model is served under.")
parser.add_argument("--predictor_host", help="The URL for the model predict function", required=True)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    server = kserve.ModelServer()
    server.start(models=[transformer])

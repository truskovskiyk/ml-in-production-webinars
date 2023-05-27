import kserve
from typing import Dict
from serving.predictor import Predictor
import argparse

class CustomModel(kserve.Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.predictor = None
       self.load()
       
    def load(self):
        self.predictor = Predictor.default_from_model_registry()

    def predict(self, request: Dict) -> Dict:
        return {"instances": [request]}
    

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME, help='The name that the model is served under.')
parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    custom_model = CustomModel(args.model_name, predictor_host=args.predictor_host)
    server = kserve.ModelServer()
    server.start(models=[custom_model])

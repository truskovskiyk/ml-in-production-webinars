import logging
from typing import Dict, Union

import boto3
from cloudevents.http import CloudEvent

import kserve
from kserve import InferRequest, InferResponse
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse
import kserve
import argparse

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

session = boto3.Session()
client = session.client('s3', endpoint_url='http://minio-service:9000', aws_access_key_id='minio', aws_secret_access_key='minio123')
digits_bucket = 'digits'



class ImageTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self._key = None

    async def preprocess(self, inputs: Union[Dict, CloudEvent, InferRequest], headers: Dict[str, str] = None) -> Union[Dict, InferRequest]:
        import pickle
        pickle.dump(inputs, open('inputs.pkl', 'wb'))

        logging.info("Received inputs %s", inputs)
        logging.info("TEST" * 10)
        # logging.info(inputs)
        # logging.info(type(inputs))
        # logging.info(type(inputs['attributes']))
        # logging.info(inputs['attributes'])
        # logging.info(inputs['attributes']['data'])
        # logging.info(type(inputs['attributes']['data']))
        logging.info("TEST" * 10)
        import json

        data = json.loads(inputs.get_data().decode('utf-8'))
        inputs = data
        if inputs['EventName'] == 's3:ObjectCreated:Put':
            bucket = inputs['Records'][0]['s3']['bucket']['name']
            key = inputs['Records'][0]['s3']['object']['key']
            self._key = key
            client.download_file(bucket, key, '/tmp/' + key)
            request = image_transform('/tmp/' + key)
            return {"instances": [request]}
        raise Exception("unknown event")

    def postprocess(self, response: Union[Dict, InferResponse, ModelInferResponse], headers: Dict[str, str] = None) \
            -> Union[Dict, ModelInferResponse]:
        logging.info("response: %s", response)
        index = response["predictions"][0]["classes"]
        logging.info("digit:" + str(index))
        upload_path = f'digit-{index}/{self._key}'
        client.upload_file('/tmp/' + self._key, digits_bucket, upload_path)
        logging.info(f"Image {self._key} successfully uploaded to {upload_path}")
        return response


DEFAULT_MODEL_NAME = "custom-model"

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME, help='The name that the model is served under.')
parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    server = kserve.ModelServer()
    server.start(models=[transformer])

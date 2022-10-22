import numpy as np
from seldon_core.seldon_client import SeldonClient
import time 
import asyncio
import requests

URL = "54.221.129.217:7777"
PRED_URL = "http://54.221.129.217:7777/seldon/default/nlp-sample/api/v1.0/predictions"
FEEDBACK_URL = "http://54.221.129.217:7777/seldon/default/nlp-sample/api/v1.0/feedback"


def send_request_sync():
    data = {"data": {"ndarray": ["this is an example"]}}
    res = requests.post(PRED_URL, json=data)
    print(res)


def send_feedback_sync():
    data = {"request": {"data": {"ndarray": ["this is an example"]}}, "truth":{"data": {"ndarray": [1]}}}
    res = requests.post(FEEDBACK_URL, json=data)
    print(res)


if __name__ == '__main__':
    # send_request_sync()
    send_feedback_sync()
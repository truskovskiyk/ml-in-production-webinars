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
    # sc = SeldonClient(namespace=NS, gateway_endpoint=URL, deployment_name=DEPLOYMENT, payload_type="ndarray")
    # y_res = sc.predict(data=np.array(["this is an example this is an example this is an example this is an example"] * 32))
    # print(f"Raw Response: {y_res.response}")

# async def main():
#     requests = [send_request(n_requests=10) for _ in range(10)]
#     await asyncio.gather(*requests)

# def run_example():
#     s = time.perf_counter()
#     asyncio.run(main())
#     elapsed = time.perf_counter() - s
#     print(f"{__file__} executed in {elapsed:0.2f} seconds.")

# def run_sync():
#     s = time.perf_counter()
#     while True:
#         send_request_sync(n_requests=10)
#         # [send_request_sync(n_requests=10) for _ in range(10)]
#     elapsed = time.perf_counter() - s
#     print(f"{__file__} executed in {elapsed:0.2f} seconds.")

if __name__ == '__main__':
    # send_request_sync()
    send_feedback_sync()
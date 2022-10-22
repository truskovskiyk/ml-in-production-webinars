import numpy as np
from seldon_core.seldon_client import SeldonClient
import time 
import asyncio

URL = "54.221.129.217:7777"
NS = "default"
DEPLOYMENT = "nlp-sample"

async def send_request_async(n_requests: int):
    sc = SeldonClient(namespace=NS, gateway_endpoint=URL, deployment_name=DEPLOYMENT, payload_type="ndarray")
    for _ in range(n_requests):
        y_res = sc.predict(data=np.array(["this is an example"]))
        print(f"Raw Response: {y_res.response}\n")
        # print(f"Predicted Class: {y_res.response['data']['ndarray']}")


def send_request_sync(n_requests: int):
    sc = SeldonClient(namespace=NS, gateway_endpoint=URL, deployment_name=DEPLOYMENT, payload_type="ndarray")
    for _ in range(n_requests):
        y_res = sc.predict(data=np.array(["this is an example this is an example this is an example this is an example"] * 1024))
        print(f"Raw Response: {y_res.response}\n")
        # print(f"Predicted Class: {y_res.response['data']['ndarray']}")

async def main():
    requests = [send_request(n_requests=10) for _ in range(10)]
    await asyncio.gather(*requests)

def run_example():
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

def run_sync():
    s = time.perf_counter()
    while True:
        send_request_sync(n_requests=10)
        # [send_request_sync(n_requests=10) for _ in range(10)]
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

if __name__ == '__main__':
    # run_example()
    run_sync()
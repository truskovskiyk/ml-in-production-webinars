import json
import random
from typing import List

from locust import HttpUser, between, task
from pydantic import BaseSettings
from tqdm import tqdm


class UPCUser(HttpUser):
    @task(weight=1)
    def random_request(self):
        headers = {
            "Content-type": "application/json",
        }
        data = {"data": {"ndarray": ["this is an example this is an example this is an example"] * 32}}
        self.client.post(
            url="/seldon/default/nlp-auto-scaling-sample/api/v1.0/predictions",
            json=data,
            auth=None,
            headers=headers,
        )

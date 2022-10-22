import numpy as np
from seldon_core.seldon_client import SeldonClient

if __name__ == '__main__':
    sc = SeldonClient(namespace="default", gateway_endpoint="54.221.129.217:7777", deployment_name="nlp-sample", payload_type="ndarray")

    for _ in range(100):
        y_res = sc.predict(data=np.array(["this is an example"]))
        print(f"Raw Response: {y_res.response}\n")
        print(f"Predicted Class: {y_res.response['data']['ndarray']}")
        # print(f"Expected class: {y_test[101]}")
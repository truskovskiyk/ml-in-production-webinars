import logging
from typing import List, Optional
import time
from serving.predictor import Predictor

logger = logging.getLogger()


class SeldonAPI:
    def __init__(self, model_id: Optional[str] = None):
        self.predictor = Predictor.default_from_model_registry(model_id=model_id)

    def predict(self, text, features_names: List[str]):
        logger.info(text)

        s = time.perf_counter()
        results = self.predictor.predict(text)
        elapsed = time.perf_counter() - s
        self._run_time = elapsed

        logger.info(results)
        return results


    def metrics(self):
        return [
            {"type": "GAUGE", "key": "gauge_runtime", "value": self._run_time}
        ]

    def tags(self):
        return {"version": self.predictor.model_id}
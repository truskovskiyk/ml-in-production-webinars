import logging
from typing import List, Optional
import time
from serving.predictor import Predictor
from sklearn.metrics import f1_score

logger = logging.getLogger()

class SeldonAPI:
    def __init__(self, model_id: Optional[str] = None):
        self.predictor = Predictor.default_from_model_registry(model_id=model_id)

        self._run_f1 = None
        self._run_time = None

    def predict(self, text, features_names: List[str]):
        logger.info(text)

        s = time.perf_counter()
        results = self.predictor.predict(text)
        elapsed = time.perf_counter() - s
        self._run_time = elapsed

        logger.info(results)
        return results


    def tags(self):
        return {"version": self.predictor.model_id}


    def send_feedback(self, features, feature_names, reward, truth, routing=""):
        logger.info("features")
        logger.info(features)
        
        logger.info("truth")
        logger.info(truth)

        results = self.predict(features)
        preds = np.argmax(results, axis=1)

        f1 = f1_score(y_true=truth, y_pred=preds),
        self._run_f1 = f1
        return [{'f1': f1}]  

    def metrics(self):
        if self._run_f1 is None:
            return [{"type": "GAUGE", "key": "gauge_runtime", "value": self._run_time}]
        else:
            return [
                {"type": "GAUGE", "key": "gauge_runtime", "value": self._run_time},
                {"type": "GAUGE", "key": f"gauge_f1", "value": self._run_f1},
            ]        
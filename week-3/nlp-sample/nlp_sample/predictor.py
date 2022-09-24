import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import logging

logger = logging.getLogger()


class Predictor:
    def __init__(self, model_load_path: str = "results/"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_load_path)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        logger.info(text)
        logger.info(list(text))
        text_encoded = self.tokenizer.batch_encode_plus(list(text), return_tensors="pt", padding=True)
        bert_outputs = self.model(**text_encoded).logits
        logger.info(bert_outputs)
        return softmax(bert_outputs).numpy()

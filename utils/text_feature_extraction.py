from transformers import BertTokenizer, BertModel
import torch


class BertFeatureExtractor:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def extract_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the last hidden state (features) for all tokens
        return outputs.last_hidden_state

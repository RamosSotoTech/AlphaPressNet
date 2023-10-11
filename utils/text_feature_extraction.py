import spacy
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel


class BertFeatureExtractor:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def extract_features(self, text):
        # if not isinstance(text, str):
        #     raise ValueError("Text must be a string.")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the last hidden state (features) for all tokens
        return outputs.last_hidden_state

    # Function to get top n important words
    def get_top_n_important_words(self, text, n):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the last hidden state (features) for all tokens
        embeddings = outputs.last_hidden_state[0]

        # Calculate the magnitude of embeddings
        magnitudes = torch.norm(embeddings, dim=1)

        # Get the indices of the top n embeddings
        top_n_indices = torch.topk(magnitudes, n).indices

        # Get the top n important words
        top_n_tokens = [self.tokenizer.convert_ids_to_tokens(int(token_id.item())) for token_id in top_n_indices]

        return top_n_tokens


class EntityRelationAnalysis:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")

    def get_entities(self, text):
        """Extract named entities using spaCy's NER."""
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, padding=True)
        outputs = self.model(**inputs)
        token_embeddings = outputs["last_hidden_state"][0]
        return token_embeddings.mean(dim=0)

    def find_related_entities(self, baseline_text, target_paragraph, threshold=0.5):
        baseline_embedding = self.get_embedding(baseline_text)
        entities = self.get_entities(target_paragraph)

        similarities = {}
        for entity in entities:
            entity_embedding = self.get_embedding(entity)
            similarity = cosine_similarity(baseline_embedding.unsqueeze(0), entity_embedding.unsqueeze(0)).item()
            if similarity > threshold:
                similarities[entity] = similarity

        sorted_entities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_entities

    def find_related_words(self, entity, description, threshold=0.5):
        entity_embedding = self.get_embedding(entity).detach()  # Detach tensor before converting
        # Tokenize the description without special tokens to get words only
        inputs = self.tokenizer(description, return_tensors="pt", add_special_tokens=False, truncation=True,
                                padding=True)
        outputs = self.model(**inputs)
        word_embeddings = outputs["last_hidden_state"][0].detach()  # Detach tensor before converting

        # Compute similarities between entity and each word in description
        similarities = cosine_similarity(entity_embedding.unsqueeze(0),
                                         word_embeddings).squeeze()  # Squeeze instead of indexing

        # Pairing each word with its similarity score
        words = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_similarities = dict(zip(words, similarities.cpu().numpy()))  # Convert similarities to numpy array

        # Filter words based on the threshold
        filtered_words = {word: score for word, score in word_similarities.items() if score >= threshold}

        # Sorting words based on similarity to entity
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        return sorted_words

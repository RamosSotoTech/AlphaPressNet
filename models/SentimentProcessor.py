import re

import torch
from transformers import pipeline
import spacy
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel


# Subject Sentiment Analysis Processor
class SuSen:
    def __init__(self):
        # Initialize the sentiment analysis pipeline
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def subject_sentiment(self, text, subject):
        """
        Analyzes sentiment towards a specific subject in a given text.

        Args:
            text (str): The input text to analyze.
            subject (str): The subject for which sentiment is analyzed.

        Returns:
            dict: A dictionary containing sentiment analysis results, including 'label' and 'score'.
                  If no relevant chunks containing the subject are found, returns None.
        """
        # Split the text by sentences or meaningful chunks (this can be improved)
        chunks = re.split(r'\.|!|\?|\n', text)

        # Filter the chunks to retain only those containing the subject
        relevant_chunks = [chunk for chunk in chunks if subject in chunk]

        if not relevant_chunks:
            return None  # or a neutral default, based on the use case

        # Evaluate sentiment for the relevant chunks
        sentiments = self.sentiment_pipeline(relevant_chunks)

        # Aggregate or pick the sentiment (this can be averaged, voted, etc.)
        # For simplicity, just return the sentiment of the first chunk
        return sentiments[0]


class BertSimilarity:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_word_embedding(self, sentence, word):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)

        # Tokenize the word to understand how it's broken down
        word_tokens = self.tokenizer.tokenize(word)
        word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)

        # Find the positions of all tokens of the word
        indices = [i for i, id in enumerate(inputs["input_ids"][0].tolist()) if id in word_ids]

        if not indices:
            raise ValueError(f"Word '{word}' not found in sentence.")

        # Average the embeddings if the word is broken into multiple tokens
        embeddings = torch.mean(outputs['last_hidden_state'][0, indices, :], dim=0)

        return embeddings

    def compute_similarity(self, base_sentence, base_word, compare_sentence):
        base_embedding = self.get_word_embedding(base_sentence, base_word)
        compare_embedding = self.get_word_embedding(compare_sentence, base_word)

        similarity = cosine_similarity(base_embedding.unsqueeze(0), compare_embedding.unsqueeze(0)).item()
        return similarity

    def find_related_words(self, base_word, candidate_words, top_n=10):
        # Compute the embedding for the base word (using a neutral sentence for context)
        base_embedding = self.get_word_embedding(f"The context of {base_word} is being analyzed.", base_word)

        similarities = {}

        for word in candidate_words:
            # Compute the embedding for each candidate word
            word_embedding = self.get_word_embedding(f"The context of {word} is being analyzed.", word)
            similarity = cosine_similarity(base_embedding.unsqueeze(0), word_embedding.unsqueeze(0)).item()
            similarities[word] = similarity

        # Sort by similarity and return the top_n words
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]


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

    def find_related_entities(self, baseline_text, target_paragraph):
        baseline_embedding = self.get_embedding(baseline_text)
        entities = self.get_entities(target_paragraph)

        similarities = {}
        for entity in entities:
            entity_embedding = self.get_embedding(entity)
            similarity = cosine_similarity(baseline_embedding.unsqueeze(0), entity_embedding.unsqueeze(0)).item()
            similarities[entity] = similarity

        sorted_entities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_entities


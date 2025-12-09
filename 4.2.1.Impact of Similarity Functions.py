# Jaccard
def similarity(self, set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

# Word2Vec
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

class Word2VecSimilarity:
    def __init__(self, sentences=None, model_path=None):

        if model_path:
            self.model = Word2Vec.load(model_path)
        else:
            tokenized_sentences = [simple_preprocess(sent) for sent in sentences]
            self.model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

    def sentence_vector(self, sentence):
        tokens = simple_preprocess(sentence)
        word_vectors = []
        for word in tokens:
            try:
                word_vectors.append(self.model.wv[word])
            except KeyError:
                continue
        if len(word_vectors) == 0:
            return np.zeros(self.model.vector_size)
        return np.mean(word_vectors, axis=0)

    def similarity(self, text1: str, text2: str) -> float:
        vec1 = self.sentence_vector(text1)
        vec2 = self.sentence_vector(text2)
        dist = cosine_distances([vec1], [vec2])[0][0]
        return 1 - dist

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

class TFIDFSimilarity:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def similarity(self, text1: str, text2: str) -> float:
        texts = [text1, text2]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        dist = cosine_distances(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return 1 - dist
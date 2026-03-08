import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:

    def __init__(self):
        self.vectors = []
        self.documents = []

    def add(self, text, vector):
        self.documents.append(text)
        self.vectors.append(vector)

    def search(self, query_vector):

        sims = cosine_similarity([query_vector], self.vectors)[0]

        best_index = np.argmax(sims)

        return self.documents[best_index], sims[best_index]
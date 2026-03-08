import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.8):

        self.entries = []
        self.hit_count = 0
        self.miss_count = 0
        self.threshold = threshold

    def search(self, vector):

        if len(self.entries) == 0:
            self.miss_count += 1
            return False, None, 0

        vectors = [e["vector"] for e in self.entries]

        sims = cosine_similarity([vector], vectors)[0]

        idx = np.argmax(sims)

        if sims[idx] > self.threshold:

            self.hit_count += 1

            return True, self.entries[idx], float(sims[idx])

        self.miss_count += 1

        return False, None, float(sims[idx])

    def add(self, query, vector, result, cluster):

        self.entries.append({
            "query": query,
            "vector": vector,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = self.hit_count + self.miss_count

        return {
            "total_entries": len(self.entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total else 0
        }

    def clear(self):

        self.entries = []
        self.hit_count = 0
        self.miss_count = 0
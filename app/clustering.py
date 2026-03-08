from sklearn.cluster import KMeans

class ClusterModel:

    def __init__(self):
        self.model = KMeans(n_clusters=3)

    def train(self, vectors):
        if len(vectors) >= 3:
            self.model.fit(vectors)

    def get_cluster(self, vector):
        try:
            return int(self.model.predict([vector])[0])
        except:
            return 0
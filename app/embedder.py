from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
corpus = []

def embed(text):

    global corpus

    corpus.append(text)

    if len(corpus) == 1:
        vectors = vectorizer.fit_transform(corpus).toarray()
        return vectors[0], vectors

    vectors = vectorizer.fit_transform(corpus).toarray()

    return vectors[-1], vectors
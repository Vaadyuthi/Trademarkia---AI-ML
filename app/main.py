from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from app.embedder import embed
from app.cache import SemanticCache
from app.clustering import ClusterModel

app = FastAPI(
    title="Trademarkia Semantic Search API",
    version="1.0.0"
)

cache = SemanticCache()
cluster = ClusterModel()


class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage():
    return """
    <html>
    <head>
        <title>Trademarkia Semantic Search</title>

        <style>

            body{
                font-family: Arial;
                background: linear-gradient(135deg,#0f172a,#1e293b);
                color:white;
                text-align:center;
                padding:80px;
            }

            h1{
                font-size:48px;
                color:#38bdf8;
            }

            p{
                font-size:20px;
                color:#cbd5f5;
                margin-top:20px;
            }

            .btn{
                display:inline-block;
                margin-top:40px;
                padding:15px 35px;
                background:#38bdf8;
                color:black;
                text-decoration:none;
                border-radius:12px;
                font-weight:bold;
                font-size:18px;
            }

            .card{
                margin-top:60px;
                background:#1e293b;
                padding:30px;
                border-radius:15px;
                width:500px;
                margin-left:auto;
                margin-right:auto;
                box-shadow:0px 10px 30px rgba(0,0,0,0.5);
            }

        </style>

    </head>

    <body>

        <h1>Trademarkia Semantic Search</h1>

        <p>Semantic AI Search Engine</p>

        <div class="card">

            <h2>Features</h2>

            <p>Semantic Search</p>
            <p>Query Cache Optimization</p>
            <p>Query Clustering</p>
            <p>Cache Statistics Monitoring</p>

        </div>

        <a class="btn" href="/docs">Open API Documentation</a>

    </body>
    </html>
    """

@app.get("/status", tags=["System"])
def status():
    return {"message": "Trademarkia Semantic Search Running"}

@app.post("/query", tags=["Search"])
def query_api(request: QueryRequest):

    q = request.query

    vector, vectors = embed(q)

    cluster.train(vectors)

    hit, entry, sim = cache.search(vector)

    if hit:
        return {
            "query": q,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),
            "result": entry["result"],
            "dominant_cluster": entry["cluster"]
        }

    result = "Result for: " + q

    cluster_id = cluster.get_cluster(vector)

    cache.add(q, vector, result, cluster_id)

    return {
        "query": q,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0.0,
        "result": result,
        "dominant_cluster": cluster_id
    }

@app.get("/cache/stats", tags=["Cache"])
def cache_stats():
    return cache.stats()

@app.delete("/cache", tags=["Cache"])
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}
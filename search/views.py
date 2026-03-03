import json
from pathlib import Path
from django.shortcuts import render
from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "circles.json", "r", encoding="utf-8") as f:
    circles = json.load(f)

with open(BASE_DIR / "circle_embeddings.json", "r") as f:
    circle_embs = json.load(f)

model = SentenceTransformer("intfloat/multilingual-e5-large")

def search_view(request):
    query = request.GET.get("q", "")
    results = []

    if query:
        query_emb = model.encode("query: " + query)
        scores = util.cos_sim(query_emb, circle_embs)[0].tolist()

        for score, circle in zip(scores, circles):
            tags = circle.get("tags", [])

            if query in tags:
                score += 1.0

            music_tags = ["音楽", "歌う", "ハーモニー", "合唱", "コーラス"]
            if any(tag in tags for tag in music_tags):
                score += 0.3

            results.append((score, circle))

        results = sorted(results, key=lambda x: x[0], reverse=True)[:5]

    return render(request, "search.html", {"results": results, "query": query})
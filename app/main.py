from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
import json
import numpy as np
import logging
from datetime import datetime

app = FastAPI()

logging.basicConfig(filename="search_logs.txt", level=logging.INFO)

@app.on_event("startup")
def startup_event():
    global model, collection, ids

    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("sample_permits_normalized.json", "r") as f:
        records = json.load(f)

    embedding_texts = []
    metadatas = []
    ids = []

    for i, record in enumerate(records):
        parts = [
            record["type"]["description"] or "",
            record["type"]["class"] or "",
            record["type"]["work_class"] or "",
            record["details"]["description"] or "",
            record["location"]["address"] or "",
            record["location"]["city"] or "",
            record["status"]["current"] or ""
        ]
        text = " | ".join(p.strip() for p in parts if p)
        embedding_texts.append(text)
        ids.append(str(i))
        metadatas.append({
            "permit_id": record["permit_id"] or "",
            "permit_type_desc": record["type"]["description"] or "",
            "permit_class": record["type"]["class"] or "",
            "work_class": record["type"]["work_class"] or "",
            "status": record["status"]["current"] or "",
            "city": record["location"]["city"] or "",
            "zip": record["location"]["zip"] or "",
            "valuation": float(record["details"]["valuation"]) if record["details"]["valuation"] else 0.0,
            "housing_units": int(record["details"]["housing_units"]) if record["details"]["housing_units"] else 0,
            "floors": int(record["details"]["floors"]) if record["details"]["floors"] else 0,
            "project_id": record["project"]["id"] or ""
        })

    embedding_vectors = np.load("permit_embeddings.npy").tolist()

    client = chromadb.Client()
    collection = client.create_collection("permits")
    collection.add(
        documents=embedding_texts,
        embeddings=embedding_vectors,
        metadatas=metadatas,
        ids=ids
    )

@app.get("/healthz")
def health():
    return {"ok": True}

class SearchRequest(BaseModel):
    query: str = Field(..., example="New mechanical permit for Drive", description="Your search query")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"permit_type_desc": "Mechanical Permit", "city": "Austin"},
        description="Optional filters for metadata fields like city, permit_type_desc, etc."
    )

@app.post("/search")
def search(request: SearchRequest):
    try:
        query_embedding = model.encode([request.query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(5, len(ids)),
            where=request.filters or {},
            include=["documents", "metadatas", "distances"]
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": results["distances"][0][i]
            })

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": request.query,
            "filters": request.filters,
            "top_result_ids": [results["metadatas"][0][i].get("permit_id", "") for i in range(min(2, len(results["ids"][0])))]
        }
        logging.info(json.dumps(log_entry))

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

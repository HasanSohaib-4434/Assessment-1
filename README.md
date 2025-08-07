### ✅ `README.md`

# Permit Search API (Austin, TX)

This project provides a semantic search API over construction permit data from the City of Austin. It leverages Sentence Transformers for text embeddings and ChromaDB for efficient similarity-based retrieval.

---

## 1. Data Source

Permit data was fetched from the official open dataset:

```

[https://data.austintexas.gov/resource/3syk-w9eu.json?\$limit=8](https://data.austintexas.gov/resource/3syk-w9eu.json?$limit=8)

```

The raw JSON file was saved as `sample_permits_raw.json`.

---

## 2. Data Normalization

Raw fields were normalized into a structured format with nested dictionaries, such as:

- `permit_id`
- `type`: description, class, work_class
- `status`: applied, issued, expires
- `location`: address, city, state, zip, lat/lon
- `contractor` and `details`
- `project` info

Missing fields were handled gracefully using `.get()` and default fallbacks like `""`.

Date strings were trimmed to remove time portion (e.g., `2023-05-17T00:00:00` → `2023-05-17`).

The final normalized data was written to `sample_permits_normalized.json`.

## 3. Embedding Generation

Text fields from each normalized record were concatenated with `" | "` separator to form embedding texts.  
Example parts used:

- type.description
- type.class
- type.work_class
- details.description
- location.address
- location.city
- status.current

These texts were embedded using `all-MiniLM-L6-v2` from `sentence-transformers`, and saved as:
permit_embeddings.npy

## 4. ChromaDB Collection

At app startup:

- The embedding vectors and metadata are loaded.
- A ChromaDB collection named `"permits"` is created.
- Documents, embeddings, and metadata are added using `.add()`.

## 5. API Endpoints

### `GET /healthz`

Returns `{"ok": true}` to confirm that the server is running.

### `POST /search`

Accepts a query string and optional filters to return the top 5 most relevant permits.

#### Request Body:

json
{
"query": "new mechanical Drive",
"filters": {
"permit_type_desc": "Mechanical Permit"
}
}

#### Response Example:

json
[
{
"text": "Mechanical Permit | Commercial | Change Out | Replacement... | AUSTIN | Active",
"metadata": {
"permit_id": "2025-094349 MP",
"permit_type_desc": "Mechanical Permit",
"permit_class": "Commercial",
...
},
"similarity": 1.48
},
...
]

### Features:

- Top-5 most similar permits based on query embeddings
- Optional metadata filtering (e.g., city, permit type)
- Similarity scores in response
- Query logging to `search_logs.txt` with timestamp, filters, and top permit IDs

## 6. How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Start the API

uvicorn app.main:app --reload

### 3. Access OpenAPI docs

Visit:
http://localhost:8000/docs

to explore the endpoints interactively.

## 8. Requirements

- `fastapi`
- `uvicorn`
- `sentence-transformers`
- `chromadb`
- `numpy`
- `pydantic`

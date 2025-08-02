# RAG-based Question Answering Webhook

This project implements a webhook that uses a Retrieval-Augmented Generation (RAG) architecture to answer questions based on a provided document.

## Features

*   **Document Processing:** Ingests documents from a URL and processes them into searchable chunks.
*   **Semantic Search:** Uses sentence embeddings to perform semantic searches on document chunks, retrieving the most relevant information.
*   **AI-Powered Question Answering:** Utilizes a Large Language Model (LLM) to generate answers based on the retrieved context from documents.
*   **Scalable Backend:** Built with FastAPI for a robust and performant API.
*   **Vector Database Integration:** Integrates with Qdrant for efficient vector storage and retrieval.
*   **Authentication:** Secure API access using bearer tokens.

## Technologies Used

*   **Backend:**
    *   Python 3.10+
    *   FastAPI
    *   Qdrant (Vector Database)
    *   Sentence Transformers (for embeddings)
    *   OpenRouter (for LLM integration)
    *   Pydantic (for data validation)
*   **Frontend:**
    *   HTML (basic `index.html` for demonstration)

## Prerequisites

Before you begin, ensure you have the following installed:

*   [Python 3.10+](https://www.python.org/downloads/)
*   [pip](https://pip.pypa.io/en/stable/installation/) (Python package installer)
*   [Git](https://git-scm.com/downloads)

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/4ryn/LLM_Document_Query
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**

    Create a `.env` file in the root directory with the following content. Replace the placeholder values with your actual API keys and URLs.

    ```
    SUPABASE_URL=your_supabase_url
    SUPABASE_KEY=your_supabase_anon_key
    QDRANT_URL=your_qdrant_cloud_url
    QDRANT_API_KEY=your_qdrant_api_key
    OPENROUTER_API_KEY=your_openrouter_api_key
    API_BEARER_TOKEN=your_secret_api_bearer_token
    QDRANT_COLLECTION_NAME=hackrx_documents
    ```

    *   `QDRANT_URL` and `QDRANT_API_KEY`: Obtain these from your Qdrant Cloud dashboard.
    *   `OPENROUTER_API_KEY`: Obtain this from your OpenRouter.ai account.
    *   `API_BEARER_TOKEN`: A secret token of your choice for authenticating API requests.

## Running the Application

1.  **Start the Backend Server:**

    Run the application using the following:

    ```bash
    python -m app.main
    ```

    The API will be accessible at `http://127.0.0.1:8000`.

2.  **Accessing the Frontend (Basic):**

    The `frontend/index.html` file provides a very basic interface. You can open this file directly in your web browser.

## API Usage

The primary API endpoint is `/api/v1/hackrx/run`.

*   **Endpoint:** `POST /api/v1/hackrx/run`
*   **Content-Type:** `application/json`
*   **Accept:** `application/json`
*   **Authentication:** `Authorization: Bearer <your_secret_api_bearer_token>`

### Request Body Example:

```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
```

*   `documents`: A URL to the document to be processed.
*   `questions`: A list of questions to be answered based on the document.

### Response Body Example:

```json
{
    "answers": [
        "The grace period for premium payment is 30 days.",
        "The waiting period for pre-existing diseases is 48 months.",
        "Yes, maternity expenses are covered after a waiting period of 24 months."
    ]
}
```

*   `answers`: A list of answers corresponding to the questions asked.

## Project Structure

```
llm-document-query/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   └── query_service.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── frontend/
│   └── in.hdextml
├── requirements.txt
├── .env
├── Procfile
├── runtime.txt
└── README.md
```
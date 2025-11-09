# MiniRAG: Retrieval-Augmented Generation System

A lightweight, interactive RAG (Retrieval-Augmented Generation) system that combines semantic search with local language model generation to answer questions based on a knowledge base of computer science and finance documents.

## 🎯 Overview

This project implements a complete RAG pipeline that:
- **Retrieves** relevant context from a knowledge base using semantic similarity search
- **Augments** the user's query with retrieved context
- **Generates** accurate, context-aware responses using a local language model

The system uses state-of-the-art embedding models and efficient vector search to provide fast, accurate answers without relying on external APIs or cloud services.

## ✨ Features

- **Semantic Search**: Uses Sentence Transformers to encode documents and queries into high-dimensional vectors
- **Efficient Retrieval**: FAISS (Facebook AI Similarity Search) for fast similarity search over large document collections
- **Local Generation**: Runs Google's FLAN-T5-base model locally for privacy and cost-effectiveness
- **Interactive CLI**: Simple command-line interface for asking questions
- **Knowledge Base**: Includes 273 documents covering computer science and finance topics

## 🏗️ Architecture

```
User Query
    ↓
Embedding Model (SentenceTransformer)
    ↓
FAISS Vector Index (Semantic Search)
    ↓
Top-K Document Retrieval
    ↓
Context Augmentation
    ↓
FLAN-T5 Generator
    ↓
Response
```

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

- `sentence-transformers`: For generating document and query embeddings
- `transformers`: For running the FLAN-T5 language model
- `faiss-cpu`: For efficient vector similarity search
- `numpy`: For numerical operations

## 💻 Usage

1. **Run the application**
   ```bash
   python main.py
   ```

2. **Wait for initialization**
   - The system will load the embedding model and build the FAISS index
   - The FLAN-T5 generator will be loaded
   - This may take a minute on first run (models are downloaded automatically)

3. **Ask questions**
   ```
   -> What is a binary search tree?
   -> Explain the time value of money
   -> How does backpropagation work?
   ```

4. **Exit the application**
   - Type `exit` or `quit` to close the application

## 🔧 How It Works

### 1. Document Embedding
Documents are encoded into dense vector representations using the `all-MiniLM-L6-v2` model, which captures semantic meaning in 384-dimensional vectors.

### 2. Index Construction
A FAISS index is built from all document embeddings, enabling fast similarity search using L2 (Euclidean) distance.

### 3. Query Processing
When a user asks a question:
1. The query is encoded into the same embedding space
2. The top-K most similar documents are retrieved (default: K=3)
3. Retrieved documents are combined into context
4. The context and query are formatted into a prompt
5. FLAN-T5 generates a response based on the augmented context

### 4. Response Generation
The generator uses the retrieved context to provide accurate, relevant answers without hallucinating information outside the knowledge base.

## 📚 Knowledge Base

The system includes a curated knowledge base covering:
- **Computer Science**: Algorithms, data structures, machine learning, neural networks, NLP, computer vision, and more
- **Finance**: Investments, stocks, bonds, trading strategies, financial metrics, and economic concepts

Total: 273 documents providing comprehensive coverage of these domains.

## 🎓 Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector Index**: FAISS IndexFlatL2 (L2 distance)
- **Language Model**: `google/flan-t5-base` (text-to-text generation)
- **Retrieval**: Top-K retrieval (default K=3)
- **Generation**: Max length 150 tokens, deterministic sampling

## 🔍 Example Queries

```
-> What is the difference between TCP and UDP?
-> Explain how gradient descent works
-> What is a dividend yield?
-> How does a hash table work?
-> What is the Sharpe ratio?
```

## 🛠️ Customization

### Modify the Knowledge Base
Edit `documents.py` to add or remove documents. The system will automatically rebuild the index on restart.

### Adjust Retrieval Parameters
Modify the `top_k` parameter in the `retrieve()` function call to retrieve more or fewer documents.

### Change the Language Model
Replace `google/flan-t5-base` with any compatible text-to-text generation model from Hugging Face.

## 📝 Limitations

- Responses are limited to 150 tokens
- Knowledge is restricted to the provided document set
- The system runs entirely locally (no internet required after initial model download)
- FLAN-T5-base is a smaller model; larger models may provide better responses but require more resources

## 🚧 Future Improvements

- [ ] Support for larger language models (LLMs)
- [ ] Web interface for easier interaction
- [ ] Support for custom document uploads
- [ ] Persistent vector index storage
- [ ] Integration with external knowledge bases
- [ ] Response streaming for better UX
- [ ] Confidence scoring for retrieved documents

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a demonstration of RAG system implementation for technical portfolios and hiring managers.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

**Note**: This project is designed to showcase RAG implementation skills. For production use, consider using larger models, implementing proper error handling, and adding more robust evaluation metrics.


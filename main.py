from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()
import numpy as np
import faiss
from documents import documents

print("Loading embedding model and building FAISS index...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve(query, top_k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]


print("Loading local LLM...")
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def rag_generate(query):
    retrieved_docs = retrieve(query, top_k=3)
    context = " ".join(retrieved_docs)
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = generator(input_text, max_length=150, do_sample=False)
    return response[0]["generated_text"]

if __name__ == "__main__":
    print("\nMiniRAG: Ask me anything!")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("-> ")
        if query.lower() in ["exit", "quit"]:
            print("Thanks for using me. Have a great day!")
            break
        answer = rag_generate(query)
        print(f"MiniRAG: {answer}\n")
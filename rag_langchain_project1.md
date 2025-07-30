
```markdown
# 🧠 RAG Pipeline Code Explanation

This document explains each line of the `rag.py` script. It covers how RAG (Retrieval-Augmented Generation) works with LangChain, Ollama, Chroma, and Llama 3.1 8B.

---

## 📦 Imports

```
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
```

| Module              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| os                  | Standard Python module for file system operations                           |
| TextLoader          | Loads `.txt` files into LangChain `Document` objects                        |
| PyPDFLoader         | Extracts text from `.pdf` files as LangChain `Document` objects             |
| RecursiveCharacterTextSplitter | Splits long documents into overlapping chunks for context         |
| Chroma              | Vector database for storing/retrieving embeddings                           |
| OllamaEmbeddings    | Embedding model running locally via Ollama                                  |
| OllamaLLM           | LLM interface for models like Llama 3.1 via Ollama                           |
| RetrievalQA         | LangChain chain to connect retriever with QA pipeline                       |

---

## 1️⃣ Load Documents (`.txt` and `.pdf`)

```
def load_documents(folder_path):
    docs = []
    for fname in os.listdir(folder_path):
        full_path = os.path.join(folder_path, fname)
        if fname.endswith(".txt"):
            loader = TextLoader(full_path)
            docs.extend(loader.load())
        elif fname.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
            docs.extend(loader.load())
    return docs

docs = load_documents("/Users/niyazea/Documents/Work/train_documents/")
```

- Loads documents from a directory.
- Supports both `.txt` and `.pdf` files.
- Returns a list of `Document` objects.

---

## 2️⃣ Split Documents into Chunks

```
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)
split_docs = text_splitter.split_documents(docs)
split_docs = [doc for doc in split_docs if doc.page_content.strip()]
```

- `chunk_size=512`: Max characters per chunk
- `chunk_overlap=60`: Overlapping context between chunks
- Blank chunks are filtered out.

---

## 3️⃣ Generate Embeddings with Ollama

```
embedding = OllamaEmbeddings(model="nomic-embed-text")
```

- Converts text chunks into vector embeddings.
- Runs locally with Ollama.

---

## 4️⃣ (Optional) Debug Embedding Generation

```
for i, doc in enumerate(split_docs):
    emb = embedding.embed_query(doc.page_content)
    if not emb:
        print(f"⚠️ No embedding for chunk {i}")
    else:
        print(f"✅ Embedded chunk {i}")
```

- Confirms that each chunk is embedded properly.
- Useful for spotting missing or invalid chunks.

---

## 5️⃣ Store in Chroma Vector Database

```
vectorstore = Chroma.from_documents(
    split_docs,
    embedding,
    persist_directory="./chroma_db"
)
```

- Stores chunk embeddings in a ChromaDB database for later retrieval.
- `persist_directory` path ensures the DB is saved on disk.

---

## 6️⃣ Initialize the LLM (Ollama: Llama 3.1)

```
llm = OllamaLLM(model="llama3.1:8b")
```

- Uses a locally running Llama 3.1 8B model from Ollama.
- Used later for answering questions.

---

## 7️⃣ Set up RetrievalQA Pipeline

```
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)
```

- Retrieves relevant documents using Chroma.
- Passes them to the LLM along with the input question (`stuff` all chunks into the prompt).
- Optionally includes document sources in the result.

---

## 8️⃣ Ask a Question

```
query = "provide summary of the 2025 Q2 results for the company Intel Corporation"
result = qa_chain.invoke(query)
```

- This performs a search over your embedded documents.
- The results are added to the LLM’s prompt to generate an answer.

---

## 📤 Output Answer & Sources

```
print("\n🧠 Answer:\n", result["result"])
print("\n📄 References:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "Unknown source"))
```

- Displays the generated answer.
- Lists files or chunks used to generate that answer.

---

## ✅ Summary of Features

| Feature                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Multi-format loading      | Supports `.txt` and `.pdf` inputs                                           |
| Chunking                 | Splits documents intelligently with overlapping context                     |
| Local embeddings         | Generates vector embeddings without using external APIs (via Ollama)        |
| Vector search            | Finds relevant text efficiently using Chroma                                |
| LLM answering            | Generates context-aware answers using Llama 3.1 locally with Ollama          |
| Transparent QA           | Outputs both answer and its document sources                                |

---

> ✨ This pipeline is ideal for private, offline document understanding tasks — e.g. financial reports, internal guides, technical documentation and more.

```

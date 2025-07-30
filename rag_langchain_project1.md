#### Imports

```python
import os
```
- **os**: Standard Python library for interacting with the operating system (file paths, directory listing, etc.).

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader
```
- **TextLoader**: Loads plain text files as LangChain `Document` objects.
- **PyPDFLoader**: Loads PDF files as LangChain `Document` objects, extracting text from each page.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
- **RecursiveCharacterTextSplitter**: Splits long documents into smaller chunks, preserving context and avoiding splitting in the middle of words/sentences.

```python
from langchain_chroma import Chroma
```
- **Chroma**: Interface to the Chroma vector database, used for storing and retrieving vector embeddings for documents.

```python
from langchain_ollama.embeddings import OllamaEmbeddings
```
- **OllamaEmbeddings**: Generates vector embeddings for text using an Ollama-compatible embedding model.

```python
from langchain_ollama import OllamaLLM
```
- **OllamaLLM**: Interface to interact with an Ollama-hosted large language model (LLM), such as Llama 3.1.

```python
from langchain.chains import RetrievalQA
```
- **RetrievalQA**: A chain that combines retrieval (from a vector store) and question-answering using an LLM.

---

#### 1. Load documents from .txt and .pdf files

```python
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
```
- **load_documents**: Function to load all `.txt` and `.pdf` files from a folder.
- **os.listdir(folder_path)**: Lists all files in the folder.
- **os.path.join**: Joins folder path and filename.
- **TextLoader(full_path)**: Loads text files.
- **PyPDFLoader(full_path)**: Loads PDF files.
- **loader.load()**: Reads the file and returns a list of `Document` objects.
- **docs.extend(...)**: Adds loaded documents to the list.

```python
docs = load_documents("/Users/niyazea/Documents/Work/train_documents/")
```
- Loads all documents from the specified folder.

---

#### 2. Split documents into manageable chunks

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)
```
- **chunk_size=512**: Each chunk will be up to 512 characters.
- **chunk_overlap=60**: Overlap 60 characters between chunks for context continuity.

```python
split_docs = text_splitter.split_documents(docs)
```
- Splits each document into smaller chunks.

```python
split_docs = [doc for doc in split_docs if doc.page_content.strip()]
```
- Removes empty chunks (where `page_content` is blank or whitespace).

---

#### 3. Generate embeddings using an Ollama-compatible embedding model

```python
embedding = OllamaEmbeddings(model="nomic-embed-text")  # Or "mxbai-embed-large"
```
- Initializes an embedding model for converting text to vectors.

---

#### 4. (Optional) Debug embedding generation

```python
for i, doc in enumerate(split_docs):
    emb = embedding.embed_query(doc.page_content)
    if not emb:
        print(f"⚠️ No embedding for chunk {i}: {doc.page_content[:100]!r}")
    else:
        print(f"✅ Embedded chunk {i} successfully.")
```
- For each chunk, generates its embedding.
- Prints a warning if embedding fails, or a success message otherwise.

---

#### 5. Store in Chroma vector database

```python
vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="./chroma_db")
```
- Stores all document chunks and their embeddings in a Chroma vector database, persisted to disk.

---

#### 6. Initialize LLM (Llama 3.1 8B via Ollama)

```python
llm = OllamaLLM(model="llama3.1:8b")
```
- Sets up the Llama 3.1 8B model for answering questions.

---

#### 7. Build a RetrievalQA chain using your vector store and LLM

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)
```
- **RetrievalQA.from_chain_type**: Creates a QA chain.
- **llm=llm**: Uses the Llama model for answering.
- **chain_type="stuff"**: Simple chain type that stuffs retrieved docs into the prompt.
- **retriever=vectorstore.as_retriever()**: Uses Chroma to retrieve relevant chunks.
- **return_source_documents=True**: Returns the source documents used for the answer.

---

#### 8. Run a query

```python
# query = "Summarize the main risks in the project."
query = "provide summary of the 2025 Q2 results for the company Intel Corporation"
result = qa_chain.invoke(query)  # .invoke() works well in LangChain 0.2+
```
- Defines a query and runs it through the QA chain.
- **qa_chain.invoke(query)**: Retrieves relevant docs, passes them to the LLM, and gets an answer.

---

#### Output the response

```python
print("\n🧠 Answer:\n", result["result"])
print("\n📄 References:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "Unknown source"))
```
- Prints the answer from the LLM.
- Prints the sources (file names or metadata) of the documents used to generate the answer.

---

### Summary of Features

- **Document loading**: Handles both text and PDF files.
- **Text splitting**: Breaks documents into overlapping chunks for better retrieval.
- **Embeddings**: Converts text to vectors for semantic search.
- **Vector database**: Stores and retrieves document vectors efficiently.
- **LLM integration**: Uses a local Llama model for answering questions.
- **Retrieval-augmented generation**: Combines retrieval and generation for accurate, source-grounded answers.
- **Source transparency**: Shows which documents were used for each answer.

Let me know if you want more detail on any specific part!

niyazelimale: provide same above info in github.com .md format

GitHub Copilot: # RAG Pipeline Code Explanation

This document explains each line of the rag.py script, including the features of the imported functions and classes.

---

## Imports

```python
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
```

- **os**: Standard Python library for interacting with the operating system (file paths, directory listing, etc.).
- **TextLoader**: Loads plain text files as LangChain `Document` objects.
- **PyPDFLoader**: Loads PDF files as LangChain `Document` objects, extracting text from each page.
- **RecursiveCharacterTextSplitter**: Splits long documents into smaller chunks, preserving context.
- **Chroma**: Interface to the Chroma vector database, used for storing and retrieving vector embeddings for documents.
- **OllamaEmbeddings**: Generates vector embeddings for text using an Ollama-compatible embedding model.
- **OllamaLLM**: Interface to interact with an Ollama-hosted large language model (LLM), such as Llama 3.1.
- **RetrievalQA**: A chain that combines retrieval (from a vector store) and question-answering using an LLM.

---

## 1. Load Documents from .txt and .pdf Files

```python
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

- **load_documents**: Loads all `.txt` and `.pdf` files from a folder.
- **os.listdir(folder_path)**: Lists all files in the folder.
- **os.path.join**: Joins folder path and filename.
- **TextLoader(full_path)**: Loads text files.
- **PyPDFLoader(full_path)**: Loads PDF files.
- **loader.load()**: Reads the file and returns a list of `Document` objects.
- **docs.extend(...)**: Adds loaded documents to the list.

---

## 2. Split Documents into Manageable Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)
split_docs = text_splitter.split_documents(docs)
split_docs = [doc for doc in split_docs if doc.page_content.strip()]
```

- **chunk_size=512**: Each chunk will be up to 512 characters.
- **chunk_overlap=60**: Overlap 60 characters between chunks for context continuity.
- **split_documents**: Splits each document into smaller chunks.
- **Removes empty chunks**: Filters out chunks with blank content.

---

## 3. Generate Embeddings Using an Ollama-Compatible Embedding Model

```python
embedding = OllamaEmbeddings(model="nomic-embed-text")  # Or "mxbai-embed-large"
```

- Initializes an embedding model for converting text to vectors.

---

## 4. (Optional) Debug Embedding Generation

```python
for i, doc in enumerate(split_docs):
    emb = embedding.embed_query(doc.page_content)
    if not emb:
        print(f"⚠️ No embedding for chunk {i}: {doc.page_content[:100]!r}")
    else:
        print(f"✅ Embedded chunk {i} successfully.")
```

- For each chunk, generates its embedding.
- Prints a warning if embedding fails, or a success message otherwise.

---

## 5. Store in Chroma Vector Database

```python
vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="./chroma_db")
```

- Stores all document chunks and their embeddings in a Chroma vector database, persisted to disk.

---

## 6. Initialize LLM (Llama 3.1 8B via Ollama)

```python
llm = OllamaLLM(model="llama3.1:8b")
```

- Sets up the Llama 3.1 8B model for answering questions.

---

## 7. Build a RetrievalQA Chain Using Your Vector Store and LLM

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)
```

- **RetrievalQA.from_chain_type**: Creates a QA chain.
- **llm=llm**: Uses the Llama model for answering.
- **chain_type="stuff"**: Simple chain type that stuffs retrieved docs into the prompt.
- **retriever=vectorstore.as_retriever()**: Uses Chroma to retrieve relevant chunks.
- **return_source_documents=True**: Returns the source documents used for the answer.

---

## 8. Run a Query

```python
# query = "Summarize the main risks in the project."
query = "provide summary of the 2025 Q2 results for the company Intel Corporation"
result = qa_chain.invoke(query)  # .invoke() works well in LangChain 0.2+
```

- Defines a query and runs it through the QA chain.
- **qa_chain.invoke(query)**: Retrieves relevant docs, passes them to the LLM, and gets an answer.

---

## Output the Response

```python
print("\n🧠 Answer:\n", result["result"])
print("\n📄 References:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "Unknown source"))
```

- Prints the answer from the LLM.
- Prints the sources (file names or metadata) of the documents used to generate the answer.

---

## Summary of Features

- **Document loading**: Handles both text and PDF files.
- **Text splitting**: Breaks documents into overlapping chunks for better retrieval.
- **Embeddings**: Converts text to vectors for semantic search.
- **Vector database**: Stores and retrieves document vectors efficiently.
- **LLM integration**: Uses a local Llama model for answering questions.
- **Retrieval-augmented generation**: Combines retrieval and generation for accurate, source-grounded answers.
- **Source transparency**: Shows which documents were used for each answer.

---

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 1. Load documents from .txt and .pdf files
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

# 2. Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)
split_docs = text_splitter.split_documents(docs)
split_docs = [doc for doc in split_docs if doc.page_content.strip()]

# 3. Generate embeddings using an Ollama-compatible embedding model
embedding = OllamaEmbeddings(model="nomic-embed-text")  # Or "mxbai-embed-large"

# 4. (Optional) Debug embedding generation
for i, doc in enumerate(split_docs):
    emb = embedding.embed_query(doc.page_content)
    if not emb:
        print(f"⚠️ No embedding for chunk {i}: {doc.page_content[:100]!r}")
    else:
        print(f"✅ Embedded chunk {i} successfully.")

# 5. Store in Chroma vector database
vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="./chroma_db")

# 6. Initialize LLM (Llama 3.1 8B via Ollama)
llm = OllamaLLM(model="llama3.1:8b")

# 7. Build a RetrievalQA chain using your vector store and LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

# 8. Run a query
# query = "Summarize the main risks in the project."
query = "provide summary of the 2025 Q2 results for the company Intel Corporation"
result = qa_chain.invoke(query)  # .invoke() works well in LangChain 0.2+

# Output the response
print("\n🧠 Answer:\n", result["result"])
print("\n📄 References:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "Unknown source"))

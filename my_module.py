import os
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# import qdrant_client

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex



# Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)


def setup_pinecone_vector_index(use_directory=False, use_local=False):
    
    if use_local:
        documents = SimpleDirectoryReader(input_dir="./data").load_data()
        return VectorStoreIndex.from_documents(documents)
    
    # Khởi tạo kết nối với Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("test-gemini")
    # pinecone_index.delete(deleteAll=True)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    if use_directory:
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.storage.docstore import SimpleDocumentStore

        documents = SimpleDirectoryReader("./data", filename_as_id=True).load_data()
        
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                GoogleGenAIEmbedding(
                    model_name="gemini-embedding-exp-03-07",
                    # api_key=os.getenv("GOOGLE_API_KEY")
                ),
            ],
            vector_store=vector_store,
            docstore=SimpleDocumentStore(),
            docstore_strategy="upserts"
        )

        nodes = pipeline.run(documents=documents)
        print(f"Số lượng Node đã được xử lý: {len(nodes)}")

        return VectorStoreIndex.from_vector_store(vector_store)
    else:
        return VectorStoreIndex.from_vector_store(vector_store)


def setup_qdrant_vector_index(use_directory=False, use_local=False, collection_name="test-gemini"):
    
    if use_local:
        documents = SimpleDirectoryReader(input_dir="./data").load_data()
        return VectorStoreIndex.from_documents(documents)
    
    # Khởi tạo kết nối với Qdrant
    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY")  # Optional nếu không có authentication
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name
    )

    if use_directory:
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.storage.docstore import SimpleDocumentStore

        documents = SimpleDirectoryReader("./data", filename_as_id=True).load_data()
        
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                GoogleGenAIEmbedding(
                    model_name="gemini-embedding-exp-03-07",
                    api_key=os.getenv("GOOGLE_API_KEY")
                ),
            ],
            vector_store=vector_store,
            docstore=SimpleDocumentStore(),
            docstore_strategy="upserts"
        )

        nodes = pipeline.run(documents=documents)
        print(f"Số lượng Node đã được xử lý với Qdrant: {len(nodes)}")

        return VectorStoreIndex.from_vector_store(vector_store)
    else:
        return VectorStoreIndex.from_vector_store(vector_store)

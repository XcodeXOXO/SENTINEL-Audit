import os
import json
import logging
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Configuration
VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "vector_store"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

class DataIngestion:
    def __init__(self):
        # We ensure idempotent operations: use ChromaDB with a persistent directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")
        self.vector_db = Chroma(
            collection_name="sentinel_knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=str(VECTOR_STORE_DIR)
        )

    def ingest_raw_markdown(self):
        """
        Ingests raw markdown vulnerability reports into Chroma.
        Idempotent operation: Checks if document names already exist.
        """
        logging.info(f"Looking for raw markdown in {RAW_DATA_DIR}")
        
        if not RAW_DATA_DIR.exists():
            logging.error("Raw data directory does not exist.")
            return

        docs_to_add = []
        for file_path in RAW_DATA_DIR.glob("*.md"):
            # Simple idempotency: use file name as ID for Chroma to prevent duplicate chunking loads if simplistic.
            docs_to_add.append(file_path)

        if not docs_to_add:
            logging.info("No raw md files found to ingest.")
            return
            
        logging.info(f"Found {len(docs_to_add)} files to ingest. Adding them now...")
        # Simplistic demonstration ingestion logic
        # For a full system, split documents appropriately.
        for path in docs_to_add:
            try:
                loader = TextLoader(str(path))
                docs = loader.load()
                # Use filename as ID
                ids = [f"{path.name}"]
                # Attempt to get docs with these ids
                existing = self.vector_db.get(ids=ids)
                if not existing or not existing.get('ids'):
                    self.vector_db.add_documents(documents=docs, ids=ids)
                    logging.info(f"Ingested {path.name}")
                else:
                    logging.info(f"Skipping {path.name}, already in index.")
            except Exception as e:
                logging.error(f"Error ingesting {path.name}: {e}")

if __name__ == "__main__":
    ingestor = DataIngestion()
    ingestor.ingest_raw_markdown()

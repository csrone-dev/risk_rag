import os
import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def model_tune_params(path, model_name, chunk_sizes, chunk_overlaps):
    tuned_params = []

    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            directory = f"./chroma_db/{model_name.replace('/', '_')}_{path.replace('2024_report_transformed/', '').replace('.pdf', '')}"
            collection = f"chunk_size_{chunk_size}_overlap_{chunk_overlap}"
            tuned_params.append(
                {
                    "model_name": model_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "persist_directory": directory,
                    "collection_name": collection,
                }
            )
    return tuned_params


def split_embedding_createDB(
    path, model_name, chunk_size, chunk_overlap, persist_directory, collection_name
):
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # load and split the Document
    loader = PyPDFLoader(path)

    splitter = RecursiveCharacterTextSplitter(
        separators="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    texts = loader.load_and_split(splitter)
    print("Number of chunks after splitting: ", len(texts))

    print("Adjusting page metadata from 0-based index to 1-based page number...")
    for index, doc in enumerate(texts):
        if "page" in doc.metadata and isinstance(doc.metadata["page"], int):
            doc.metadata["page"] += 1
        else:
            print(
                f"Warning: Document metadata missing 'page' key or 'page' is not an integer. Metadata: {doc.metadata}"
            )

        chunk_id = f"chunk_{index}"
        doc.metadata["chunk_id"] = chunk_id

    print(f"Collection name: {collection_name}")

    # create embeddings and store in chromaDB
    vectorDB = Chroma.from_documents(
        documents=texts,
        embedding=hf_embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(
        f"ChromaDB directory '{model_name.replace('/', '_')}' collection '{collection_name}' has been created"
    )

    return vectorDB


if __name__ == "__main__":
    dir_path = "2024_report_transformed"
    paths = [f for f in os.listdir("2024_report_transformed") if f.endswith(".pdf")]
    model_name = "moka-ai/m3e-base"
    chunk_sizes = [300]  # 200, 300, 500, 800
    chunk_overlaps = [50]  # 50, 100

    for path in paths:
        path = os.path.join(dir_path, path)
        print(path.replace("2024_report_transformed/2024年永續報告書(中)", ""))
        params = model_tune_params(path, model_name, chunk_sizes, chunk_overlaps)

        for tuned_params in params:
            split_embedding_createDB(
                path=path,
                model_name=tuned_params["model_name"],
                chunk_size=tuned_params["chunk_size"],
                chunk_overlap=tuned_params["chunk_overlap"],
                persist_directory=tuned_params["persist_directory"],
                collection_name=tuned_params["collection_name"],
            )

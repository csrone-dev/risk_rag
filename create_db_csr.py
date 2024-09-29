import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def model_tune_params(path, model_names, chunk_sizes, chunk_overlaps):
    tuned_params = []
    
    for model_name in model_names:
        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                directory = f"./chroma_db/{model_name.replace('/', '_')}_{path.replace('csr_reports/', '').replace('.pdf', '')}"
                collection = f"chunk_size_{chunk_size}_overlap_{chunk_overlap}"
                tuned_params.append({
                    "model_name": model_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "persist_directory": directory,
                    "collection_name": collection
                })
    return tuned_params

def split_embedding_createDB(path, model_name, chunk_size, chunk_overlap, persist_directory, collection_name):
    hf_embeddings = HuggingFaceEmbeddings(model_name = model_name)
    
    # load and split the Document
    loader = PyPDFLoader(path)

    splitter = RecursiveCharacterTextSplitter(
        separators="\n", 
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap
    )

    texts = loader.load_and_split(splitter)
    print("Number of chunks after splitting: ", len(texts))

    print(f"Collection name: {collection_name}")

    # create embeddings and store in chromaDB
    vectorDB = Chroma.from_documents(
        documents = texts, 
        embedding= hf_embeddings, 
        persist_directory = persist_directory,
        collection_name = collection_name
    )
    print(f"ChromaDB directory '{model_name.replace('/', '_')}' collection '{collection_name}' has been created")

    return vectorDB


def main():
    path = "csr_reports/2022_05155853.pdf"
    
    # model_names = ["aspire/acge_text_embedding", "intfloat/multilingual-e5-base", "BAAI/bge-large-zh-v1.5"]
    model_names = ["BAAI/bge-large-zh-v1.5"]
    chunk_sizes = [200] # 200, 300, 400, 500
    chunk_overlaps = [50] # 50, 100

    params = model_tune_params(path, model_names, chunk_sizes, chunk_overlaps)

    for tuned_params in params:
        split_embedding_createDB(
            path = path, 
            model_name = tuned_params["model_name"], 
            chunk_size = tuned_params["chunk_size"], 
            chunk_overlap = tuned_params["chunk_overlap"],            
            persist_directory = tuned_params["persist_directory"],
            collection_name = tuned_params["collection_name"]
        )

if __name__ == "__main__":
    main()


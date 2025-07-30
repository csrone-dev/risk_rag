import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = ['Times New Roman', 'AppleGothic']  # 先用 Times New Roman，找不到中文字才 fallback 用 AppleGothic
plt.rcParams['axes.unicode_minus'] = False

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_collection_names(chunk_sizes, chunk_overlaps):
    collection_names = []
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            collection_name = f"chunk_size_{chunk_size}_overlap_{chunk_overlap}"
            collection_names.append(collection_name)
    return collection_names


def query_retrieval(
    risk_info_path,
    report_path,
    model_name,
    chunk_sizes,
    chunk_overlaps,
    percentiles,
    selected_risks=None
):
    df = pd.read_csv(risk_info_path)
    collection_names = get_collection_names(chunk_sizes, chunk_overlaps)
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    results = []

    # 轉換風險名稱為 index
    if selected_risks is not None:
        selected_risk_indices = df[df["風險名稱"].isin(selected_risks)].index.tolist()
    else:
        selected_risk_indices = df.index.tolist()  # 如果沒指定就全部跑


    for collection_name in collection_names:
        db = Chroma(
            persist_directory=f"./chroma_db/{model_name.replace('/', '_')}_{report_path.replace('2024_report_transformed/', '').replace('.pdf', '')}",
            collection_name=collection_name,
            embedding_function=embedding_function,
            collection_metadata = {"hnsw:space": "cosine"}
        )

        for idx in selected_risk_indices:
            report_name = os.path.basename(report_path).replace(".pdf", "").replace("2024年永續報告書(中)", "")
            row = df.iloc[idx]
            risk_name = row["風險名稱"]
            definition = row["風險定義"]
            query = f"{risk_name}是指{definition}"
            print(f"\n▶ 查詢風險：{risk_name}\n→ {query}")

            retrieved_results = db.similarity_search_with_relevance_scores(query=query, k=5000, score_threshold=0)
            retrieved_scores = np.array([score for _, score in retrieved_results])

            result_row = {}
            result_row["report_name"] = report_name
            result_row["risk_name"] =  risk_name

            for p in percentiles:
                percentile_score = np.percentile(retrieved_scores, p)
                if max(retrieved_scores) < 0.8:
                    threshold = "top-3"
                    selected_chunks_info = []
                    for doc, score in retrieved_results[:3]:
                        selected_chunks_info.append({
                            # "page": doc.metadata.get("page"),
                            # "score": score,
                            "chunk_id": doc.metadata.get("chunk_id"),
                            "content": doc.page_content
                        })
                else:
                    threshold = max(percentile_score, 0.8)
                    selected_chunks_info = []
                    for doc, score in retrieved_results:
                        if score >= threshold:
                            selected_chunks_info.append({
                                # "page": doc.metadata.get("page"),
                                # "score": score,
                                "chunk_id": doc.metadata.get("chunk_id"),
                                "content": doc.page_content
                            })

                # unique_pages = set(chunk["page"] for chunk in selected_chunks_info)

                result_row["score"] = threshold
                # result_row["pages"] = sorted(unique_pages)
                # result_row["num_page"] = len(unique_pages)
                result_row["chunks"] = selected_chunks_info
                result_row["num_chunk"] = len(selected_chunks_info)

            results.append(result_row)
    return results


if __name__ == "__main__":
    risk_info_path = "risk_info.csv"
    report_paths = [f for f in os.listdir("2024_report_transformed/phase3") if f.endswith('.pdf')]
    # report_paths = ["2024_report_transformed/2024年永續報告書(中)皇普2528.pdf"]
    model_name = "moka-ai/m3e-base" 
    chunk_sizes = [300]
    chunk_overlaps = [50] 
    percentiles=[98] 


    for report_path in report_paths:
        start_time = time.time()
        # print(report_path)
        company_name = report_path.replace("2024年永續報告書(中)", "").replace(".pdf", "")
        print(company_name)

        results = query_retrieval(
            risk_info_path,
            report_path,
            model_name,
            chunk_sizes,
            chunk_overlaps,
            percentiles,

        )
        print(results)


        # === 寫出所有報告書的所有風險結果（all info for myself）===
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"output_retrieval/retrieve_page_all_info_{company_name}.csv", index=False, encoding="utf-8")
        print("Detailed retrieval info stored.")
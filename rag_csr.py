import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from rerankers import Reranker, Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    api_key = api_key,
    model = "gpt-4o-mini",
    temperature = 0.5,
    max_tokens = 1500
)

def get_collection_names(chunk_sizes, chunk_overlaps):
    collection_names = []
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            collection_name = f"chunk_size_{chunk_size}_overlap_{chunk_overlap}"
            collection_names.append(collection_name)
    return collection_names

def create_user_template(pointer_name, definition, top_chunks):
    # to-do: change the response format to json (boolean & str)
    return f"""
        指標名稱：{pointer_name}
        指標定義：{definition}
        
        ### 任務 ###
        根據以下內容判斷此報告書是否揭露 "{pointer_name}"：
        {top_chunks}

        ### 回覆格式 ###
        如果報告書中沒有揭露 "{pointer_name}"，回覆「答案：0」即可，不需提供其他資訊。
        如果報告書中有揭露 "{pointer_name}"，回覆「答案：1，理由：」，其中「理由：」後要接模型判斷依據的原文，也就是報告書中的哪些部分提供了這些資訊。            

        答案:
    """

def load_reranker(reranker):
    ranker = Reranker(reranker, model_type = "cross-encoder")

    return ranker

def query_retrieval(pointer_path, csr_path, model_names, ranker, chunk_sizes, chunk_overlaps, num_chunks, rerank_chunks):
    df = pd.read_csv(pointer_path)
    collection_names = get_collection_names(chunk_sizes, chunk_overlaps)

    for model_name in model_names:
        embedding_function = SentenceTransformerEmbeddings(model_name = model_name)
        for collection_name in collection_names:
            db = Chroma(
                persist_directory = f"./chroma_db/{model_name.replace('/', '_')}_{csr_path.replace('csr_reports/', '').replace('.pdf', '')}",
                collection_name = collection_name,
                embedding_function = embedding_function
            )
            
            results = []
            retry_pointers = []
            for idx, row in df.iterrows():
                pointer_num = row["pointer_num"]
                pointer_name = row["pointer_name"]
                definition = row["description"]
                query = f'此報告書是否揭露"{pointer_name}"? "{pointer_name}"定義如下：{definition}'

                retrieved_results = db.similarity_search(query = query, k = num_chunks)

                # re-rank from retrieved results
                docs = [Document(text = result.page_content, metadata = result.metadata) for result in retrieved_results]
                rerank_results = ranker.rank(query = query, docs = docs)

                retrieved_res_content = [rerank_result.text for rerank_result in rerank_results]
                top_chunks = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(retrieved_res_content[:rerank_chunks])])
                # print(top_chunks)

                user_template = create_user_template(pointer_name, definition, top_chunks)

                prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "你是企業社會責任專家，你的任務是判斷此報告書中是否揭露給定的風險指標，用 zh-TW 回答。"),
                        ("human", "{input}")
                ])

                chain = prompt_template | model

                # to-do: change it to parallel computing & take back the results at once
                with get_openai_callback() as cb:
                    res = chain.invoke({"input": user_template})
                    print(cb, "\n")

                if "答案：0" not in res.content and "答案：1" not in res.content:
                    retry_pointers.append([pointer_num, pointer_name, definition, top_chunks])
                else:
                    results.append([pointer_num, res.content])
                    print(res.content) # type: str

            # call gpt api again for answers with correct format until no pointer has to be retried
            print(f"retry pointers: {retry_pointers}")
            while retry_pointers: 
                for i, (pointer_num, pointer_name, definition, top_chunks) in enumerate(retry_pointers):
                    create_user_template(pointer_name, definition, top_chunks)

                    with get_openai_callback() as cb:
                        res = chain.invoke({"input": user_template})
                        print(cb, "\n")

                    if "答案：0" in res.content or "答案：1" in res.content:
                        del retry_pointers[i]
                        results.append([pointer_num, res.content])
                        print(res.content) # type: str

            print(results)

    return results 

def get_report_info(csr_path):
    pattern = r"(\d{4})_(\d+)\.pdf"
    match = re.search(pattern, csr_path)
    report_year = match.group(1)
    company_num = match.group(2)
    company_num = company_num.zfill(8)

    return report_year, company_num

def get_output(csr_path, answer_path, results, output_df):
    df_answer = pd.read_csv(answer_path)
    df_answer = df_answer.dropna(subset = ["A0002"]) # drop NA
    df_answer["A0002"] = df_answer["A0002"].astype(int).astype(str) # float -> int -> str (for loc)
    df_answer["A0002"] = df_answer["A0002"].str.zfill(8) # company_num needs to be 8 digits
    df_answer.set_index("A0002", inplace = True)
    
    correct = 0
    output = {}
    company_num = get_report_info(csr_path)[1]
    output["company_num"] = company_num

    for result in results:
        pointer_num = result[0]
        result_content = result[1]

        if "答案：0" in result_content:
            output[f"{pointer_num}_output"] = 0
            output[f"{pointer_num}_answer"] = df_answer.loc[company_num, pointer_num] 
            output[f"{pointer_num}_reason"] = ""
        else: # "答案：1" 
            output[f"{pointer_num}_output"] = 1
            output[f"{pointer_num}_answer"] = df_answer.loc[company_num, pointer_num]
            output[f"{pointer_num}_reason"] = result_content.split('理由：')[1]

        correct += int(output[f"{pointer_num}_output"] == output[f"{pointer_num}_answer"]) # see if the output is correct
    
    output["accuracy"] = correct / len(results)  # accuracy of the report

    # output (model's output + correct answer + reason + accuracy)
    output_df = pd.concat([output_df, pd.DataFrame([output])], ignore_index = True)  # append the company's output to output_df 
    print(f"The output for {company_num} has been created.")

    return output_df

def main():
    csr_paths = ["csr_reports/2022_03374805.pdf", "csr_reports/2022_04200199.pdf", "csr_reports/2022_05155853.pdf"]
    pointer_path = "pointers.csv"
    answer_path = "test_risk_2022.csv"
    # chunk_sizes = [200, 300, 400, 500]
    # chunk_overlaps = [50, 100]
    # model_names = ["aspire/acge_text_embedding", "intfloat/multilingual-e5-base", "BAAI/bge-large-zh-v1.5"]
    
    model_names = ["BAAI/bge-large-zh-v1.5"]
    reranker = "BAAI/bge-reranker-base"
    chunk_sizes = [200] # 調大一點
    chunk_overlaps = [50]
    num_chunks = 10
    rerank_chunks = 3

    ranker = load_reranker(reranker)

    output_df = pd.DataFrame()
    for csr_path in csr_paths:
        start_time = time.time()
        results = query_retrieval(pointer_path, csr_path, model_names, ranker, chunk_sizes, chunk_overlaps, num_chunks, rerank_chunks)
        output_df = get_output(csr_path, answer_path, results, output_df)
        end_time = time.time()
        total_time = end_time - start_time

        output_df.loc[output_df["company_num"] == get_report_info(csr_path)[1], "Time spent (sec)"] = f"{total_time:.2f}"
        print(f"Time spent on {csr_path.replace("csr_reports/", "")}: {total_time:.2f} seconds")

    report_year = get_report_info(csr_paths[0])[0] 
    output_df.to_csv(f"{report_year}_output.csv", index = False, encoding = "utf-8") 
    print(f"The combined output for {report_year} has been created.")

if __name__ == "__main__":
    main()

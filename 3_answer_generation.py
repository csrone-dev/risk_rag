import os
import time
import pandas as pd
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
# from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import List, Literal, Optional 
from langchain_core.output_parsers import PydanticOutputParser

from prompt.zero_shot import ZERO_SHOT_PROMPT
from prompt.zero_shot_cot import ZERO_SHOT_COT_PROMPT
from prompt.few_shot import FEW_SHOT_PROMPT
from prompt.few_shot_cot import FEW_SHOT_COT_PROMPT


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# gen_model = "gpt-4.1"
gen_model = "gpt-4.1-mini"
# gen_model = "gemini-1.5-flash-001"

model = ChatOpenAI(
    api_key=api_key,
    model=gen_model,
    temperature=0,
    # max_tokens=1500,
)

DisclosureType = Literal["關鍵字", "風險描述", "風險因應"]

class ChunkDisclosure(BaseModel):
    reasoning: Optional[str] = None
    is_disclosed: int
    chunk_id: str
    sentence: Optional[str] = None
    disclosure_type: Optional[List[DisclosureType]] = None

class ChunkDisclosureList(BaseModel):
    results: List[ChunkDisclosure]
    

def create_user_template(
    prompt_type, risk_name, risk_def, top_chunks,
    pos_chunk_1, pos_1, pos_reason_1, disclosure_type_1,
    pos_chunk_2, pos_2, pos_reason_2, disclosure_type_2,
    neg_1, neg_reason_1  
):
    if prompt_type == "ZERO_SHOT_PROMPT":
        return ZERO_SHOT_PROMPT.format(
            risk_name=risk_name, risk_def=risk_def, top_chunks=top_chunks
        )
    elif prompt_type == "ZERO_SHOT_COT_PROMPT":
        return ZERO_SHOT_COT_PROMPT.format(
            risk_name=risk_name, risk_def=risk_def, top_chunks=top_chunks
        )
    elif prompt_type == "FEW_SHOT_PROMPT":
        return FEW_SHOT_PROMPT.format(
            risk_name=risk_name, risk_def=risk_def, top_chunks=top_chunks,
            pos_chunk_1=pos_chunk_1, pos_1=pos_1, disclosure_type_1=disclosure_type_1,
            pos_chunk_2=pos_chunk_2, pos_2=pos_2, disclosure_type_2=disclosure_type_2,
            neg_1=neg_1 
        )
    else :
        return FEW_SHOT_COT_PROMPT.format(
            risk_name=risk_name, risk_def=risk_def, top_chunks=top_chunks,
            pos_chunk_1=pos_chunk_1, pos_1=pos_1, pos_reason_1=pos_reason_1, disclosure_type_1=disclosure_type_1,
            pos_chunk_2=pos_chunk_2, pos_2=pos_2, pos_reason_2=pos_reason_2, disclosure_type_2=disclosure_type_2,
            neg_1=neg_1, neg_reason_1=neg_reason_1 
        )


def gen_answer(
    prompt_type, risk_name, risk_def, top_chunks,
    pos_chunk_1, pos_1, pos_reason_1, disclosure_type_1,
    pos_chunk_2, pos_2, pos_reason_2, disclosure_type_2,
    neg_1, neg_reason_1 
):
    parser = PydanticOutputParser(pydantic_object=ChunkDisclosureList)
    prompt = create_user_template(
        prompt_type, risk_name, risk_def, top_chunks,
        pos_chunk_1, pos_1, pos_reason_1, disclosure_type_1,
        pos_chunk_2, pos_2, pos_reason_2, disclosure_type_2,
        neg_1, neg_reason_1  
    )
    # print(prompt)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是企業永續專家，你的任務是判斷此報告書中是否揭露給定的風險與其揭露類別。如果 disclosure_type 有「關鍵字」，務必把 sentence 中的關鍵詞前後用 ** 包起來。用 zh-TW 回答。",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt_template | model | parser
    # chain = prompt_template | model 

    # response = chain.invoke({"input": prompt})
    # res, cost, tokens = response.dict(), 0, 0
    
    with get_openai_callback() as cb:
        response = chain.invoke({"input": prompt})
        res = response.model_dump()
        cost = cb.total_cost
        tokens = cb.total_tokens
        
    return res, cost, tokens


if __name__ == "__main__":
    # report_paths = [f for f in os.listdir("retrieve_output") if f.endswith(".csv")]
    report_paths = ["output_retrieval/retrieve_page_all_info_上海商銀5876.csv"]
    risk_info = pd.read_csv("risk_info.csv")
    few_shot_examples = pd.read_csv("risk_examples.csv")

    model_name = "moka-ai/m3e-base"  
    chunk_sizes = [300] 
    chunk_overlaps = [50] 
    prompt_types = ["ZERO_SHOT_PROMPT", "ZERO_SHOT_COT_PROMPT", "FEW_SHOT_PROMPT", "FEW_SHOT_COT_PROMPT"]
    type_keys = ["關鍵字", "風險描述", "風險因應"]
    
    output = []
    for report_path in report_paths:
        need_to_retry = []
        start_time = time.time()
        total_cost = 0
        total_tokens = 0

        company = report_path.replace("output_retrieval/retrieve_page_all_info_", "").replace(".csv", "")
        retrieved_chunk_df = pd.read_csv(report_path)
        company_chunk_df = retrieved_chunk_df[retrieved_chunk_df["report_name"] == company]

        collection_name = f"chunk_size_300_overlap_50"
        embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        db = Chroma(
            persist_directory=f"./chroma_db/{model_name.replace('/', '_')}_2024年永續報告書(中){company}",
            collection_name=collection_name,
            embedding_function=embedding_function,
            collection_metadata = {"hnsw:space": "cosine"}
        )          

        for idx, row in risk_info.iloc[1:].iterrows():
            try:
                risk_num = row["風險代碼"]
                risk_name = row["風險名稱"]
                risk_def = row["風險定義（統整）"]
                print(risk_name)

                # 若需要重跑
                # rerun_risks = ["業務風險"]
                # if risk_name not in rerun_risks:
                #     continue  # 跳過不是目標風險代碼的項目
                # print(risk_name)

                # get few-shot examples
                example_rows = few_shot_examples[few_shot_examples["風險名稱"] == risk_name]
                ex = example_rows.iloc[0]
                pos_chunk_1 = ex["正例1_chunk"]
                pos_1 = ex["正例1_bold"]
                pos_reason_1 = ex["正例1_原因"]
                disclosure_type_1 = [
                    key for key in type_keys 
                    if ex.get(f"{key}_1", False) 
                ]

                pos_chunk_2 = ex["正例2_chunk"]
                pos_2 = ex["正例2_bold"]
                pos_reason_2 = ex["正例2_原因"]
                disclosure_type_2 = [
                    key for key in type_keys
                    if ex.get(f"{key}_2", False) 
                ]
                
                neg_1 = ex["反例1"]
                neg_reason_1 = ex["反例1_原因"]

                # get retrieved chunks
                risk_chunk_row = company_chunk_df[company_chunk_df["risk_name"] == risk_name]
                top_chunks = risk_chunk_row["chunks"].iloc[0]

                for prompt_type in prompt_types:
                    print(prompt_type)
                    llm_answer = None
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            llm_answer, cost, tokens = gen_answer(
                                prompt_type, risk_name, risk_def, top_chunks,
                                pos_chunk_1, pos_1, pos_reason_1, disclosure_type_1,
                                pos_chunk_2, pos_2, pos_reason_2, disclosure_type_2,
                                neg_1, neg_reason_1
                            )
                            total_cost += cost
                            total_tokens += tokens
                            # print(llm_answer)
                            break
                        except Exception as e:
                            print(f"第 {attempt+1} 次嘗試失敗：{e}")
                            if attempt == max_retries - 1:
                                need_to_retry.append((prompt_type, risk_name))
                                print("已達最大重試次數，跳過此項。")
                

                    if llm_answer and "results" in llm_answer:
                        disclosed_chunks = [chunk for chunk in llm_answer["results"] if chunk["is_disclosed"] == 1]
                    else:
                        continue
                    # print(disclosed_chunks)

                    for chunk in disclosed_chunks:
                        retrieved = db.get(where={"chunk_id": chunk["chunk_id"]})
                        output.append({
                            "prompt_type": prompt_type,
                            "風險代碼": risk_num, 
                            "風險名稱": risk_name, 
                            "風險定義": risk_def, 
                            "chunk_id": chunk.get("chunk_id"),
                            "頁數": retrieved["metadatas"][0]["page"],
                            "文字段落": retrieved["documents"][0],
                            "揭露句子": chunk.get("sentence"), 
                            "揭露類別": ', '.join(chunk["disclosure_type"]),
                            "模型推論過程": chunk.get("reasoning")
                        })
                        # print(output)
            except Exception as e:
                print(f"error occurred: {e}")
                
        end_time = time.time()
        total_time = end_time - start_time
        print(f"需要重試的風險: {need_to_retry}")
        print(f"花費總時間（分）: {total_time / 60}")
        print(f"model: {gen_model}，總共花費: ${total_cost: 8f} USD，使用 token 數: {total_tokens}")

        # == output for dev ===          
        output_df = pd.DataFrame(output)
        output_df.to_csv("output_generation/gen_result_for_dev_gpt4.1-mini_上海商銀5876.csv", index=False, encoding="utf-8")
        print(f"output for dev has been created.")
        
        # == output for 書院 ==
        # 刪除 prompt_type 欄位、刪除重複的資料、按照 "風險名稱" 排序讓相同的風險名稱行擺在一起
        output_df_for_sinyi = output_df.drop(columns=["prompt_type", "揭露類別", "模型推論過程"])
        output_df_for_sinyi = output_df_for_sinyi.drop_duplicates()

        # sorted_df = output_df_for_sinyi.sort_values(by="風險名稱").reset_index(drop=True)
        sorted_df = (
            output_df_for_sinyi
            .sort_values(by=["風險代碼", "chunk_id"], ascending=[True, True])
            .reset_index(drop=True)
        )

        # 把沒揭露的風險加到 sorted_df 的最底部
        existing_risks = set(sorted_df["風險代碼"].tolist())
        missing_rows = []
        for idx, row in risk_info.iterrows():
            if row["風險代碼"] not in existing_risks:
                missing_rows.append({
                    "風險代碼": row["風險代碼"],
                    "風險名稱": row["風險名稱"],
                    "風險定義": row["風險定義（統整）"],
                    "chunk_id": "",
                    "頁數": "",
                    "文字段落": "",
                    "揭露句子": "",
                })

        # 加進原本的 sorted_df 中並存成 csv
        if missing_rows:
            missing_df = pd.DataFrame(missing_rows)
            sorted_df = pd.concat([sorted_df, missing_df], ignore_index=True)

        sorted_df.to_csv("output_generation/gen_result_for_sinyi_gpt4.1-mini_上海商銀5876.csv", index=False, encoding="utf-8")
        print(f"output for sinyi has been created.")

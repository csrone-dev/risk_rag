import os
import re
import time
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import PydanticOutputParser

from prompt.v3.zero_shot import ZERO_SHOT_PROMPT
from prompt.v3.zero_shot_cot import ZERO_SHOT_COT_PROMPT
from prompt.v3.few_shot import FEW_SHOT_PROMPT
from prompt.v3.few_shot_cot import FEW_SHOT_COT_PROMPT

# ---------- 共用設定 ----------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
GEN_MODEL = "gpt-4.1-mini"
PROMPT_TYPES = [
    "ZERO_SHOT_PROMPT",
    "ZERO_SHOT_COT_PROMPT",
    "FEW_SHOT_PROMPT",
    "FEW_SHOT_COT_PROMPT",
]
TYPE_KEYS = ["關鍵字", "風險描述", "風險因應"]
MAX_WORKERS = 6  # 調高可加速，但要注意 API 速率限制
MAX_RETRIES = 5      # 單一 worker 內部重試次數

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# ---------- Pydantic schema ----------
DisclosureType = Literal["關鍵字", "風險描述", "風險因應"]


class ChunkDisclosure(BaseModel):
    reasoning: Optional[str] = None
    is_disclosed: int
    chunk_id: str
    sentence: Optional[str] = None
    disclosure_type: Optional[List[DisclosureType]] = None


class ChunkDisclosureList(BaseModel):
    results: List[ChunkDisclosure]

# ---------- 清除控制字元 ----------
def clean_ctrl(s: str) -> str:
    if not isinstance(s, str):
        return s
    return CTRL.sub("", s)

# ---------- Prompt & LLM ----------
def create_user_template(prompt_type: str, **kwargs) -> str:
    if prompt_type == "ZERO_SHOT_PROMPT":
        return ZERO_SHOT_PROMPT.format(**kwargs)
    if prompt_type == "ZERO_SHOT_COT_PROMPT":
        return ZERO_SHOT_COT_PROMPT.format(**kwargs)
    if prompt_type == "FEW_SHOT_PROMPT":
        return FEW_SHOT_PROMPT.format(**kwargs)
    return FEW_SHOT_COT_PROMPT.format(**kwargs)


def call_llm(prompt: str) -> dict:
    """單次 LLM 呼叫並回傳 (json, cost, tokens)"""
    model = ChatOpenAI(api_key=API_KEY, model=GEN_MODEL, temperature=0)
    parser = PydanticOutputParser(pydantic_object=ChunkDisclosureList)
    chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是企業永續專家，你的任務是判斷此報告書中是否揭露給定的風險與其揭露類別。如果 disclosure_type 有「關鍵字」，務必把 sentence 中的關鍵詞前後用 ** 包起來。用 zh-TW 回答。",
                ),
                ("human", "{input}"),
            ]
        )
        | model
        | parser
    )
    with get_openai_callback() as cb:
        resp = chain.invoke({"input": prompt})
        return resp.model_dump(), cb.total_cost, cb.total_tokens


# ---------- Thread worker ----------
def worker(task):
    (
        company_name,
        db,
        risk_row,
        ex_row,
        top_chunks,
        prompt_type,
    ) = task

    risk_num = risk_row["風險代碼"]
    risk_name = risk_row["風險名稱"]
    risk_def = risk_row["風險定義"]

    # if risk_name == "保險風險": # 金融業要註解掉
    #     return [], 0, 0, []
    
    if risk_name == "食品安全衛生風險": # 食品業要註解掉
        return [], 0, 0, []

    print(f"→ 正在處理：風險「{risk_name}」 | PromptType：{prompt_type}")


    if pd.notna(risk_row.get("特別注意事項", "")):
        risk_def = f"{risk_def} **特別注意：**{risk_row['特別注意事項'].strip()}"

    top_chunks = clean_ctrl(top_chunks)

    # few-shot欄位
    kwargs = dict(
        company_name=company_name,
        risk_name=risk_name,
        risk_def=risk_def,
        top_chunks=top_chunks,
        pos_chunk_1=ex_row["正例1_chunk"],
        pos_1=ex_row["正例1_bold"],
        pos_reason_1=ex_row["正例1_原因"],
        disclosure_type_1=[
            k for k in TYPE_KEYS if ex_row.get(f"{k}_1", False)
        ],
        pos_chunk_2=ex_row["正例2_chunk"],
        pos_2=ex_row["正例2_bold"],
        pos_reason_2=ex_row["正例2_原因"],
        disclosure_type_2=[
            k for k in TYPE_KEYS if ex_row.get(f"{k}_2", False)
        ],
        neg_1=ex_row["反例1"],
        neg_reason_1=ex_row["反例1_原因"],
    )
    prompt = create_user_template(prompt_type, **kwargs)
    # print(prompt)

    # --- retry loop ---
    for attempt in range(MAX_RETRIES):
        try:
            ans, cost, toks = call_llm(prompt)
            break
        except Exception as e:
            print(f"⚠️ 風險「{risk_name}」 + {prompt_type} 第 {attempt+1} 次呼叫失敗，錯誤訊息：{e}")
            if attempt == MAX_RETRIES - 1:
                return [], 0, 0, [(prompt_type, risk_name, str(e))]

    rows = []
    if ans and "results" in ans:
        for chunk in ans["results"]:
            if chunk["is_disclosed"] != 1:
                continue
            meta = db.get(where={"chunk_id": chunk["chunk_id"]})
            rows.append(
                {
                    "prompt_type": prompt_type,
                    "風險代碼": risk_num,
                    "風險名稱": risk_name,
                    "風險定義": risk_def,
                    "chunk_id": chunk["chunk_id"],
                    "頁數": meta["metadatas"][0]["page"],
                    "文字段落": meta["documents"][0],
                    "揭露句子": chunk.get("sentence"),
                    "揭露類別": ", ".join(chunk.get("disclosure_type") or []),
                    "模型推論過程": chunk.get("reasoning"),
                }
            )
    return rows, cost, toks, []


# ---------- 主程式 ----------
if __name__ == "__main__":
    report_path = "output_retrieval/retrieve_page_all_info_福邦證6026.csv" #  
    company_name = report_path.replace("output_retrieval/retrieve_page_all_info_", "").replace(".csv", "")
    model_name = "moka-ai/m3e-base" 

    risk_info = pd.read_csv("risk_info.csv")
    few_shot = pd.read_csv("risk_examples.csv")
    retrieved_df = pd.read_csv(report_path)
    company_df = retrieved_df[retrieved_df["report_name"] == company_name]

    # 建 DB (一次即可，threads 共用)
    emb = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma(
        persist_directory=f"./chroma_db/{model_name.replace('/', '_')}_2024年永續報告書(中){company_name}",
        collection_name="chunk_size_300_overlap_50",
        embedding_function=emb,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 準備所有任務
    tasks = []
    for _, r in risk_info.iloc[1:].iterrows():
        ex_rows = few_shot[few_shot["風險名稱"] == r["風險名稱"]]
        if ex_rows.empty:
            continue
        ex = ex_rows.iloc[0]

        c_row = company_df[company_df["risk_name"] == r["風險名稱"]]
        if c_row.empty:
            continue
        chunks = c_row["chunks"].iloc[0]

        for ptype in PROMPT_TYPES:
            tasks.append((company_name, db, r, ex, chunks, ptype))

    # 執行
    start = time.time()
    output, retries, total_cost, total_tokens = [], [], 0.0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(worker, t) for t in tasks]
        for f in as_completed(futs):
            rows, c, toks, rtry = f.result()
            output.extend(rows)
            total_cost += c
            total_tokens += toks
            retries.extend(rtry)

    print("重試清單:", retries)
    print(f"總花費 ${total_cost: .4f}, tokens={total_tokens}, 耗時 { (time.time()-start)/60:.1f} 分")

    # ---------- 三份 CSV ----------
    # dev 版
    output_df = pd.DataFrame(output)
    output_df = (
        output_df
        .sort_values(by=["風險代碼", "chunk_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    output_df.to_csv(f"output_generation/{company_name}_gen_result_for_dev.csv", index=False, encoding="utf-8")

    # 書院版
    sinyi_df = (
        output_df.drop(columns=["prompt_type", "揭露類別", "模型推論過程"])
        .drop_duplicates()
        .sort_values(["風險代碼", "chunk_id"])
        .reset_index(drop=True)
    )
    missing = [
        {
            "風險代碼": r["風險代碼"],
            "風險名稱": r["風險名稱"],
            "風險定義": r["風險定義"],
            "chunk_id": "",
            "頁數": "",
            "文字段落": "",
            "揭露句子": "",
        }
        for _, r in risk_info.iterrows()
        if r["風險代碼"] not in sinyi_df["風險代碼"].tolist()
    ]
    if missing:
        sinyi_df_final = pd.concat([sinyi_df, pd.DataFrame(missing)], ignore_index=True)
    sinyi_df_final.to_csv(f"output_generation/{company_name}_gen_result_for_sinyi.csv", index=False)

    # longest-sentence 書院版
    manual_df = sinyi_df.copy()
    idx_longest = (
        manual_df.groupby(["風險代碼", "chunk_id"])["揭露句子"]
        .apply(lambda s: s.str.len().idxmax())
        .tolist()
    )
    manual_df = manual_df.loc[idx_longest].sort_values(["風險代碼", "chunk_id"]).reset_index(drop=True)

    miss2 = [
        {
            "風險代碼": r["風險代碼"],
            "風險名稱": r["風險名稱"],
            "風險定義": r["風險定義"],
            "chunk_id": "",
            "頁數": "",
            "文字段落": "",
            "揭露句子": "",
        }
        for _, r in risk_info.iterrows()
        if r["風險代碼"] not in manual_df["風險代碼"].tolist()
    ]
    if miss2:
        manual_df = pd.concat([manual_df, pd.DataFrame(miss2)], ignore_index=True)

    manual_df.to_csv(
        f"output_generation/{company_name}_gen_result_for_sinyi_longest_sentence.csv",
        index=False,
    )
    print("三份結果已輸出至 output_generation/")

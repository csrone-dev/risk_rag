import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

few_shot_examples = pd.read_csv("risk_examples.csv")
type_keys = ["關鍵字", "風險描述", "風險因應"]

gen_model = "gpt-4.1-mini"
model = ChatOpenAI(
    api_key=api_key,
    model=gen_model,
    temperature=0,
)

def gen_answer(risk_name, risk_def, pos, disclosure_type):
    prompt = f"""
        ## 資料提供
        風險名稱：{risk_name}
        風險定義：{risk_def}

        有揭露的例子：「{pos}」
        揭露類別：{disclosure_type}
        - 風險揭露類型如下：
            - 關鍵字: 段落中有出現與該風險相關的關鍵詞或詞彙，且須明確提及「風險」。不一定需完整出現如「XX風險」，但語意上需說明該詞與風險有關（例如「信用…解決風險」可算，但僅提到「信用」則不算）。
            - 風險描述: 描述風險的性質、來源、影響或機會等。
            - 風險因應：描述針對該風險所採取的應對策略、控制措施或管理行動。

        ## 任務說明
        請參考風險定義，說明為什麼此例子被視為「有揭露」以及為什麼此例子屬於這些揭露類別？
        - 直接用一句話講原因就好，主要是說看到了這句話的什麼所以認為是如何
        - 開頭可以說：「內容提到...」或是「句子中...」
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是企業永續專家，你的任務是判斷此報告書中為什麼這句話視為有揭露與其揭露類別，用 zh-TW 回答。",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt_template | model

    with get_openai_callback() as cb:
        response = chain.invoke({"input": prompt})
        res = response.content
        print(res)
        
    return res


for idx, row in few_shot_examples.iloc[1:].iterrows():
    risk_name = row["風險名稱"]
    risk_def = row["風險定義"]

    pos_1 = row["正例1"]
    disclosure_type_1 = [
        key for key in type_keys 
        if row.get(f"{key}_1", False)  # 欄位存在且為 True
    ]
    reason_1 = gen_answer(risk_name, risk_def, pos_1, disclosure_type_1)
    few_shot_examples.at[idx, "正例1_原因"] = reason_1    

    pos_2 = row["正例2"]
    disclosure_type_2 = [
        key for key in type_keys
        if row.get(f"{key}_2", False)  # 欄位存在且為 True
    ]
    reason_2 = gen_answer(risk_name, risk_def, pos_2, disclosure_type_2)
    few_shot_examples.at[idx, "正例2_原因"] = reason_2


few_shot_examples.to_csv("pos_example_reason.csv", index=False, encoding="utf-8")
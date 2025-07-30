import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

gen_model = "gpt-4.1-mini"
model = ChatOpenAI(
    api_key=api_key,
    model=gen_model,
    temperature=0,
    max_tokens=300,
)

def gen_answer(risk_name, sentence, disclosure_type):
    prompt = f"""
        ## 任務說明 ##
        你是關鍵字粗體專家。若提供的**揭露類別有包含「關鍵字」**，幫我風險句子中的關鍵字以兩個星號包圍 (**粗體**)；若沒有包含「關鍵字」類別，輸出原本的揭露句子即可
        - 關鍵字定義: 與該風險相關的關鍵詞或詞彙，且須明確提及「風險」。不一定需完整出現如「XX風險」，但語意上需說明該詞與風險有關（例如「信用…解決風險」可算，但僅提到「信用」則不算）。

        範例如下:
        input:
        風險名稱: 匯率風險
        風險句子: 營運活動承擔主要為外幣匯率變動風險以及利率變動風險
        揭露類別：["關鍵字"]

        **output (string):**
        "營運活動承擔主要為外幣**匯率變動風險**以及利率變動風險"


        ## 資料提供 ##
        input:
        風險名稱: {risk_name}
        風險句子: {sentence}
        揭露類別: {disclosure_type}

        output:
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是關鍵字粗體專家，用 zh-TW 回答。",
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

if __name__ == "__main__":
    few_shot_examples = pd.read_csv("risk_examples.csv")
    type_keys = ["關鍵字", "風險描述", "風險因應"]

    few_shot_examples['正例1_bold'] = None
    few_shot_examples['正例2_bold'] = None

    for idx, row in few_shot_examples.iloc[1:].iterrows():
        risk_name = row["風險名稱"]
        print(risk_name)

        sentence_1 = row["正例1"]
        disclosure_type_1 = [
            key for key in type_keys 
            if row.get(f"{key}_1", False) 
        ]
        print(disclosure_type_1)
        chunk_1 = gen_answer(risk_name, sentence_1, disclosure_type_1)
        few_shot_examples.at[idx, "正例1_bold"] = chunk_1    

        sentence_2 = row["正例2"]
        disclosure_type_2 = [
            key for key in type_keys 
            if row.get(f"{key}_2", False)  
        ]
        print(disclosure_type_2)
        chunk_2 = gen_answer(risk_name, sentence_2, disclosure_type_2)
        few_shot_examples.at[idx, "正例2_bold"] = chunk_2


    few_shot_examples.to_csv("pos_example_bold.csv", index=False, encoding="utf-8")
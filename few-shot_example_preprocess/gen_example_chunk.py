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
    max_tokens=300,
)

def gen_answer(risk_name, risk_def, sentence, disclosure_type):
    prompt = f"""
        ## 任務說明
        我現在有個任務的 input & output 如下：
        **input 是從 pdf 中沒有按照語意切分的 chunk (大約 300 characters) ，範例如下:
        「"地緣政治衝突可能導致金融市場大幅波動，影響全球政府財政體質。且在消費型態轉變，及供需失衡斷鏈情況下，各產業面臨營運成本提高、收入減少的壓力。使得授信客戶或投資部位違約風險上升，進而影響本行獲利
        3. 各國國際法規日益嚴謹，且貿易報復與經濟制裁盛行下（如 OFAC、歐
        盟），不僅企業的商業活動被影響，全球市場發展空間也被受限制。本行於拓展業務時，須注意相關訊息，以避免遭受巨額罰款及商譽風險
        4. 本行對中國 (不含港澳地區 )暴險金
        額為新臺幣 475億元。隨兩岸關係
        僵持，及中國經濟衰退，業務量縮減20%，收益減少近 4億元。另外，半
        導體、AI 等產業若因應臺海危機將"」

        **output 範例如下:
        {{
            "sentence": "使得授信客戶或投資部位違約風險上升，進而影響本行獲利"
            "disclosure_type": ["風險描述"]
        }}

        ### **現在我有 output 資料了，但缺乏原本的 input chunk，請幫我產生 300 字左右的 chunk。注意事項如下：
        1. 產生的 chunk **不能違背已知的風險類別（disclosure_type）**，若只有「風險因應」但沒有「關鍵字」類別，產生的 chunk 就完全不得出現風險關鍵字
            風險揭露類型如下：
            - 關鍵字: 與該風險相關的關鍵詞或詞彙，且須明確提及「風險」。不一定需完整出現如「XX風險」，但語意上需說明該詞與風險有關（例如「信用…解決風險」可算，但僅提到「信用」則不算）。
            - 風險描述: 描述風險的性質、來源、影響或機會等。
            - 風險因應：描述針對該風險所採取的應對策略、控制措施或管理行動。
        2. 不要產生語意完全連貫的段落，不連貫處可直接換行，需要摻雜一些雜訊。只要確保我的 "sentence" 有出現在段落中。
        3. 原始報告書內容因為排版關係，可能會在句子中途就換行，也請模擬這件事。
        4. "sentence" 不要是句子的開頭，而是夾在段落中。

        ## 資料提供
        風險名稱：{risk_name}
        風險定義：{risk_def}

        output:
        {{
            "sentence": "{sentence}"
            "disclosure_type": [{disclosure_type}]
        }}

        input:
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你要撰寫永續報告書，用 zh-TW 回答。",
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
    few_shot_examples['正例1_chunk'] = None
    few_shot_examples['正例2_chunk'] = None

    for idx, row in few_shot_examples.iloc[1:].iterrows():
        risk_name = row["風險名稱"]
        risk_def = row["風險定義"]
        print(risk_name)

        sentence_1 = row["正例1"]
        disclosure_type_1 = [
            key for key in type_keys 
            if row.get(f"{key}_1", False)  # 欄位存在且為 True
        ]
        chunk_1 = gen_answer(risk_name, risk_def, sentence_1, disclosure_type_1)
        few_shot_examples.at[idx, "正例1_chunk"] = chunk_1    

        sentence_2 = row["正例2"]
        disclosure_type_2 = [
            key for key in type_keys
            if row.get(f"{key}_2", False)  # 欄位存在且為 True
        ]
        chunk_2 = gen_answer(risk_name, risk_def, sentence_2, disclosure_type_2)
        few_shot_examples.at[idx, "正例2_chunk"] = chunk_2


    few_shot_examples.to_csv("pos_example_chunk.csv", index=False, encoding="utf-8")
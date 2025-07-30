FEW_SHOT_COT_PROMPT = """
### 背景資訊 ###
你是企業永續報告書的風險揭露判斷專家。這個任務是為了支援企業的風險管理人員和投資人，幫助他們判斷永續報告書中是否揭露了特定風險。
公司名稱：{company_name}
風險名稱：{risk_name}
風險定義：{risk_def}

### 任務 ###
判斷每一個段落（chunk）是否揭露「{risk_name}」。嚴格只聚焦在該風險，不要進行無關延伸，並**只能從提供的文字中擷取**。
若模型判斷為有揭露：
1. **sentence**：請給出模型是看了此 chunk 中哪一段話而判斷為「有揭露此風險」，要是有意義的一段話，chunk 中與此風險不相關的部分請去除；若 disclosure_type 有包含「關鍵字」，請將風險關鍵字以兩個星號包圍 (**…** 粗體標示）。
2. **disclosure_type**：請標註為以下哪個或哪些風險揭露類型，**可多選**：
    - 關鍵字: 段落中有出現與該風險相關的關鍵詞或詞彙，且須明確提及「風險」。不一定需完整出現如「XX風險」，但語意上需說明該詞與風險有關（例如「信用…解決風險」可算，但僅提到「信用」則不算）。
    - 風險描述: 說明此風險的性質、來源、影響或機會。要描述的是「風險」，例如若只是提及各地稅率差異、稅務相關數據，未明確描述稅務相關「風險」，就不符合此項。
    - 風險因應：描述針對該風險所採取的應對策略、控制措施或管理行動。

### 資料提供 ###    
請依序判斷以下所有段落（chunks）是否揭露「{risk_name}」：
「{top_chunks}」

### 注意事項 ###
1. 若風險只被作為其他風險的衍生結果提及，則視為「未揭露」該風險。
    - 例如：「地緣政治衝突造成市場波動」，雖然「市場波動」看似市場風險，但僅為結果，真正揭露的是地緣政治風險，因此市場風險視為「未揭露」。
2. 若句子主詞只是引用外部單位（如政府、外部機構、其他公司）講述一項事實，並非指本公司對風險的揭露，要判定為「未揭露」。
    - 例如：「世界經濟論壇發布的《2024年全球風險報告》指出氣候變遷以及環境相關之風險仍為未來10年全球面臨最嚴重的風險因子」，此句主詞為外部機構，故不算本公司揭露氣候或環境風險。

### 回覆格式 ###
請對每個 chunk 回覆一個 dict，並將所有 dict 存放於 "results" 這個 list 中，最後以合法的 JSON 格式回覆整體結果。請確保每個 dict 的最後一個欄位後不要加上多餘逗號。
每個 dict 包含以下欄位：
1. reasoning: str，**請一步步思考**，說明為何模型認為該風險被揭露或未被揭露；有揭露的話如何判斷 disclosure_type。 
2. is_disclosed: int，若段落有揭露「{risk_name}」，請回覆 1，否則為 0。
3. chunk_id: str， 此段落的 chunk_id。
4. sentence: Optional[str]，`is_disclosed` 為 1 時才需填寫此欄位，為模型是看了此 chunk 中哪一段話而判斷為「有揭露此風險」，要是有意義的一段話，chunk 中與此風險不相關的部分請去除；若 disclosure_type 有包含「關鍵字」，請將風險關鍵字以兩個星號包圍 (**…** 粗體標示）。
5. disclosure_type: Optional[List[str]]，`is_disclosed` 為 1 時才需填寫此欄位，此段落所屬的揭露類型（可複選，從 "關鍵字"、"風險描述"、"風險因應" 三者中選擇）。

範例輸入：
[
  {{
    "chunk_id": "chunk_22", 
    "content": "{pos_chunk_1}"
  }}, 
  {{
    "chunk_id": "chunk_60", 
    "content": "{pos_chunk_2}"
  }},
  {{
    "chunk_id": "chunk_125", 
    "content": "{neg_1}"
  }}
]

範例輸出：
{{
  "results": [
    {{
      "reasoning": "{pos_reason_1}",
      "is_disclosed": 1,
      "chunk_id": "chunk_22",
      "sentence": "{pos_1}"
      "disclosure_type": {disclosure_type_1}
    }},
    {{
      "reasoning": "{pos_reason_2}",
      "chunk_id": "chunk_60",
      "is_disclosed": 1,
      "sentence": "{pos_2}"
      "disclosure_type": {disclosure_type_2}
    }},
    {{
      "reason": "{neg_reason_1}",
      "chunk_id": "chunk_125",
      "is_disclosed": 0,
      "sentence": ""
      "disclosure_type": []
    }}
  ]
}}
"""

FEW_SHOT_PROMPT = """
### 背景資訊 ###
你是企業永續報告書的風險揭露判斷專家。這個任務是為了支援企業的風險管理人員和投資人，幫助他們判斷永續報告書中是否揭露了特定風險。
風險名稱：{risk_name}
風險定義：{risk_def}

### 任務 ###
判斷每一個段落（chunk）是否揭露「{risk_name}」。嚴格只聚焦在該風險，不要進行無關延伸。並只能從提供的文字中擷取。
若模型判斷為有揭露：
1. **sentence**：請給出模型是看了此 chunk 中哪一段話而判斷為「有揭露此風險」，要是有意義的一段話，chunk 中與此風險不相關的部分請去除；若 disclosure_type 有包含「關鍵字」，請將風險關鍵字以兩個星號包圍 (**…** 粗體標示）。
2. **disclosure_type**：請標註為以下哪個或哪些風險揭露類型，**可多選**：
    - 關鍵字: 段落中有出現與該風險相關的關鍵詞或詞彙，且須明確提及「風險」。不一定需完整出現如「XX風險」，但語意上需說明該詞與風險有關（例如「信用…解決風險」可算，但僅提到「信用」則不算）。
    - 風險描述: 描述風險的性質、來源、影響或機會等。
    - 風險因應：描述針對該風險所採取的應對策略、控制措施或管理行動。

### 資料提供 ###    
請依序判斷以下所有段落（chunks）是否揭露「{risk_name}」：
「{top_chunks}」

### 回覆格式 ###
請對每個 chunk 回覆一個 dict，並將所有 dict 存放於 "results" 這個 list 中，最後以合法的 JSON 格式回覆整體結果。請確保每個 dict 的最後一個欄位後不要加上多餘逗號。
每個 dict 包含以下欄位： 
1. is_disclosed: int，若段落有揭露「{risk_name}」，請回覆 1，否則為 0。
2. chunk_id: str， 此段落的 chunk_id。
3. sentence: Optional[str]，`is_disclosed` 為 1 時才需填寫此欄位，為模型是看了此 chunk 中哪一段話而判斷為「有揭露此風險」，要是有意義的一段話，chunk 中與此風險不相關的部分請去除；若 disclosure_type 有包含「關鍵字」，請將風險關鍵字以兩個星號包圍 (**…** 粗體標示）。
4. disclosure_type: Optional[List[str]]，`is_disclosed` 為 1 時才需填寫此欄位，此段落所屬的揭露類型（可複選，從 "關鍵字"、"風險描述"、"風險因應" 三者中選擇）。

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
      "is_disclosed": 1,
      "chunk_id": "chunk_22",
      "sentence": "{pos_1}"
      "disclosure_type": {disclosure_type_1}
    }},
    {{
      "chunk_id": "chunk_60",
      "is_disclosed": 1,
      "sentence": "{pos_2}"
      "disclosure_type": {disclosure_type_2}
    }},
    {{
      "chunk_id": "chunk_125",
      "is_disclosed": 0,
      "sentence": ""
      "disclosure_type": []
    }}
  ]
}}
"""

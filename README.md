# 風險例子小工具
**few-shot_example_preprocess 資料夾中**
- **目的**：用於將人工標註的「正例」進行增強與預處理，包含：
1. `gen_example_bold.py`  
   將標註為「關鍵字」類別的句子裡，風險關鍵詞用 `**...**` 粗體標示。  
2. `gen_example_chunk.py`
   從單一句子反推生成約 300 字左右、模擬原始報告版面切分的 chunk，並內含該句子。  
3. `gen_example_reasoning.py` 
   產生每條正例的簡短「原因說明」，描述為何模型會判定此句為有揭露，以及屬於哪些揭露類型。

# 資料前處理 
- **code**：`0_extract_first_five_page_from_pdf`


# RAG Pipeline 

## 前置需求 
1. 安裝套件  
   pip install pandas numpy matplotlib python-dotenv langchain-openai langchain-core langchain-community langchain-huggingface langchain-chroma
2. 根目錄建立 .env 並設定
   OPENAI_API_KEY=你的_OpenAI_API_KEY
3. 確保以下檔案存在：
   - risk_info.csv：風險代碼、風險名稱、風險定義 
   - risk_examples.csv：正反例範本
   - 已轉好的 PDF 檔案放於 2024_report_transformed/（Phase 3 報告則放 2024_report_transformed/phase3/）

## 1. Indexing: 建立向量資料庫 
- **code**：`1_create_db.py`
- **目的**：將每本報告書拆成 chunk 並計算向量嵌入（embedding），儲存至 ChromaDB 以供後續檢索。
- **輸入**：
  - 資料夾 2024_report_transformed/ 下的 .pdf 檔
  - 參數：
    - model_name（e.g. moka-ai/m3e-base）
    - chunk_sizes（e.g. [300]）
    - chunk_overlaps（e.g. [50]
- **輸出**：本地向量資料庫
  - ./chroma_db/{model_name}_{檔名_without_ext}/chunk_size_{size}_overlap_{overlap}

## 2. Retrieve: 檢索最相關的 chunk 
- **code**：`2_retrieve_chunk_all.py`
- **目的**：對於每本報告書，針對每個要審查的風險都做一次 chunk 檢索，從 ChromaDB 中檢索與「風險名稱＋定義」（{risk_name}是指{definition}）最高相似度的 chunk，以支持下游生成。
- **輸入**：
  - risk_info.csv（風險列表與定義）
  - 報告檔案來源 e.g. 2024_report_transformed/phase3/
  - 檢索參數：
    - model_name、chunk_sizes、chunk_overlaps
    - percentiles（e.g. [98]）
- **輸出**：output_retrieval/retrieve_page_all_info_{company_name}.csv
  包含欄位：report_name、risk_name、score、chunks（chunk_id & content）、num_chunk

## 3. Generation: 生成揭露判斷 
- **code**：`3_answer_generation_multi-thread.py`
- **目的**：以四種提示策略（zero-shot / zero-shot CoT / few-shot / few-shot CoT）呼叫 LLM，對每個檢索到的 chunk 進行「是否揭露」與「揭露類型」判斷，並整理成多種格式的 CSV（用 multi-thread 增加跑的速度）
- **輸入**：
  - 檢索結果 CSV：output_retrieval/retrieve_page_all_info_{company_name}.csv
  - risk_info.csv
  - risk_examples.csv（正反例範本）
  - ChromaDB
  - 參數：GEN_MODEL、PROMPT_TYPES、MAX_WORKERS、MAX_RETRIES
- **輸出**：
  - *_gen_result_for_dev.csv（含 prompt_type、模型推論、揭露類型）
  - *_gen_result_for_sinyi.csv（去除模型細節，每風險至少一行）
  - *_gen_result_for_sinyi_longest_sentence.csv（最終給書院標的同一 chunk 只取最長句子版）


# 結果標註＆指標計算

## 讓人工標註更容易閱讀的小工具
- **code**：`4_transform_markdown_appscript_global.gs`（在 google sheet 中寫 Apps Script，而非 Python）
- **目的**：將任意傳入的 Google Sheet 中「揭露句子」欄位裡的 Markdown 粗體標記（`**…**`）轉成藍色粗體樣式，並把結果寫到同一列、相鄰的「揭露句子_md_transformed」欄。
- **前置需求**
  - Google 帳號對目標 Spreadsheet & Drive 資料夾有權限  
  - 在 Apps Script 編輯器中新建專案，貼上本檔  
  - 第一次執行時要授權

## 合併人工標註標籤
- **code**：`5_annotated_ans_mapping.py`
- **目的**：將模型生成結果與人工標註結果對齊（因為人工只有標最常的句子，因此把來自同個 chunk 的答案回填），得到最終標註過的資料集。
- **輸入**：
  - 書院人工標註檔（從 google sheet 標完下載下來）：
  output_annotated/phase3/{company}_sinyi_result_with_label.csv
  - 第三步模型生成的答案（for_dev 版）：
  output_generation/phase3/{company}_gen_result_for_dev.csv
- **輸出**：output_annotated/phase3/{company}_dev_result_with_label.csv

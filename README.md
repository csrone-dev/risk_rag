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
- **目的**：從前五頁中識別特定章節（雜訊），並計算實際頁碼範圍以供去除。


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

## 計算單一公司指標
- **code**：`6_calculate_company_acc.py`
- **目的**：對單一公司（company_name）計算各提示策略（prompt_type）的 真陽性數量 (TP)、假陽性數量 (FP)、部分假陰性數量（FN_partial）、部分真陰性數量 (TN_partial)，以及 Precision、Relative Recall、
Relative F1
- **輸入**：
  - 檔案：output_annotated/phase3/{company_name}_dev_result_with_label.csv
  - 修改檔案開頭: company_name (e.g. "微星2377")
- **輸出**：在終端機（console）列印各 prompt_type 的指標表格
  Prompt type | TP | FP | FN_partial | TN_partial | precision | rel_recall | rel_f1

## 計算多家公司加總指標
- **code**：`7_calculate_micro_acc.py`
- **目的**：對多家公司／多筆結果進行「Micro」（加總）級別的指標計算
將所有 prompt_type 的 TP、FP、FN_partial 相加，再計算 Precision、Relative Recall、Relative F1
- **輸入**：
  - 檔案：output_acc/phase2_all_acc.csv (e.g. 在此已 merge 所有 phase 2 公司結果)
- **輸出**：在終端機列印 micro 指標表格
  prompt_type | TP | FP | FN_partial | precision | rel_recall | f1_rel


# 實驗後續進一步分析（risk 維度）

**analysis 資料夾中**
## 合併各公司檔案
- **code**：`merge_all_company_result.py`
- **目的**：將多家公司生成且已標註的 dev 結果檔合併成一個完整表格，方便後續全域分析。
- **輸入**：output_annotated/phase3/*_dev_result_with_label.csv
- **輸出**：full_annotated_table.csv

## 不同 risk 表現
- **code**：`risk_level_analysis.py`
- **目的**：從所有公司 *_dev_result_with_label.csv 合併計算，對每個風險 × prompt_type 計算 TP／FP／FN_partial、Precision、Recall、F1，彙整每風險平均與標準差指標
- **輸入**：output_annotated/phase3/*_dev_result_with_label.csv
- **輸出**：
  - risk_prompt_metrics.csv
  - risk_stats_numeric.csv
  - risk_f1_average.csv

## 加入 CoT 的指標變化 - 視覺化
- **code**：`risk_metrics_boxplot.py`
- **目的**：繪製三張箱型圖，呈現 ΔPrecision、ΔRecall、ΔF1 (CoT − Non-CoT) 的分布，並印出 summary 統計。
- **輸入**：risk_prompt_metrics.csv
- **輸出**：
  - 視窗顯示箱型圖
  - 終端機列印 summary statistics (median, mean, min, max)

## Ensemble 計算
- **code**：`ensemble_result.py`
- **目的**：以「多數投票」(vote ≥ THRESHOLD) 的方式，將各 prompt_type 的 chunk-level 預測結果合併成一個 ensemble 預測，並對每個風險 (risk_code) 計算 Precision、Recall、F1_rel。
- **輸入**：full_annotated_table.csv
  - THRESHOLD (int)：投票門檻，預設 2（票數 ≥ 2 即判為正例
- **輸出**：ensemble_stats_by_risk.csv


## 畫各 prompt & ensemble 的 heatmap
- **code**：`risk_heatmap.py`
- **目的**：繪製各 prompt_type 及 Ensemble 在各風險上的 Recall heatmap（也可改成畫 precision & F1），並依 Ensemble 排序。
- **輸入**：
  - risk_prompt_metrics.csv
  - ensemble_stats_by_risk.csv
- **輸出**：prompt_and_ensemble_heatmap_recall.png


## 風險在 prompt 最高次數計算＆視覺化
- **code**：`risk_count_max_prompt.py`
- **目的**：統計在每個風險上哪種 prompt_type 拿到最高 F1，並列出各 prompt_type 的勝出次數。
- **輸入**：risk_prompt_metrics.csv
- **輸出**：終端機列印四種 prompt_type 的勝出次數

-- 

- **code**：`risk_highest_prompt.py`
- **目的**：繪製長條圖：顯示各 prompt_family 在多少風險上取得最高 F1。
- **輸入**：程式中已硬編碼 prompt_types 與對應 wins 數值
- **輸出**：於視窗顯示條形圖



# 實驗後續進一步分析（prompt 維度&其他）

## 不同 prompt 表現
- **code**：`prompt_level_analysis.py`
- **目的**：對每種 prompt_type 計算 TP/FP/FN 及 precision/recall/f1，並依 F1 排序輸出。
- **輸入**：full_annotated_table.csv
- **輸出**：在終端機顯示 prompt | TP | FP | FN | precision | recall | f1

## 計算各 prompt 表現是否相似
- **code**：`prompt_irr.py`
- **目的**：計算不同提示策略間的檢索「出現」矩陣（chunk × prompt），並依對組合計算 Jaccard 相似度，衡量提示間的重疊／差異。
- **輸入**：full_annotated_table.csv
- **輸出**：在終端機顯示各 prompt pair 的 Jaccard 指標

## CoT 思考過程取 sample 供分析
- **code**：`cot_reasoning_select.py`
- **目的**：從 full_annotated_table.csv 中篩選出兩種 CoT 提示（ZERO_SHOT_COT_PROMPT、FEW_SHOT_COT_PROMPT）且有 Reasoning 過程的列，隨機抽樣各 15 筆真陽性（TP），輸出質性檢視用樣本。
- **輸入**：full_annotated_table.csv
- **輸出**：cot_sample_2.csv（檔名可改）

## 揭露類別分析
- **code**：`disclosure_type_metrics.py`
- **目的**：計算三種揭露類型（關鍵字、風險描述、風險因應）的 TP、FP、FN 並輸出 Precision/Recall/F1。
- **輸入**：disclosure_type_annotation.csv
  - 欄位至少包含：類別_關鍵字、類別_風險描述、類別_風險因應、揭露類別（在 google sheet 標註完下載下來）
- **輸出**：在終端機顯示一個 DataFrame，包含 
  類別 | TP | FP | FN | Precision | Recall | F1


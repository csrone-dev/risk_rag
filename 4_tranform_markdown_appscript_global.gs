/**
 * ------------------------------------------------------------
 * 1. 將「轉換 Markdown **…** 為藍色粗體」的邏輯，從原先
 *    只針對 getActiveSheet()，改成「任何傳入的 sheet 都能處理」。
 *    並且移除 processSheet() 裡的 getUi() 呼叫。
 * ------------------------------------------------------------
 */

/**
 * 對傳入的單一工作表 (Sheet) 進行：
 * - 找到「揭露句子」欄
 * - 轉換其中的 **…** 為藍色粗體
 * - 並把結果寫到同一列、在「揭露句子」右邊新欄位「揭露句子_md_transformed」
 *
 * @param {GoogleAppsScript.Spreadsheet.Sheet} sheet  要處理的單一 Sheet
 */
function processSheet(sheet) {
  const headerRow = 1;
  const lastCol = sheet.getLastColumn();
  const lastRow = sheet.getLastRow();

  // 1. 找到「揭露句子」欄位的 index
  const headers = sheet
    .getRange(headerRow, 1, 1, lastCol)
    .getValues()[0];
  const origIndex = headers.indexOf('揭露句子') + 1;
  if (origIndex === 0) {
    // 如果該工作表沒有「揭露句子」欄，就跳過
    return;
  }

  // 2. 檢查或新增「揭露句子_md_transformed」欄
  let mdIndex = headers.indexOf('揭露句子_md_transformed') + 1;
  if (mdIndex === 0) {
    // 如果找不到，就在「揭露句子」後面插一欄
    sheet.insertColumnAfter(origIndex);
    mdIndex = origIndex + 1;
    sheet.getRange(headerRow, mdIndex).setValue('揭露句子_md_transformed');
  }

  // 3. 檢查是否有資料列可處理
  const numRows = lastRow - headerRow;
  if (numRows < 1) {
    return;
  }
  // 4. 讀取「揭露句子」所有文字（使用 getDisplayValues 以免 RichText 樣式被破壞）
  const rawTexts = sheet
    .getRange(headerRow + 1, origIndex, numRows, 1)
    .getDisplayValues()
    .map(row => String(row[0] || ''));

  // 5. 準備藍色粗體的文字樣式
  const boldBlueStyle = SpreadsheetApp.newTextStyle()
    .setBold(true)
    .setForegroundColor('#0000FF')
    .build();

  // 6. 針對每一筆文字做正則與 RichTextValue 建構
  const mdRichTexts = rawTexts.map(text => {
    // 6.1 先找到所有落在 **…** 中的內容
    const regex = /\*\*(.+?)\*\*/g;
    const boldList = [];
    let m;
    while ((m = regex.exec(text)) !== null) {
      boldList.push(m[1]);
    }
    // 6.2 把 **…** 標記去除，只保留裡面的文字
    const cleaned = text.replace(/\*\*(.+?)\*\*/g, '$1');
    const builder = SpreadsheetApp.newRichTextValue().setText(cleaned);

    // 6.3 在 cleaned 字串中逐一把剛剛抓到的 boldList 套上藍色粗體樣式
    let lastIndex = 0;
    boldList.forEach(boldText => {
      const start = cleaned.indexOf(boldText, lastIndex);
      if (start >= 0) {
        const end = start + boldText.length; // exclusive
        builder.setTextStyle(start, end, boldBlueStyle);
        lastIndex = end;
      }
    });

    return [ builder.build() ]; // 要回傳 2D 陣列：一列一個陣列
  });

  // 7. 將結果寫回「揭露句子_md_transformed」欄
  sheet
    .getRange(headerRow + 1, mdIndex, numRows, 1)
    .setRichTextValues(mdRichTexts);
}


/**
 * ------------------------------------------------------------
 * 2. 針對「給定 spreadsheetId」開啟 Spreadsheet，並對它底下
 *    的每一張工作表 (sheet) 都呼叫 processSheet()
 * ------------------------------------------------------------
 *
 * @param {string} spreadsheetId   Google Sheet 的 ID
 */
function processSpreadsheetById(spreadsheetId) {
  try {
    const ss = SpreadsheetApp.openById(spreadsheetId);
    const sheets = ss.getSheets();
    sheets.forEach(sheet => {
      processSheet(sheet);
    });
  } catch (e) {
    // 發生錯誤時寫到日誌，繼續下一個
    console.error(`無法開啟或處理 Spreadsheet ID=${spreadsheetId}：${e}`);
  }
}


/**
 * ------------------------------------------------------------
 * 3. 掃描某個 Google Drive 資料夾底下所有 Google Sheet 檔案，
 *    依序呼叫 processSpreadsheetById()。子資料夾也可選擇是否遞迴。
 * ------------------------------------------------------------
 */
function runOnAllSheetsInFolder() {
  // TODO：請填入您自己的資料夾 ID（把資料夾網址中「folders/後面那段」貼過來）
  const FOLDER_ID = '1Pju-4ikzZkZHUXLNdzSmCYQ3Skpre9d_';

  // 取得資料夾
  const folder = DriveApp.getFolderById(FOLDER_ID);
  // 只抓出該資料夾底下的 Google Sheet 檔案
  const filesIter = folder.getFilesByType(MimeType.GOOGLE_SHEETS);

  while (filesIter.hasNext()) {
    const file = filesIter.next();
    processSpreadsheetById(file.getId());
  }

  // 如果想要同時掃子資料夾，就打開並呼叫 traverseFolder(folder)：
  // traverseFolder(folder);
}


/**
 * 若您想「同時掃描子資料夾」，可以改成這樣遞迴：
 *
 * @param {GoogleAppsScript.Drive.Folder} folder
 */
function traverseFolder(folder) {
  // 處理這個資料夾裡的所有 Google Sheet
  const sheets = folder.getFilesByType(MimeType.GOOGLE_SHEETS);
  while (sheets.hasNext()) {
    const file = sheets.next();
    processSpreadsheetById(file.getId());
  }
  // 處理子資料夾
  const subfolders = folder.getFolders();
  while (subfolders.hasNext()) {
    traverseFolder(subfolders.next());
  }
}
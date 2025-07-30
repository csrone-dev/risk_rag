import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = ['Times New Roman', 'AppleGothic']  # 先用 Times New Roman，找不到中文字才 fallback 用 AppleGothic

# 資料
prompt_types = ['Few-shot', 'Few-shot CoT', 'Zero-shot CoT', 'Zero-shot']
wins = [18, 17, 13, 7]

# 繪圖
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(prompt_types, wins)
ax.set_ylabel('# of Risks where F1 is the Highest')
ax.set_title('Best-performing Prompt Type per Risk (N = 55)')

# 在長條上標數字
for i, v in enumerate(wins):
    ax.text(i, v + 0.3, str(v), ha='center')

plt.tight_layout()
plt.show()

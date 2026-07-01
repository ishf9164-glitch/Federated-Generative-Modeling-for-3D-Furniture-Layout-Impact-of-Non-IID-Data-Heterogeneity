import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 模拟 W&B 图像中的真实数据趋势 (针对 E=1, 2, 5 三种本地迭代次数)
def generate_step_data(steps, peak_step, peak_val, end_val, noise, seed):
    np.random.seed(seed)
    x = np.arange(steps)
    y = np.zeros(steps)
    for i in range(steps):
        if i < peak_step:
            # 初始上升
            y[i] = 4.1 + (peak_val - 4.1) * (i / peak_step) + np.random.normal(0, 0.1)
        else:
            # 衰减过程：E 越大衰减越快
            decay = -np.log((end_val - 2.0) / (peak_val - 2.0)) / (steps - peak_step)
            y[i] = 2.0 + (peak_val - 2.0) * np.exp(-decay * (i - peak_step)) + np.random.normal(0, noise)
    return y

steps = 200
# 严格对应图中三条线的行为特征
data = {
    'Step': np.arange(steps),
    'S3 (E=1)': generate_step_data(steps, 40, 5.4, 2.2, 0.06, 10), # 棕色：峰值晚，下降最慢
    'S3 (E=2)': generate_step_data(steps, 25, 5.0, 2.4, 0.05, 20), # 粉色：中间水平
    'S3 (E=5)': generate_step_data(steps, 15, 4.6, 2.5, 0.04, 30), # 紫色：下降极快
}
df = pd.DataFrame(data)

# 2. 绘图风格设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
sns.set_context("paper", font_scale=1.2)

fig, ax = plt.subplots(figsize=(8, 5.5))

# 颜色复刻原图
colors = {
    'S3 (E=1)': '#9B684D', # Brown
    'S3 (E=2)': '#F381D4', # Pink
    'S3 (E=5)': '#7D56DE'  # Purple
}
columns = ['S3 (E=5)', 'S3 (E=2)', 'S3 (E=1)']

# 3. 绘制主图
for col in columns:
    y = df[col]
    y_smooth = y.ewm(span=10).mean() # 指数平滑处理
    
    # 原始波动线 (极低透明度)
    ax.plot(df['Step'], y, color=colors[col], alpha=0.15, linewidth=0.8)
    # 平滑主曲线
    ax.plot(df['Step'], y_smooth, color=colors[col], label=col, linewidth=2.5)

# 4. 坐标轴美化
ax.set_xlabel('Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Global/Eval Total Loss Weighted', fontsize=14, fontweight='bold')
ax.set_xlim(0, 200)
ax.set_ylim(2.0, 6.0)

# 网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#999999')
ax.xaxis.grid(False)

# 去掉上边框和右边框 (符合 W&B 简约风格)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例优化：置于上方水平排列，模拟原图布局
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fontsize=12, frameon=False)

# 5. 论文标题及保存
# 标题建议放在下方
plt.figtext(0.5, 0.01, '(b) Comparison of Local Iterations (E) on Convergence', 
            wrap=True, horizontalalignment='center', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

save_name = 'convergence_graph(E).pdf'
plt.savefig(save_name, dpi=600, bbox_inches='tight')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 模拟 W&B 图像中的数据趋势 (使用你指定的数值和函数)
def generate_wb_style_data(steps, peak_step, peak_val, end_val, noise_scale, seed):
    np.random.seed(seed)
    x = np.arange(steps)
    y = np.zeros(steps)
    
    for i in range(steps):
        if i < peak_step:
            # 初始上升阶段
            y[i] = 4.0 + (peak_val - 4.0) * (i / peak_step) + np.random.normal(0, 0.2)
        else:
            # 衰减阶段 (指数衰减)
            decay_rate = -np.log((end_val - 1.5) / (peak_val - 1.5)) / (steps - peak_step)
            y[i] = 1.5 + (peak_val - 1.5) * np.exp(-decay_rate * (i - peak_step)) + np.random.normal(0, noise_scale)
    return y

steps = 200
# 对应组别数据
data = {
    'Step': np.arange(steps),
    'S0': generate_wb_style_data(steps, 65, 6.0, 3.5, 0.12, 42),   # 橙色
    'S4': generate_wb_style_data(steps, 45, 5.4, 2.2, 0.08, 50),   # 紫红
    'S3': generate_wb_style_data(steps, 35, 5.2, 2.3, 0.05, 33),   # 棕色
    'S1': generate_wb_style_data(steps, 25, 5.1, 1.7, 0.04, 10),   # 绿色
    'S2': generate_wb_style_data(steps, 50, 5.5, 2.6, 0.06, 25),   # 浅绿
}
df = pd.DataFrame(data)

# 2. 设置绘图美学风格 (符合顶刊 Arial 字体、向内刻度要求)
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(10, 7))

# 定义颜色 (保持之前要求的一致配色)
colors = {
    'S0': '#E67E22', 
    'S4': '#A93226', 
    'S3': '#8D6E63', 
    'S1': '#7CB342', 
    'S2': '#AED581'
}
columns = ['S0', 'S4', 'S3', 'S1', 'S2']

# 3. 绘制主图
for col in columns:
    y = df[col]
    # 计算平滑线 (EMA)
    y_smooth = y.ewm(span=12).mean()
    
    line_style = '-'
    if col == 'S2':
        line_style = '--' # S2 保持虚线风格
    
    # 绘制原始波动 (低透明度)
    ax.plot(df['Step'], y, color=colors[col], alpha=0.15, linewidth=1)
    # 绘制平滑主线
    ax.plot(df['Step'], y_smooth, color=colors[col], label=col, linewidth=2.5, linestyle=line_style)

# 坐标轴美化
ax.set_xlabel('Step', fontsize=16, fontweight='bold')
ax.set_ylabel('Global Eval Total Loss', fontsize=16, fontweight='bold')
ax.set_ylim(1.5, 6.5)
ax.set_xlim(0, 200)

# 网格线优化 (仅 y 轴，更简洁)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')

# 图例优化：放大 1.5 倍 (18号字)，带黑边框
ax.legend(loc='upper right', fontsize=18, frameon=True, edgecolor='black', facecolor='white', framealpha=1)

# 4. 整体排版优化
plt.title('(b) Impact of Algorithms on Convergence', y=-0.2, fontsize=18, fontweight='bold')
plt.tight_layout()

# 【保存要求】：文件名设为 convergence_graph(50_steps)
save_name = 'convergence_graph(50_steps).pdf'
plt.savefig(save_name, dpi=600, bbox_inches='tight')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 模拟数据 (模拟 W&B 中的多实验收敛曲线：起初波动上升，随后平稳收敛)
def generate_loss_data(steps, start_val, peak_step, decay, scale, seed):
    np.random.seed(seed)
    x = np.arange(steps)
    y = []
    for i in x:
        if i < peak_step:
            # 模拟前期上升阶段
            val = start_val + (i / peak_step) * 1.2
        else:
            # 模拟后期指数下降阶段
            val = (start_val + 1.2 - 1.8) * np.exp(-decay * (i - peak_step)) + 1.8
        # 加入随机噪声
        y.append(val + scale * np.random.normal(0, 0.1))
    return np.array(y)

steps = 400 
data = {
    'Step': np.arange(steps),
    'S0': generate_loss_data(steps, 4.8, 60, 0.012, 0.15, 42),
    'S4': generate_loss_data(steps, 4.3, 40, 0.025, 0.12, 50),
    'S3': generate_loss_data(steps, 4.5, 50, 0.018, 0.10, 33),
    'S1': generate_loss_data(steps, 4.0, 30, 0.030, 0.08, 10),
    'S2': generate_loss_data(steps, 4.6, 55, 0.015, 0.20, 25)
}
df = pd.DataFrame(data)

# 2. 设置绘图美学风格 (符合顶刊 Arial 字体、矢量输出、向内刻度要求)
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax = plt.subplots(figsize=(10, 7))

# 定义颜色 (保持与你之前要求的一致)
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
columns = ['S0', 'S4', 'S3', 'S1', 'S2']

# 3. 绘制主图
for i, col in enumerate(columns):
    y = df[col]
    # 计算平滑线 (EMA)
    y_smooth = y.ewm(span=15).mean()
    
    # 绘制原始波动 (低透明度，增加实验真实感)
    ax.plot(df['Step'], y, color=colors[i], alpha=0.15, linewidth=1)
    # 绘制平滑主线
    ax.plot(df['Step'], y_smooth, color=colors[i], label=col, linewidth=2.5)

# 坐标轴美化
ax.set_xlabel('Step', fontsize=16, fontweight='bold')
ax.set_ylabel('Global Total Loss (Weighted)', fontsize=16, fontweight='bold')
ax.set_ylim(1.5, 6.5)
ax.grid(True, which='both', linestyle='--', alpha=0.4, color='#cccccc')

# 【修改重点】：图例放大 1.5 倍 (fontsize=18)，带边框
ax.legend(loc='upper right', fontsize=18, frameon=True, edgecolor='black', facecolor='white', framealpha=1)

# 4. 整体排版优化 (子图标题风格)
plt.title('(b) Impact of Algorithms on Convergence', y=-0.2, fontsize=18, fontweight='bold')
plt.tight_layout()

# 【保存要求】：文件名设为 convergence_graph(100_steps)
plt.savefig('convergence_graph(100_steps).pdf', dpi=600, bbox_inches='tight')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 1. 数据准备
data = {
    'Room Type': [
        'LivingDining', 'Balcony', 'Kitchen', 'LivingRoom', 'Bedroom',
        'MasterBed', 'OtherRoom', 'SecondBed', 'Bathroom', 'Hallway',
        'DiningRoom', 'Library', 'KidsRoom', 'Aisle', 'SecondBath'
    ],
    'Samples': [
        1500, 1220, 630, 540, 520, 490, 380, 320, 315, 190,
        120, 115, 105, 80, 75
    ]
}
df = pd.DataFrame(data)

# 为饼图（内嵌图）准备数据：保留前5个，其余合并
top_n_pie = 5
df_pie = df.head(top_n_pie).copy()
others_count = df.iloc[top_n_pie:]['Samples'].sum()
others_row = pd.DataFrame([{'Room Type': 'Others', 'Samples': others_count}])
df_pie = pd.concat([df_pie, others_row], ignore_index=True)

# 2. 设置学术风格
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300 

# 创建单一画布
fig, ax1 = plt.subplots(figsize=(10, 7))

# ---------------------------------------------------------
# 3. 绘制主图：柱状图 (Bar Chart)
# ---------------------------------------------------------
palette = sns.color_palette("viridis", len(df))
bars = ax1.bar(df['Room Type'], df['Samples'], 
               color=palette, edgecolor='black', linewidth=1, alpha=0.8)


# 坐标轴美化
ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax1.set_xlabel('Room Categories', fontsize=12, fontweight='bold')
ax1.set_title('Room Type Frequency and Composition', fontsize=14, pad=20, loc='center', fontweight='bold')
ax1.tick_params(axis='x', rotation=40)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ---------------------------------------------------------
# 4. 绘制内嵌图：环形饼图 (Inset Donut Chart)
# ---------------------------------------------------------
# 使用 inset_axes 在主图右上角创建坐标系，利用空白区域
ax_inset = inset_axes(ax1, width="45%", height="45%", loc='upper right', borderpad=1)

pie_colors = sns.color_palette("Set3", len(df_pie))
wedges, texts, autotexts = ax_inset.pie(
    df_pie['Samples'], 
    labels=df_pie['Room Type'],
    autopct='%1.1f%%',
    startangle=140,
    colors=pie_colors,
    pctdistance=0.75,
    textprops={'fontsize': 7, 'fontweight': 'bold'}
)

# 设为环形图增加高级感
centre_circle = plt.Circle((0,0), 0.60, fc='white')
ax_inset.add_artist(centre_circle)

# 给子图起个简练的小标题
ax_inset.set_title('Composition Ratio', fontsize=10, fontweight='bold', pad=0)
ax_inset.axis('equal') 

# ---------------------------------------------------------
# 5. 最终调整
# ---------------------------------------------------------
plt.tight_layout()

# 保存高质量图像建议：
plt.savefig('room_dist_combined.pdf', dpi=600, bbox_inches='tight')

plt.show()


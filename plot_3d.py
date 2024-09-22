import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载和逐行解析文件
x, y, z, roll, pitch, yaw = [], [], [], [], [], []

# with open('target_pose_umi_2024_06_06_19_25.json') as f:
with open('target_pose_umi_2024_06_14_13_29.json') as f:
    for line in f:
        entry = json.loads(line.strip())
        x.append(entry[0])
        y.append(entry[1])
        z.append(entry[2])
        roll.append(entry[3])
        pitch.append(entry[4])
        yaw.append(entry[5])

# 选择第2000行到第2300行的数据
# start_index = 1300
# end_index = 1900
# x = x[start_index:end_index]
# y = y[start_index:end_index]
# z = z[start_index:end_index]
# roll = roll[start_index:end_index]
# pitch = pitch[start_index:end_index]
# yaw = yaw[start_index:end_index]



# 创建3D图
fig = plt.figure(figsize=(18, 15))
ax = fig.add_subplot(111, projection='3d')

# 绘制位置数据
ax.plot(x, y, z, label='end pose', color='blue')

# 控制箭头的绘制密度，例如每隔5个点绘制一个箭头
arrow_density = 5

# 将姿态数据表示为箭头
print("len(x)==",len(x))
for i in range(len(x)):
    if i % arrow_density == 0:
        # 计算姿态方向
        arrow_length = 0.02  # 调整箭头长度
        dx = arrow_length * np.cos(yaw[i]) * np.cos(pitch[i])
        dy = arrow_length * np.sin(yaw[i]) * np.cos(pitch[i])
        dz = arrow_length * np.sin(pitch[i])
        
        # 绘制箭头
        ax.quiver(x[i], y[i], z[i], dx, dy, dz, color='red')

# 设置标题和标签
ax.set_title('末端位置和姿态数据', fontproperties=font_prop)
# ax.set_xlabel('X', fontproperties=font_prop)
# ax.set_ylabel('Y', fontproperties=font_prop)
# ax.set_zlabel('Z', fontproperties=font_prop)
# ax.set_title('末端位置和姿态数据')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# 调整视角
ax.view_init(elev=20., azim=30)

# 显示图形
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
nsga3_rl_df = pd.read_csv('./data/pareto_front_nsga3_rl.csv')
nsga3_df = pd.read_csv('./data/pareto_front_nsga3.csv')

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制NSGA3-RL曲线
plt.scatter(nsga3_rl_df['Operation_Cost'], nsga3_rl_df['Carbon_Emissions'], color='blue', label='NSGA3-RL', s=50)

# 绘制NSGA3曲线
plt.scatter(nsga3_df['Operation_Cost'], nsga3_df['Carbon_Emissions'], color='red', label='NSGA3', s=50)

# 设置图表标题和坐标轴标签
plt.title('Pareto Front Comparison')
plt.xlabel('Operation Cost (¥)')
plt.ylabel('Carbon Emissions (ton)')

# 添加图例
plt.legend()

# 显示图表
plt.grid(True)
plt.savefig('pareto_front_comparison.png')
plt.show()

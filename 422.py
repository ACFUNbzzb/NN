import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 关闭 oneDNN 提示，避免输出干扰

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置 matplotlib 使用中文字体
plt.rcParams['font.family'] = 'SimHei'  # 黑体（适用于 Windows）
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# ========== 加载模型 ==========
model = load_model('saved_model/battery_model.h5', custom_objects={'mse': MeanSquaredError()})

# ========== 模拟新输入数据（5个系统，每个系统10个样本） ==========
sample_num = 10
input_dim = model.input_shape[0][1]  # 获取每个输入的特征维度
X_new_raw = [np.random.rand(sample_num, input_dim) for _ in range(5)]

# ========== 假设你有原模型训练时用的 scalers ==========
# 如果没有保存原 scalers，可以假设此处仅用于测试
scalers = [StandardScaler().fit(X) for X in X_new_raw]
X_new_scaled = [scalers[i].transform(X_new_raw[i]) for i in range(5)]

# ========== 构造多输入格式 ==========
X_new_total = list(zip(*X_new_scaled))
X_new_input = [np.array([x[i] for x in X_new_total]) for i in range(5)]

# ========== 模拟真实目标值（仅用于测试） ==========
y_true = np.random.rand(sample_num)

# ========== 模型预测 ==========
y_pred = model.predict(X_new_input).flatten()


# ========== 平滑函数 ==========
def moving_average(data, window_size=3):
    if len(data) < window_size:
        return data  # 样本太少则跳过平滑
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ========== 可视化（平滑后） ==========
os.makedirs('prediction_output', exist_ok=True)
plt.figure(figsize=(12, 6))
plt.plot(moving_average(y_true), label='真实值 平滑', alpha=0.8)
plt.plot(moving_average(y_pred), label='预测值 平滑', alpha=0.8)
plt.title('预测 vs 真实（平滑版）')
plt.xlabel('样本编号')
plt.ylabel('目标值')
plt.legend()
plt.grid(True)
plt.savefig('prediction_output/prediction_vs_true_smoothed.png')
plt.show()
print("已保存平滑后的预测图：prediction_output/prediction_vs_true_smoothed.png")

# ========== 打印评估指标 ==========
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"评估结果：")
print(f" - MSE: {mse:.4f}")
print(f" - MAE: {mae:.4f}")
print(f" - R²: {r2:.4f}")


import os
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

# ========= 模块1：准备化学式数据并提取特征 =========
def prepare_multi_system_inputs():
    # 定义化学系统的基本化学式列表
    system_bases = [
        'Li26Ni27FeO54', 'Li25Ni27Fe2O54', 'Li24Ni27Fe3O54', 'Li23Ni27Fe4O54', 'Li22Ni27Fe5O53',
        'Li26Ni27O54', 'Li27Ni27O54', 'Li27Ni27O53', 'Li27Ni27O52', 'Li25Ni27CoMnO54',
        'Li26Ni27CoO54', 'Li25Ni27Co2O54', 'Li24Ni27Co3O54', 'Li23Ni27Co4O54', 'Li22Ni27Co5O53',
        'Li26Ni27MnO54', 'Li25Ni27Mn2O54', 'Li24Ni27Mn3O54', 'Li23Ni27Mn4O54', 'Li22Ni27Mn5O53',
        'Li26Ni27AlO54', 'Li25Ni27Al2O54', 'Li24Ni27Al3O54', 'Li23Ni27Al4O54', 'Li22Ni27Al5O53'
    ]
    # 使用matminer中的ElementProperty提取化学成分特征
    featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
    X_all, y_all = [], []  # 存储特征和目标值

    # 遍历每个化学式
    for formula in system_bases:
        try:
            # 使用StrToComposition将化学式转换为组成对象
            df = pd.DataFrame({'formula': [formula]})
            df = StrToComposition().featurize_dataframe(df, col_id='formula', ignore_errors=True)
            # 使用ElementProperty提取化学成分特征
            df = featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=True)
            # 获取提取的特征
            features = df.drop(columns=['formula', 'composition']).values[0]
            X_all.append(features)
            # 人工生成目标值（这里只是示例，可以根据实际需要调整）
            y_all.append(0.4 + 0.01 * len(y_all))
        except Exception:
            continue

    # 将所有样本的特征合并成一个numpy数组
    X_merged = np.array(X_all)
    y_merged = np.array(y_all)
    print("📊 特征分布统计信息：")
    print(pd.DataFrame(X_merged).describe())  # 输出特征的统计信息

    return X_merged, y_merged


# ========= 模块2：标准化数据 =========
def standardize_and_align(X_raw, y_array):
    # 使用StandardScaler进行数据标准化
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)
    return X_scaled, y_array, X_raw.shape[1], scaler


# ========= 模块3：自定义数据集结构 =========
class MultiSystemDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # 特征
        self.y = y  # 目标值

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# ========= 模块4：定义神经网络结构 =========
class BatteryModel(nn.Module):
    def __init__(self, input_dim):
        super(BatteryModel, self).__init__()
        # 定义一个简单的全连接神经网络结构
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  # 输入层到隐藏层
            nn.ReLU(),                  # 激活函数
            nn.Dropout(0.1),             # Dropout正则化
            nn.Linear(256, 128),         # 隐藏层到隐藏层
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)            # 隐藏层到输出层
        )

    def forward(self, x):
        return self.fc(x)


# ========= 主程序入口 =========
if __name__ == '__main__':
    os.makedirs("training_output", exist_ok=True)  # 创建训练输出目录
    os.makedirs("prediction_output", exist_ok=True)  # 创建预测输出目录
    os.makedirs("saved_model", exist_ok=True)  # 创建保存模型的目录
    shap.initjs()  # 初始化SHAP库

    # ===== 步骤1：数据准备 =====
    X_raw, y = prepare_multi_system_inputs()  # 准备特征和目标值
    X_scaled, y, input_dim, scaler = standardize_and_align(X_raw, y)  # 数据标准化
    dataset = MultiSystemDataset(X_scaled, y)  # 创建自定义数据集

    # 划分训练集和验证集
    train_size = max(1, int(0.4 * len(dataset)))  # 训练集大小
    val_size = max(1, len(dataset) - train_size)  # 验证集大小
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # 随机拆分数据集
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 训练数据加载器
    val_loader = DataLoader(val_dataset, batch_size=8)  # 验证数据加载器

    # ===== 步骤2：构建模型 =====
    model = BatteryModel(input_dim)  # 创建模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    loss_fn = nn.MSELoss()  # 定义损失函数

    # ===== 步骤3：训练模型 =====
    train_losses, val_losses = [], []  # 存储训练损失和验证损失
    for epoch in range(50):  # 训练50轮
        model.train()  # 设置为训练模式
        total_loss = 0
        # 训练阶段
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/50"):
            optimizer.zero_grad()  # 清零梯度
            pred = model(X_batch)  # 模型预测
            loss = loss_fn(pred, y_batch.view(-1, 1))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失
        train_losses.append(total_loss / len(train_loader))  # 记录平均训练损失

        # 验证阶段
        model.eval()  # 设置为评估模式
        val_loss, preds, trues = 0, [], []
        with torch.no_grad():  # 不计算梯度
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)  # 模型预测
                val_loss += loss_fn(pred, y_batch.view(-1, 1)).item()  # 计算验证损失
                preds.append(pred)
                trues.append(y_batch)
        val_losses.append(val_loss / len(val_loader))  # 记录平均验证损失

        # 计算R²评分
        r2 = r2_score(torch.cat(trues).numpy(), torch.cat(preds).numpy())
        print(f"✅ Epoch {epoch + 1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, R²={r2:.4f}")

    # ===== 步骤4：绘制训练损失曲线 =====
    plt.figure()
    plt.plot(train_losses, label="Training Loss", color='blue')  # 训练损失曲线
    plt.plot(val_losses, label="Validation Loss", color='orange')  # 验证损失曲线
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_output/loss_curve_pytorch.png")  # 保存损失曲线图像
    plt.close()

    # ===== 步骤5：模型评估与图形生成 =====
    X_test = torch.tensor(X_scaled[:10], dtype=torch.float32)  # 测试数据
    y_test = torch.tensor(y[:10], dtype=torch.float32).view(-1, 1)  # 测试标签
    model.eval()
    with torch.no_grad():  # 不计算梯度
        y_pred = model(X_test).numpy()  # 预测结果

    y_true = y_test.numpy()
    residuals = y_pred.flatten() - y_true.flatten()  # 计算残差

    # 📈 预测值 vs 真实值（折线图）
    plt.figure()
    plt.plot(y_true, label="True Values", marker='o', color='blue')  # 真实值
    plt.plot(y_pred, label="Predicted Values", marker='x', color='red')  # 预测值
    plt.title("Predicted vs True Values (Line Plot)")
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("prediction_output/prediction_vs_true.png")  # 保存预测值与真实值的折线图
    plt.close()

    # 📉 残差图
    plt.figure()
    plt.scatter(y_true, residuals, color='green', label="Residual Points")  # 残差点
    plt.axhline(0, color='red', linestyle='--', label="Zero Residual")  # 0残差线
    plt.xlabel("True Values")
    plt.ylabel("Residuals (Predicted - True)")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.legend()
    plt.savefig("prediction_output/residual_plot.png")  # 保存残差图
    plt.close()

    # 📊 残差直方图
    plt.figure()
    plt.hist(residuals, bins=10, color='orange', edgecolor='black')  # 残差直方图
    plt.title("Prediction Error Histogram")
    plt.xlabel("Error Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("prediction_output/error_histogram.png")  # 保存残差直方图
    plt.close()

    # 📄 保存评估指标
    mse = mean_squared_error(y_true, y_pred)  # 计算均方误差
    mae = mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差
    r2 = r2_score(y_true, y_pred)  # 计算R²得分
    spearman_corr, _ = spearmanr(y_true, y_pred)  # 计算Spearman相关系数
    pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())  # 计算Pearson相关系数
    with open("prediction_output/evaluation_metrics.txt", "w") as f:
        f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}\n")
        f.write(f"Spearman: {spearman_corr:.4f}\nPearson: {pearson_corr:.4f}\n")  # 保存评估指标

    # ===== 步骤7：保存模型参数 =====
    torch.save(model.state_dict(), "saved_model/battery_model.pth")  # 保存模型参数
    print("✅ 模型训练与保存完成，所有图像与评估文件已生成。")

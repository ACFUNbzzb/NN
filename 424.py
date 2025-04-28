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
from sklearn.model_selection import train_test_split
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr


# ========= 模块1：准备化学式数据并提取特征 =========
def prepare_multi_system_inputs():
    """准备化学式数据并提取特征"""
    system_bases = [
        'Li26Ni27FeO54', 'Li25Ni27Fe2O54', 'Li24Ni27Fe3O54', 'Li23Ni27Fe4O54', 'Li22Ni27Fe5O53',
        'Li26Ni27O54', 'Li27Ni27O54', 'Li27Ni27O53', 'Li27Ni27O52', 'Li25Ni27CoMnO54',
        'Li26Ni27CoO54', 'Li25Ni27Co2O54', 'Li24Ni27Co3O54', 'Li23Ni27Co4O54', 'Li22Ni27Co5O53',
        'Li26Ni27MnO54', 'Li25Ni27Mn2O54', 'Li24Ni27Mn3O54', 'Li23Ni27Mn4O54', 'Li22Ni27Mn5O53',
        'Li26Ni27AlO54', 'Li25Ni27Al2O54', 'Li24Ni27Al3O54', 'Li23Ni27Al4O54', 'Li22Ni27Al5O53',
        # 假设你有50个真实样本，继续补充列表
    ]
    featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
    X_all, y_all = [], []

    for formula in system_bases:
        try:
            # 将化学式转换为化学成分
            df = pd.DataFrame({'formula': [formula]})
            df = StrToComposition().featurize_dataframe(df, col_id='formula', ignore_errors=True)
            # 提取化学成分特征
            df = featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=True)
            # 获取特征值
            features = df.drop(columns=['formula', 'composition']).values[0]
            X_all.append(features)
            # 模拟标签：基于当前样本的索引生成一个标签
            y_all.append(0.4 + 0.01 * len(y_all))
        except Exception as e:
            # 发生错误时打印出错化学式
            print(f"❌ 处理化学式 {formula} 时发生错误: {e}")
            continue

    # 将所有特征数据和标签合并为 numpy 数组
    X_merged = np.array(X_all)
    y_merged = np.array(y_all)
    print("\U0001F4CA 特征分布统计信息：")
    print(pd.DataFrame(X_merged).describe())

    return X_merged, y_merged


# ========= 模块2：标准化数据 =========
def standardize_and_align(X_raw, y_array):
    """标准化特征"""
    # 使用StandardScaler对特征进行标准化，使其均值为0，标准差为1
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)
    return X_scaled, y_array, X_raw.shape[1], scaler


# ========= 模块3：自定义数据集结构 =========
class MultiSystemDataset(Dataset):
    """PyTorch数据集封装"""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# ========= 模块4：定义神经网络结构 =========
class BatteryModel(nn.Module):
    """简单MLP回归模型"""
    def __init__(self, input_dim, hidden_units=64):
        super(BatteryModel, self).__init__()
        # 定义一个简单的全连接神经网络（MLP）
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_units),  # 输入层到隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_units, hidden_units),  # 第二个隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_units, 1)  # 输出层
        )

    def forward(self, x):
        return self.fc(x)


# ========= 模块5：数据扩充函数 =========
def augment_with_noise(X, y, target_size=100, noise_level_X=0.01, noise_level_y=0.005):
    """扩充数据到指定数量（强制设置为100），并添加小幅噪声"""
    X_aug, y_aug = [X.copy()], [y.copy()]
    rng = np.random.default_rng()

    # 计算当前数据集的大小
    current_size = X.shape[0]

    # 如果当前数据量小于目标数量，持续扩充数据
    while len(X_aug) * current_size < target_size:
        noise_X = rng.normal(0, noise_level_X, X.shape)  # 对特征添加噪声
        noise_y = rng.normal(0, noise_level_y, y.shape)  # 对标签添加噪声
        X_new = X + noise_X
        y_new = y + noise_y
        X_aug.append(X_new)
        y_aug.append(y_new)

    # 最终数据集大小设置为目标大小
    X_final = np.vstack(X_aug)[:target_size]
    y_final = np.hstack(y_aug)[:target_size]
    return X_final, y_final


# ========= 主程序入口 =========
if __name__ == '__main__':
    # 创建输出文件夹
    os.makedirs("training_output", exist_ok=True)
    os.makedirs("prediction_output", exist_ok=True)
    os.makedirs("saved_model", exist_ok=True)

    shap.initjs()  # 启用SHAP可视化工具

    # 原始数据准备
    X_raw, y_raw = prepare_multi_system_inputs()
    X_scaled, y, input_dim, scaler = standardize_and_align(X_raw, y_raw)

    # 第一步：划分15%的原始数据作为测试集
    X_trainval_raw, X_test_raw, y_trainval_raw, y_test_raw = train_test_split(X_scaled,
                                                                              y, test_size=0.15, random_state=42)

    # 第二步：扩充训练集到100个样本
    X_trainval_aug, y_trainval_aug = augment_with_noise(X_trainval_raw, y_trainval_raw, target_size=100)

    # 记录每折的评估指标
    all_metrics = {
        'mse': [],
        'mae': [],
        'r2': [],
        'spearman': [],
        'pearson': []
    }

    # 进行10次交叉验证
    for fold in range(1, 11):
        print(f"\n🚀 开始第 {fold}/10 次交叉验证")

        # 从扩充的训练集中划分验证集
        X_train, X_val, y_train, y_val = train_test_split(X_trainval_aug, y_trainval_aug, test_size=0.15, random_state=fold*10)

        # 创建数据集和数据加载器
        train_dataset = MultiSystemDataset(X_train, y_train)
        val_dataset = MultiSystemDataset(X_val, y_val)
        test_dataset = MultiSystemDataset(X_test_raw, y_test_raw)  # 测试集一直是真实原始数据！

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        test_loader = DataLoader(test_dataset, batch_size=8)

        # 定义模型、优化器和损失函数
        model = BatteryModel(input_dim, hidden_units=64)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        best_val_loss = float('inf')  # 初始最优验证损失
        train_losses, val_losses = [], []  # 存储训练和验证损失

        for epoch in range(100):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch.view(-1, 1))  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 优化步骤
                total_loss += loss.item()

            train_losses.append(total_loss / len(train_loader))  # 记录训练损失

            # 验证损失
            model.eval()
            val_loss = 0
            preds, trues = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    pred = model(X_batch)
                    val_loss += loss_fn(pred, y_batch.view(-1, 1)).item()
                    preds.append(pred)
                    trues.append(y_batch)

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)  # 记录验证损失

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss  # 更新最优验证损失

        # 保存每折的模型
        torch.save(model.state_dict(), f'saved_model/model_fold_{fold}.pt')

        # 测试评估
        model.eval()
        X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_raw, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()

        # 绘制训练和验证损失曲线
        plt.figure()
        plt.plot(train_losses, label="Training Loss", color='blue')
        plt.plot(val_losses, label="Validation Loss", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_output/loss_curve_pytorch.png")
        plt.close()

        # 计算评估指标
        y_true = y_test_tensor.numpy()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        residuals = y_pred.flatten() - y_true.flatten()

        # 预测值 vs 真实值折线图
        plt.figure()
        plt.plot(y_true, label="True Values", marker='o', color='blue')
        plt.plot(y_pred, label="Predicted Values", marker='x', color='red')
        plt.title("Predicted vs True Values (Test Set)")
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        plt.legend()
        plt.grid(True)
        plt.savefig("prediction_output/prediction_vs_true_test.png")
        plt.close()

        # 残差图
        plt.figure()
        plt.scatter(y_true, residuals, color='green', label="Residual Points")
        plt.axhline(0, color='red', linestyle='--', label="Zero Residual")
        plt.xlabel("True Values")
        plt.ylabel("Residuals (Predicted - True)")
        plt.title("Residual Plot (Test Set)")
        plt.grid(True)
        plt.legend()
        plt.savefig("prediction_output/residual_plot_test.png")
        plt.close()

        # 残差直方图
        plt.figure()
        plt.hist(residuals, bins=10, color='orange', edgecolor='black')
        plt.title("Prediction Error Histogram (Test Set)")
        plt.xlabel("Error Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig("prediction_output/error_histogram_test.png")
        plt.close()

        # 存储各折的评估指标
        all_metrics['mse'].append(mse)
        all_metrics['mae'].append(mae)
        all_metrics['r2'].append(r2)
        all_metrics['spearman'].append(spearman_corr)
        all_metrics['pearson'].append(pearson_corr)

        print(f"✅ Fold {fold}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Spearman={spearman_corr:.4f},"
              f" Pearson={pearson_corr:.4f}")

    # 汇总10次交叉验证的结果
    print("\n📊 交叉验证10次结果统计：")
    with open("prediction_output/crossval_metrics_summary.txt", "w") as f:
        for metric_name in all_metrics.keys():
            mean_val = np.mean(all_metrics[metric_name])
            std_val = np.std(all_metrics[metric_name])
            result_line = f"{metric_name.upper()}: Mean={mean_val:.4f}, Std={std_val:.4f}"
            print(result_line)
            f.write(result_line + "\n")

    # 绘制R²分数分布箱线图
    plt.figure()
    plt.boxplot(all_metrics['r2'], patch_artist=True, labels=['R²'])
    plt.title('R² Score Distribution Across 10 Folds')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.savefig("prediction_output/r2_score_boxplot.png")
    plt.close()

    print("\n✅ 全部完成，交叉验证指标和图表已保存！")

    # ===== 步骤7：保存模型参数 =====
    torch.save(model.state_dict(), "saved_model/battery_model.pth")  # 保存模型参数
    print("✅ 模型训练与保存完成，所有图像与评估文件已生成。")

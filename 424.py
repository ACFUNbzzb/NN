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
    """准备50个化学式样本，并提取对应的特征"""
    system_bases = [
        'Li26Ni27FeO54', 'Li25Ni27Fe2O54', 'Li24Ni27Fe3O54', 'Li23Ni27Fe4O54', 'Li22Ni27Fe5O53',
        'Li26Ni27O54', 'Li27Ni27O54', 'Li27Ni27O53', 'Li27Ni27O52', 'Li25Ni27CoMnO54',
        'Li26Ni27CoO54', 'Li25Ni27Co2O54', 'Li24Ni27Co3O54', 'Li23Ni27Co4O54', 'Li22Ni27Co5O53',
        'Li26Ni27MnO54', 'Li25Ni27Mn2O54', 'Li24Ni27Mn3O54', 'Li23Ni27Mn4O54', 'Li22Ni27Mn5O53',
        'Li26Ni27AlO54', 'Li25Ni27Al2O54', 'Li24Ni27Al3O54', 'Li23Ni27Al4O54', 'Li22Ni27Al5O53',
        'Li26Ni27Fe3MnO54', 'Li25Ni27Fe2Mn2O54', 'Li24Ni27FeMn3O54', 'Li23Ni27Mn4FeO54', 'Li22Ni27Mn5FeO53',
        'Li26Ni27Fe3CoO54', 'Li25Ni27Fe2Co2O54', 'Li24Ni27FeCo3O54', 'Li23Ni27Co4FeO54', 'Li22Ni27Co5FeO53',
        'Li26Ni27CoMn2O54', 'Li25Ni27CoMn3O54', 'Li24Ni27CoMn4O54', 'Li23Ni27CoMn5O54', 'Li22Ni27CoMn6O53',
        'Li26Ni27Co2AlO54', 'Li25Ni27Co2Al2O54', 'Li24Ni27Co2Al3O54', 'Li23Ni27Co2Al4O54', 'Li22Ni27Co2Al5O53',
        'Li26Ni27Mn2AlO54', 'Li25Ni27Mn2Al2O54', 'Li24Ni27Mn2Al3O54', 'Li23Ni27Mn2Al4O54', 'Li22Ni27Mn2Al5O53'
    ]

    featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
    X_all, y_all = [], []
    success_list, fail_list = [], []

    for idx, formula in enumerate(system_bases):
        try:
            df = pd.DataFrame({'formula': [formula]})
            df = StrToComposition().featurize_dataframe(df, col_id='formula', ignore_errors=False)
            df = featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=False)
            features = df.drop(columns=['formula', 'composition']).values[0]
            X_all.append(features)
            y_all.append(0.4 + 0.01 * idx)  # 模拟标签
            success_list.append(formula)
        except Exception as e:
            print(f"❌ 错误处理化学式 {formula}: {e}")
            fail_list.append(formula)

    X_merged = np.array(X_all)
    y_merged = np.array(y_all)

    # ✅ 说明统计表格内容
    print(f"\n📊 当前成功提取特征的样本数: {X_merged.shape[0]}，每个样本的特征维度: {X_merged.shape[1]}")
    print("📈 以下为特征列的统计分布（注意：是每列的描述，不是样本行数）:")
    print(pd.DataFrame(X_merged).describe())

    # ✅ 打印成功和失败列表
    print(f"\n✅ 成功提取化学式数量: {len(success_list)}")
    print(f"❌ 提取失败数量: {len(fail_list)}")
    if fail_list:
        print("失败化学式如下：", fail_list)

    return X_merged, y_merged


# ========= 模块2：数据标准化 =========
def augment_with_noise(X, y, target_size=100, noise_level_X=0.01, noise_level_y=0.005):
    """扩充数据到指定数量（100），并添加噪声"""
    X_aug, y_aug = [X.copy()], [y.copy()]
    rng = np.random.default_rng()
    while len(X_aug) * X.shape[0] < target_size:
        noise_X = rng.normal(0, noise_level_X, X.shape)
        noise_y = rng.normal(0, noise_level_y, y.shape)
        X_aug.append(X + noise_X)
        y_aug.append(y + noise_y)
    X_final = np.vstack(X_aug)[:target_size]
    y_final = np.hstack(y_aug)[:target_size]
    return X_final, y_final


# ========= 模块3：数据集封装 =========
class MultiSystemDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# ========= 模块4：模型结构 =========
class BatteryModel(nn.Module):
    def __init__(self, input_dim, hidden_units=64):
        super(BatteryModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x):
        return self.fc(x)


# ========= 主程序入口 =========
if __name__ == '__main__':
    os.makedirs("training_output", exist_ok=True)
    os.makedirs("prediction_output", exist_ok=True)
    os.makedirs("saved_model", exist_ok=True)
    shap.initjs()

    # 原始数据准备
    X_raw, y_raw = prepare_multi_system_inputs()
    num_total = X_raw.shape[0]

    if num_total < 20:
        raise ValueError(f"❗ 原始样本数仅有 {num_total}，不足以划分10个验证和足够训练。")

    # 1️⃣ 划分数据集：10个验证，其余用于训练
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_raw, y_raw, test_size=10, random_state=42
    )

    # 2️⃣ 再从训练集中划分 20% 为测试集
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_train_raw, y_train_raw, test_size=0.2, random_state=1
    )

    # 3️⃣ 记录特征维度
    input_dim = X_raw.shape[1]

    # 4️⃣ 标准化（仅对训练集 fit，再 transform 其它）
    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 5️⃣ 扩增训练集到 100 个样本
    X_train_aug, y_train_aug = augment_with_noise(X_train_scaled, y_train_raw, target_size=100)

    # ✅ 打印检查信息
    print("\n📦 数据集划分检查：")
    print(f"原始样本总数（成功 featurize 的）：{num_total}")
    print(f"验证集样本数（原始样本中抽10个）：{len(X_val_raw)}")
    print(f"测试集样本数（从训练集中再分出20%）：{len(X_test_raw)}")
    print(f"最终用于扩增的原始训练样本数：{len(X_train_raw)}")
    print(f"扩增后训练样本数：{X_train_aug.shape[0]}")
    print(f"训练集特征维度：{X_train_aug.shape[1]}")
    print(f"训练标签 shape: {y_train_aug.shape}")
    print(f"验证标签 shape: {y_val_raw.shape}")
    print(f"测试标签 shape: {y_test_raw.shape}")

    # 🔍 标签分布图
    plt.figure()
    plt.hist(y_train_aug, bins=20, alpha=0.6, label="Train (Augmented)", color='blue')
    plt.hist(y_val_raw, bins=10, alpha=0.6, label="Validation", color='green')
    plt.hist(y_test_raw, bins=10, alpha=0.6, label="Test", color='red')
    plt.title("Target Distribution Across Datasets")
    plt.xlabel("Target Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("prediction_output/target_distribution.png")
    plt.close()

    # 🎯 初始化交叉验证评估指标存储
    all_metrics = {'mse': [], 'mae': [], 'r2': [], 'spearman': [], 'pearson': []}

    # 🔁 交叉验证训练
    for fold in range(1, 11):
        print(f"\n🚀 开始第 {fold}/10 次交叉验证")

        train_dataset = MultiSystemDataset(X_train_aug, y_train_aug)
        val_dataset = MultiSystemDataset(X_val_scaled, y_val_raw)
        test_dataset = MultiSystemDataset(X_test_scaled, y_test_raw)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=8, num_workers=0)

        model = BatteryModel(input_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        best_val_loss = float('inf')
        train_losses, val_losses = [], []

        for epoch in range(100):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch.view(-1, 1))
                if torch.isnan(loss):
                    print("❌ Loss 出现 NaN，停止训练")
                    break
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_losses.append(total_loss / len(train_loader))

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    pred = model(X_batch)
                    val_loss += loss_fn(pred, y_batch.view(-1, 1)).item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        # 🧠 保存模型
        torch.save(model.state_dict(), f'saved_model/model_fold_{fold}.pt')

        # 📊 测试集评估
        model.eval()
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_raw, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy()

        # 📈 可视化损失
        plt.figure()
        plt.plot(train_losses, label="Training Loss", color='blue')
        plt.plot(val_losses, label="Validation Loss", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"training_output/loss_curve_fold_{fold}.png")
        plt.close()

        # 📊 评估指标
        y_true = y_test_tensor.numpy()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        residuals = y_pred.flatten() - y_true.flatten()

        # 🔎 各类图表
        plt.figure()
        plt.plot(y_true, label="True", marker='o')
        plt.plot(y_pred, label="Predicted", marker='x')
        plt.title("Predicted vs True (Test Set)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"prediction_output/pred_vs_true_fold_{fold}.png")
        plt.close()

        plt.figure()
        plt.scatter(y_true, residuals, label="Residuals")
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("True Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.grid(True)
        plt.savefig(f"prediction_output/residual_fold_{fold}.png")
        plt.close()

        plt.figure()
        plt.hist(residuals, bins=10, edgecolor='black')
        plt.title("Error Histogram")
        plt.savefig(f"prediction_output/error_hist_fold_{fold}.png")
        plt.close()

        all_metrics['mse'].append(mse)
        all_metrics['mae'].append(mae)
        all_metrics['r2'].append(r2)
        all_metrics['spearman'].append(spearman_corr)
        all_metrics['pearson'].append(pearson_corr)

        print(
            f"✅ Fold {fold}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")

    # 🧾 汇总结果
    print("\n📊 交叉验证结果统计：")
    with open("prediction_output/crossval_metrics_summary.txt", "w") as f:
        for metric_name, values in all_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            result = f"{metric_name.upper()}: Mean={mean_val:.4f}, Std={std_val:.4f}"
            print(result)
            f.write(result + "\n")

    # 📦 R² 分布图
    plt.figure()
    plt.boxplot(all_metrics['r2'], patch_artist=True, labels=['R²'])
    plt.title('R² Score Distribution')
    plt.grid(True)
    plt.savefig("prediction_output/r2_score_boxplot.png")
    plt.close()

    print("\n✅ 全部完成，交叉验证图表和评估指标已保存！")

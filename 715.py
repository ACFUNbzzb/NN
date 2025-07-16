# ------------------------
# 导入所需的库
# ------------------------
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn工具
from sklearn.model_selection import RepeatedKFold, GridSearchCV, learning_curve, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# 统计工具
from scipy.stats import spearmanr, pearsonr

# XGBoost回归器
from xgboost import XGBRegressor

# matminer用于材料科学特征工程
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty


# ------------------------
# 数据增强函数：给数据加噪声
# ------------------------
def augment_with_noise(X, y, target_size=100, noise_level_X=0.001, noise_level_y=0.01):
    """
    对输入的特征矩阵X和标签y进行增强，生成带随机噪声的新样本。
    """
    X_aug, y_aug = [X.copy()], [y.copy()]
    rng = np.random.default_rng()

    # 循环生成带噪声的新数据，直到样本数达到target_size
    while len(X_aug) * X.shape[0] < target_size:
        noise_X = rng.normal(0, noise_level_X, X.shape)
        noise_y = rng.normal(0, noise_level_y, y.shape)
        X_aug.append(X + noise_X)
        y_aug.append(y + noise_y)

    # 拼接所有增强数据
    X_final = np.vstack(X_aug)[:target_size]
    y_final = np.vstack(y_aug)[:target_size]
    return X_final, y_final


# ------------------------
# 绘制性能指标汇总条形图
# ------------------------
def plot_summary_metrics(all_metrics, target_columns):
    """
    将交叉验证得到的多次指标结果进行平均，绘制每个目标的汇总图。
    """
    for target in target_columns:
        subset = pd.DataFrame(all_metrics[target])
        mean_vals = subset.mean(axis=0)
        std_vals = subset.std(axis=0)

        plt.figure(figsize=(8, 5), dpi=300)
        bars = plt.bar(
            mean_vals.index,
            mean_vals.values,
            yerr=std_vals.values,
            capsize=5,
            color='lightcoral',
            edgecolor='black'
        )
        plt.title(f"Summary Metrics - {target}")
        plt.ylabel("Metric Value")

        # 给每个bar加上数值标注
        for bar, mean, std in zip(bars, mean_vals.values, std_vals.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + std + 0.01,
                     f"{mean:.4f}", ha='center', fontsize=10)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join("prediction_output", "Metrics", f"summary_metrics_{target}.png"), dpi=300)
        plt.close()


# ------------------------
# 绘制学习曲线
# ------------------------
def generate_learning_curves(X_train, y_train, target_columns, best_params):
    """
    针对每一个目标变量，绘制学习曲线，用于查看模型是否过拟合或欠拟合。
    """
    for i, target_name in enumerate(target_columns):
        y_target = y_train[:, i]

        # 如果目标变量几乎是常数，跳过绘图
        if np.std(y_target) < 1e-8:
            print(f"⚠️ Target {target_name} is constant. Skipping learning curve.")
            continue

        model = XGBRegressor(**best_params, random_state=42)

        # 生成学习曲线
        train_sizes, train_errors, val_errors = learning_curve(
            model,
            X_train,
            y_target,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        train_errors_mean = -train_errors.mean(axis=1)
        val_errors_mean = -val_errors.mean(axis=1)

        plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(train_sizes, train_errors_mean, label='Training Error', marker='o')
        plt.plot(train_sizes, val_errors_mean, label='Validation Error', marker='x')
        plt.title(f"Learning Curve - {target_name}")
        plt.xlabel("Training Size")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("prediction_output", "Learning_Curves", f"learning_curve_{target_name}.png"))
        plt.close()


# ------------------------
# 绘制平均残差柱状图
# ------------------------
def plot_avg_residuals(all_y_true, all_y_pred, target_columns):
    """
    计算平均残差并绘制柱状图。
    """
    residuals = np.vstack(all_y_true) - np.vstack(all_y_pred)
    avg_residuals = np.mean(residuals, axis=0)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.bar(target_columns, avg_residuals, color='orange')
    plt.title("Average Residuals per Target")
    plt.xlabel("Target")
    plt.ylabel("Average Residual")
    plt.tight_layout()
    plt.savefig(os.path.join("prediction_output", "Residuals", "average_residuals.png"), dpi=300)
    plt.close()


# ------------------------
# 绘制平均误差分布直方图
# ------------------------
def plot_avg_error_distribution(all_y_true, all_y_pred, target_columns):
    """
    绘制所有目标变量平均误差分布的直方图。
    """
    errors = np.vstack(all_y_true) - np.vstack(all_y_pred)
    avg_errors = np.mean(errors, axis=0)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.hist(avg_errors, bins=20, color='lightgreen', edgecolor='black')
    plt.title("Average Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join("prediction_output", "Distribution", "average_error_distribution.png"), dpi=300)
    plt.close()


# ------------------------
# 绘制平均实际 vs 预测散点图
# ------------------------
def plot_avg_scatter_actual_vs_predicted(all_y_true, all_y_pred, target_columns):
    """
    绘制平均实际值与平均预测值之间的散点图。
    """
    avg_y_true = np.mean(all_y_true, axis=0)
    avg_y_pred = np.mean(all_y_pred, axis=0)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(avg_y_true, avg_y_pred, color='purple')
    plt.plot([avg_y_true.min(), avg_y_true.max()],
             [avg_y_true.min(), avg_y_true.max()], 'r--')
    plt.title("Average Actual vs Predicted")
    plt.xlabel("Average Actual Values")
    plt.ylabel("Average Predicted Values")
    plt.tight_layout()
    plt.savefig(os.path.join("prediction_output", "Scatter_Plots", "average_scatter.png"), dpi=300)
    plt.close()


# ------------------------
# SHAP值分析
# ------------------------
def shap_analysis(model, X_train, feature_names):
    """
    对每个输出目标的模型做SHAP解释，绘制SHAP值总结图。
    """
    for i, estimator in enumerate(model.estimators_):
        print(f"Analyzing SHAP for target {i}...")
        explainer = shap.Explainer(estimator.predict, X_train)
        shap_values = explainer(X_train).values

        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join("prediction_output", "SHAP", f"shap_summary_plot_target_{i}.png"), dpi=300)
        plt.close()


# ------------------------
# 绘制平均特征重要性
# ------------------------
def plot_average_feature_importance(model, feature_names):
    """
    统计多输出模型所有子模型的平均特征重要性，并绘制前20特征。
    """
    feature_importance = np.zeros(len(feature_names))

    for estimator in model.estimators_:
        booster = estimator.get_booster()
        scores = booster.get_score(importance_type='weight')
        scores = {k: float(v) for k, v in scores.items()}

        for feat, score in scores.items():
            idx = int(feat[1:])
            if idx < len(feature_names):
                feature_importance[idx] += score

    feature_importance /= len(model.estimators_)

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    }).sort_values("Importance", ascending=False).head(20)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.bar(df["Feature"], df["Importance"], color='skyblue')
    plt.xticks(rotation=90)
    plt.title("Top 20 Average Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join("prediction_output", "Feature_Importance", "top_20_average_feature_importance.png"),
                dpi=300)
    plt.close()

    return feature_importance


# ------------------------
# 绘制目标变量之间的相关性热力图
# ------------------------
def plot_target_correlation_heatmap(all_y_true, target_columns):
    """
    绘制各目标变量间的皮尔逊相关系数热力图。
    """
    df_true = pd.DataFrame(all_y_true, columns=target_columns)
    corr_matrix = df_true.corr(method='pearson')

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                cbar=True, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap Between Targets")
    plt.tight_layout()
    plt.savefig(os.path.join("prediction_output", "Correlation", "target_correlation_heatmap.png"), dpi=300)
    plt.close()


# ------------------------
# 绘制Top20特征的相关性热力图
# ------------------------
def plot_feature_correlation_heatmap(X, feature_names, feature_importance):
    """
    从特征中选取前20重要的特征，绘制它们之间的相关性热力图。
    """
    top_idx = np.argsort(feature_importance)[-20:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    print("Top 20 features for correlation heatmap:")
    print(top_features)

    X_top = X[:, top_idx]
    df_X = pd.DataFrame(X_top, columns=top_features)
    corr_matrix = df_X.corr(method='pearson')

    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(corr_matrix, cmap='coolwarm', square=True,
                annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap (Top 20 Features)")
    plt.tight_layout()
    plt.savefig(os.path.join("prediction_output", "Correlation", "feature_correlation_heatmap.png"), dpi=300)
    plt.close()


# ------------------------
# 绘制Parity Plot（预测 vs 实际散点图）
# ------------------------
def plot_parity_plot(model, X, y, feature_names, target_columns, test_size=0.2):
    """
    对每个目标变量绘制Parity Plot，展示预测值和实际值的一致性。
    """
    for i, target in enumerate(target_columns):
        y_target = y[:, i]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_target, test_size=test_size, random_state=42
        )

        # 从MultiOutputRegressor中提取单模型参数
        params = model.get_params()['estimator'].get_params()
        params.pop('random_state', None)

        reg = XGBRegressor(**params, random_state=42)
        reg.fit(X_train, y_train)

        y_train_pred = reg.predict(X_train)
        y_test_pred = reg.predict(X_test)

        r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)

        plt.figure(figsize=(6, 6), dpi=300)
        plt.scatter(y_train, y_train_pred, color='orange', label='Training set', alpha=0.7, s=30)
        plt.scatter(y_test, y_test_pred, color='skyblue',
                    label=f'Testing set\n$R^2$={r2:.3f}, MAE={mae:.3f}',
                    alpha=0.7, s=30)
        plt.plot([min(y_target), max(y_target)],
                 [min(y_target), max(y_target)],
                 'k--', label='Reference Line')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Parity Plot - {target}')
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join("prediction_output", "Scatter_Plots", f"parity_plot_{target}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Parity plot saved: {out_path}")


# ------------------------
# 主程序入口
# ------------------------
if __name__ == '__main__':
    # 创建存储输出的文件夹
    os.makedirs("prediction_output", exist_ok=True)
    folders = ["Learning_Curves", "Scatter_Plots", "Residuals", "Feature_Importance", "Distribution", "Metrics", "SHAP",
               "Correlation"]
    for f in folders:
        os.makedirs(os.path.join("prediction_output", f), exist_ok=True)

    # -------------------------------
    # 1. 加载数据
    # -------------------------------
    data_file = './0709.xlsx'
    df = pd.read_excel(data_file)
    target_columns = df.columns[1:]  # 除去第一列化学式，其余都是目标
    formula_list = df.iloc[:, 0].tolist()
    y_raw = df.iloc[:, 1:].values

    # -------------------------------
    # 2. 化学式转换为特征向量
    # -------------------------------
    featurizer = ElementProperty.from_preset("magpie")
    df_formula = pd.DataFrame({'formula': formula_list})

    # 化学式 → Composition对象
    df_formula = StrToComposition().featurize_dataframe(df_formula, col_id='formula', ignore_errors=False)

    # Composition对象 → 数值特征
    df_formula = featurizer.featurize_dataframe(df_formula, col_id='composition', ignore_errors=False)

    # -------------------------------
    # 3. 合并体积数据
    # -------------------------------
    df_volumes = pd.read_csv("volumes.csv")
    df_volumes.columns = df_volumes.columns.str.strip()  # 去掉可能的空格
    old_formula_col = df_volumes.columns[0]
    old_volume_col = df_volumes.columns[1]

    df_volumes.rename(columns={old_formula_col: 'formula'}, inplace=True)
    df_volumes.rename(columns={old_volume_col: 'volume_of_cell'}, inplace=True)

    # 与公式特征表合并
    df_formula = pd.merge(
        df_formula,
        df_volumes,
        on='formula',
        how='left'
    )

    df_formula = df_formula.fillna(0)  # 缺失体积填0

    # 提取特征矩阵
    feature_names = list(df_formula.drop(columns=['formula', 'composition']).columns)
    X_raw = df_formula.drop(columns=['formula', 'composition']).values
    print("✅ Shape of X_raw after adding volume feature:", X_raw.shape)

    # -------------------------------
    # 4. 数据增强
    # -------------------------------
    X_aug, y_aug = augment_with_noise(
        X_raw, y_raw,
        target_size=max(len(X_raw), 100),
        noise_level_X=0.001,
        noise_level_y=0.01 * np.std(y_raw, axis=0)
    )

    # -------------------------------
    # 5. 网格搜索寻找最佳超参数
    # -------------------------------
    param_grid = {
        'estimator__max_depth': [3, 5],
        'estimator__learning_rate': [0.01, 0.05],
        'estimator__n_estimators': [500],
        'estimator__reg_alpha': [0],
        'estimator__reg_lambda': [0]
    }

    base_model = XGBRegressor(eval_metric='rmse')
    multi_model = MultiOutputRegressor(base_model)
    grid_search = GridSearchCV(multi_model, param_grid, scoring='r2', cv=3, n_jobs=-1)
    grid_search.fit(X_aug, y_aug)

    best_params_raw = grid_search.best_params_
    best_params = {k.replace('estimator__', ''): v for k, v in best_params_raw.items() if k.startswith('estimator__')}
    print("✅ Best params:", best_params)

    # -------------------------------
    # 6. 使用多次交叉验证训练
    # -------------------------------
    repeats = 5
    folds = 10
    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=42)

    all_metrics = {t: {'mse': [], 'mae': [], 'r2': [], 'spearman': [], 'pearson': []} for t in target_columns}
    all_y_true = []
    all_y_pred = []

    for train_idx, val_idx in rkf.split(X_aug):
        X_train, X_val = X_aug[train_idx], X_aug[val_idx]
        y_train, y_val = y_aug[train_idx], y_aug[val_idx]

        model = MultiOutputRegressor(
            XGBRegressor(eval_metric='rmse', **best_params, random_state=42)
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        all_y_true.append(y_val)
        all_y_pred.append(y_pred)

        for i, target in enumerate(target_columns):
            y_val_t = y_val[:, i]
            y_pred_t = y_pred[:, i]
            mse = mean_squared_error(y_val_t, y_pred_t)
            mae = mean_absolute_error(y_val_t, y_pred_t)
            r2 = r2_score(y_val_t, y_pred_t)
            spearman_corr, _ = spearmanr(y_val_t, y_pred_t)
            pearson_corr, _ = pearsonr(y_val_t, y_pred_t)

            all_metrics[target]['mse'].append(mse)
            all_metrics[target]['mae'].append(mae)
            all_metrics[target]['r2'].append(r2)
            all_metrics[target]['spearman'].append(spearman_corr)
            all_metrics[target]['pearson'].append(pearson_corr)

    # -------------------------------
    # 7. 绘图与结果分析
    # -------------------------------
    plot_summary_metrics(all_metrics, target_columns)
    generate_learning_curves(X_aug, y_aug, target_columns, best_params)
    shap_analysis(model, X_aug, feature_names)

    all_y_true = np.vstack(all_y_true)
    all_y_pred = np.vstack(all_y_pred)

    feature_importance = plot_average_feature_importance(model, feature_names)
    plot_avg_residuals(all_y_true, all_y_pred, target_columns)
    plot_avg_error_distribution(all_y_true, all_y_pred, target_columns)
    plot_avg_scatter_actual_vs_predicted(all_y_true, all_y_pred, target_columns)
    plot_target_correlation_heatmap(all_y_true, target_columns)
    plot_feature_correlation_heatmap(X_aug, feature_names, feature_importance)
    plot_parity_plot(model, X_aug, y_aug, feature_names, target_columns)

    print("✅ All processes completed!")

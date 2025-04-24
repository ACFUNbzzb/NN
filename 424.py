import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

# 设置 matplotlib 使用中文字体
plt.rcParams['font.family'] = 'SimHei'  # 黑体（适用于 Windows）
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# ========== 功能函数：准备多个系统，每个系统多个样本（增加到 30 个样本） ==========

def prepare_multi_system_inputs():
    # 定义每个系统的化学式公式
    systems = {
        'system1': [f'Li{27 - i}Ni{27 + i}O54' for i in range(20)],  # system1: 30 个样本
        'system2': [f'Li{26 - i}Ni27Co{i}O54' for i in range(1, 21)],  # system2: 30 个样本
        'system3': [f'Li27Ni{26 - i}Co{i}O54' for i in range(1, 21)],  # system3: 30 个样本
        'system4': [f'Li27Ni{26 - i}Mn{i}O54' for i in range(1, 21)],  # system4: 30 个样本
        'system5': [f'Li27Ni27O{53 - i}' for i in range(20)],  # system5: 30 个样本
    }

    X_systems, y = [], []  # 初始化样本数据和目标值列表

    # 遍历每个系统
    for name, formulas in systems.items():
        X_list = []  # 存储每个系统的特征
        for i, formula in enumerate(formulas):
            # 将化学式转化为 DataFrame 以便进行特征提取
            df = pd.DataFrame({'formula': [formula]})
            # 使用 StrToComposition 将化学式转换为 Composition 类型
            df = StrToComposition().featurize_dataframe(df, col_id='formula')
            # 使用 ElementProperty 提取化学成分的属性（特征）
            featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
            df = featurizer.featurize_dataframe(df, col_id='composition')
            # 提取特征（去掉 'formula' 和 'composition' 列）
            X = df.drop(columns=['formula', 'composition']).values[0]
            X_list.append(X)
            # 对应的目标值，目标值随着样本索引的增加线性增长
            y.append(0.4 + 0.01 * i)
        X_systems.append(np.array(X_list))  # 将当前系统的特征加入 X_systems 中

    return X_systems, np.array(y)  # 返回所有系统的特征和目标值


# ========== 主程序入口 ==========

if __name__ == '__main__':
    # 初始化 SHAP
    shap.initjs()

    # 准备输入数据和目标值（将样本数增加至 30）
    X_systems, y = prepare_multi_system_inputs()

    # 对每个系统的输入特征进行独立标准化
    scalers = [StandardScaler().fit(X) for X in X_systems]  # 为每个系统的特征拟合一个标准化器
    X_scaled = [scalers[i].transform(X_systems[i]) for i in range(5)]  # 对每个系统的数据进行标准化处理

    # 合并不同系统的特征
    X_total = list(zip(*X_scaled))  # 转置，使得 X_total 每个元素包含不同系统的特征
    X_input = [np.array([x[i] for x in X_total]) for i in range(5)]  # 生成最终的输入数据

    # 定义输入层，每个输入对应一个系统的特征
    inputs = [Input(shape=(X.shape[1],), name=f'input_system{i + 1}') for i, X in enumerate(X_scaled)]

    processed = []
    # 对每个输入进行处理
    for inp in inputs:
        x = Dense(128, activation='relu', kernel_regularizer='l2')(inp)  # 全连接层
        x = BatchNormalization()(x)  # 批归一化
        x = Dropout(0.4)(x)  # Dropout
        processed.append(x)

    # 合并所有处理过的特征
    merged = Concatenate(name='merged_features')(processed)
    x = Dense(256, activation='relu', kernel_regularizer='l2')(merged)  # 全连接层
    x = Dropout(0.4)(x)  # Dropout
    x = Dense(128, activation='relu', kernel_regularizer='l2')(x)  # 全连接层
    x = Dropout(0.4)(x)  # Dropout
    output = Dense(1)(x)  # 输出层

    # 构建模型
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])  # 编译模型

    # 设置 EarlyStopping，防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # 训练模型
    history = model.fit(X_input, y, epochs=20, batch_size=2, validation_split=0.2, callbacks=[early_stopping])

    # 创建输出目录
    os.makedirs("training_output", exist_ok=True)
    # 绘制训练过程的损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练 Loss')
    plt.plot(history.history['val_loss'], label='验证 Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练过程 Loss 曲线")
    plt.legend()
    plt.grid(True)
    # 保存损失曲线图
    plt.savefig("training_output/loss_curve.png")
    plt.close()
    print("训练 Loss 曲线图已保存为 training_output/loss_curve.png")

    # 测试模型（使用前 10 个样本）
    X_test = [X[:10] for X in X_scaled]  # 选择测试集数据（前 10 个样本）
    X_test_total = list(zip(*X_test))  # 转置
    X_test_input = [np.array([x[i] for x in X_test_total]) for i in range(5)]  # 生成测试输入数据
    y_test = y[:10]  # 选择对应的目标值

    # 预测并计算评估指标
    y_pred = model.predict(X_test_input)
    print(f"均方误差 (MSE): {mean_squared_error(y_test, y_pred):.4f}")
    print(f"平均绝对误差 (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    # 保存模型
    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/battery_model.h5")
    print("模型已保存为 saved_model/battery_model.h5")

    # 获取合并层的输出，用于生成简化模型
    intermediate_model = Model(inputs=model.inputs, outputs=model.get_layer('merged_features').output)
    X_feat = intermediate_model.predict(X_test_input)

    # 构建简化模型，使用 Model 来定义模型
    simplified_input = Input(shape=(X_feat.shape[1],))  # 通过合并层的输出形状作为输入
    x = Dense(256, activation='relu', kernel_regularizer='l2')(simplified_input)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.4)(x)
    output = Dense(1)(x)

    simplified_model = Model(inputs=simplified_input, outputs=output)  # 使用 Model 来定义模型
    simplified_model.compile(optimizer='adam', loss='mse')

    # 训练简化模型
    simplified_model.fit(X_feat, y_test, epochs=10, verbose=0)

    # ========== 生成 SHAP 图片功能 ==========

    # 使用 SHAP 进行模型解释
    explainer = shap.KernelExplainer(simplified_model.predict, X_feat)
    shap_values = explainer.shap_values(X_feat)

    # 生成 SHAP 概要图 (Summary plot)，显示所有特征的影响力
    shap.summary_plot(shap_values, X_feat, max_display=20)

    # 创建 SHAP 输出目录并保存概要图
    os.makedirs("shap_output", exist_ok=True)
    plt.savefig("shap_output/summary_plot.png")
    plt.close()  # 关闭当前图形，以避免过多图形显示

    print("✅ SHAP 概要图已保存为 shap_output/summary_plot.png")

    # 修复 force plot 为空问题
    sample_idx = 0  # 选择第一个样本进行解释
    shap_arr = np.array(shap_values)
    mean_abs_shap = np.abs(shap_arr).mean(axis=1)[sample_idx]
    top_20_indices = np.argsort(mean_abs_shap)[-20:][::-1]  # 获取贡献最大的20个特征
    top_feat_values = X_feat[sample_idx][top_20_indices]
    top_shap_values = shap_arr[sample_idx][top_20_indices]

    # 创建 SHAP 输出目录并保存 force plot
    if len(top_shap_values) > 0:
        force_plot = shap.force_plot(
            explainer.expected_value,
            top_shap_values,
            top_feat_values,
            feature_names=[f"特征{i + 1}" for i in top_20_indices],
            matplotlib=False
        )
        shap.save_html("shap_output/force_plot_sample0_top20.html", force_plot)
        print("✅ force plot 成功生成并保存。")
    else:
        print("⚠️ 无法生成 force plot：SHAP 值为空，请检查输入数据或索引逻辑。")




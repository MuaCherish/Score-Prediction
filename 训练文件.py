import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import random
import joblib
import datetime
import warnings
import os

# 忽略警告
warnings.simplefilter("ignore", UserWarning)

######################################################

# 读取数据集
data = pd.read_csv('数据/特征/final.csv')
print("----------------------------\n")
print(">>>数据集已载入\n")
print("----------------------------\n")

# 提取特征和目标变量
features = data[['Release Date', 'Number of Reviews', 'Plays', 'Playing', 'Wishlist']]
label = data['Rating']

# 训练集测试集分割
train_idx = int(0.7 * len(data))
train_features = features[:train_idx]
train_label = label[:train_idx]

test_features = features[train_idx:]
test_label = label[train_idx:]

######################################################

# 修改一下测试集的索引
new_test_label = []  # 用于存储修改后的数值和索引的新列表
for index, value in enumerate(test_label):
    new_index = index
    new_value = value  # 直接获取元组中的值
    new_test_label.append((new_index, new_value))
test_label = new_test_label

# 保存一下测试集的value
actual_values = []
for index, value in test_label:
    actual_values.append(value)
test_label_value = actual_values

# 将test变为DataFrame
test_features = pd.DataFrame(test_features)
test_label_value = pd.DataFrame(test_label_value)

# 保存为CSV文件
test_features.to_csv('数据/数据集/测试_特征.csv', index=False)
test_label_value.to_csv('数据/数据集/测试_目标.csv', index=False)

print(">>>训练-测试集已分割(7:3)\n")
print(">>>测试集已保存\n")
print("----------------------------\n")

######################################################

# 构建随机森林回归模型
Rating_predict_model = RandomForestRegressor(
    n_estimators=500,
    max_features=2,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=1,
    min_impurity_decrease=0,
    bootstrap=True,
    oob_score=True,
    random_state=10,
    criterion='squared_error'
)
print(">>>随机森林回归模型已建立\n")
print("----------------------------\n")

######################################################

# # 初始训练集为空
# X_train = np.empty((0, features.shape[1]))
# y_train = np.empty(0)
#
# # 训练模型
# r2_scores = []
# print(">>>训练模型阶段:\n")
# for i in range(500):
#     # 读取第i条训练数据
#     new_x = features.iloc[i]
#     new_y = label.iloc[i]
#
#     # 添加到训练集
#     X_train = np.concatenate((X_train, new_x.values.reshape(1, -1)), axis=0)
#     y_train = np.concatenate((y_train, np.array([new_y])), axis=0)
#
#     # 训练模型并在测试集上评估
#     Rating_predict_model.fit(X_train, y_train)
#     predict = Rating_predict_model.predict(test_features)
#     r2 = r2_score(test_label_value, predict)
#     r2_scores.append(r2)
#     print(f"已训练{i}次,当前训练集量为{len(X_train)}条,模型拟合度为{r2*100:.3f}%")
#
#     # # 停止条件
#     # if r2 >= 0.5723:
#     #     break
#
# print("\n>>>模型训练完成!\n")
#
# plt.plot(r2_scores)
# plt.xlabel('data number')
# plt.ylabel('scores')
# plt.title('r2_scores')
# plt.show()

######################################################

# 训练模型
Rating_predict_model.fit(train_features, train_label)

# 预测和评价
predict = Rating_predict_model.predict(test_features)
predict_df = pd.DataFrame(predict)

######################################################

# # 将模型保存到本地
# now = datetime.datetime.now()
# model_filename = now.strftime("%Y%m%d_%H%M%S") + ".pkl"
#
# # 训练Your_model并得到模型对象Rating_predict_model
# joblib.dump(Rating_predict_model, '数据/模型/' + model_filename)
#
# print(">>>模型已保存到./数据/模型文件夹\n")
# print("----------------------------\n")

######################################################

# 评估
print(">>>进入评估环节：")

# 均方误差评估
mse = mean_squared_error(test_label_value, predict)
print(f'随机森林回归模型均方误差评估: {mse * 100:.2f}%')

# 绝对误差评估
mae = mean_absolute_error(test_label_value, predict)
print(f'随机森林回归模型绝对误差评估: {mae * 100:.2f}%')

# 拟合程度
r2 = r2_score(test_label_value, predict)
print(f'随机森林回归模型拟合度评估: {r2 * 100:.2f}%')

######################################################

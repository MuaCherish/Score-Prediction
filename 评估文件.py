import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os
import joblib

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, mean_absolute_error
from sklearn.metrics import r2_score

# 初始化
test_features = pd.read_csv('数据/数据集/测试_特征.csv')
test_label = pd.read_csv('数据/数据集/测试_目标.csv')

# 在模型文件夹中找到最新生成的模型文件
model_dir = '数据/模型'
model_files = os.listdir(model_dir)
model_files.sort()
latest_model_file = model_files[-1]
latest_model_path = os.path.join(model_dir, latest_model_file)
Rating_predict_model = joblib.load(latest_model_path)
print("----------------------------\n")
print(">>>模型已加载\n")
print("----------------------------\n")

# 预测和评价
predict = Rating_predict_model.predict(test_features)
predict_df = pd.DataFrame(predict)

######################################################

# 评估
print(">>>进入评估环节：")

# 均方误差评估
mse = mean_squared_error(test_label, predict)
print(f'随机森林回归模型均方误差评估: {mse*100:.2f}%')

# 绝对误差评估
mae = mean_absolute_error(test_label, predict)
print(f'随机森林回归模型绝对误差评估: {mae * 100:.2f}%')

# 拟合程度
r2 = r2_score(test_label, predict)
print(f'随机森林回归模型拟合度评估: {r2 * 100:.2f}%')
print("\n----------------------------\n")

######################################################

# # 设置窗口大小
# plt.figure(figsize=(12, 6))
#
# # 设置纵坐标最大值为 8
# plt.ylim(1, 6)
#
# # 预测集 predict 的折线图,蓝色
# plt.plot(predict, color='blue', label='predict')
#
# # 测试集 test_label 显示为点状,红色
# x = [i for i in range(len(test_label))]
# plt.scatter(x, test_label, color='red', label='test_label')
#
# # 备注
# plt.xlabel('number')
# plt.ylabel('Rating')
# plt.title('Predict VS test')
#
# # 显示图例
# plt.legend()
# plt.show()

######################################################

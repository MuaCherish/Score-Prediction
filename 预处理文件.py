import pandas as pd  # 数据处理库
import numpy as np  # 科学计算库
import os  # 文件操作

###############################################################

# 读取CSV文件
df = pd.read_csv('数据/数据集/原始数据集.csv', encoding='gbk')
print("-----------------------------------")
print("\n>>>原始数据集已载入\n")
print("-----------------------------------\n")

###############################################################

# 打乱训练集
df = df.sample(frac=1).reset_index(drop=True)
print(">>>数据集已打乱\n")
print("-----------------------------------\n")

###############################################################

# 删除指定行
df = df[df['Title'] != 'Elden Ring: Shadow of the Erdtree']
df = df[df['Title'] != 'Deltarune']

# 输出
print(">>>Elden Ring: Shadow of the Erdtree因为在当前数据集里未发布，已作为无用项删除")
print(">>>Deltarune因为在当前数据集里未发布，已作为无用项删除\n")
print("-----------------------------------\n")

###############################################################

# 获取 Rating 列
rating = df['Rating']

# 循环遍历 Rating 的值
for value in rating:
    # 如果值小于 2,删除该行
    if value < 2:
        df.drop(df[df['Rating'] == value].index, inplace=True)

print(">>>过于离散点已删除\n")
print("-----------------------------------\n")

###############################################################

# 获取列名列表
cols = df.columns

# 遍历每一列,检查缺失值
for col in cols:
    # 检查该列是否有NaN
    if df[col].isnull().any():
        # 打印该列名称和NaN个数
        print(col, "缺失项有：", df[col].isnull().sum(), "个")

        # 删除包含NaN的行
        df.drop(df[df[col].isnull()].index, inplace=True)

# 输出
print(">>>已删除缺失项\n")
print("-----------------------------------\n")

###############################################################

# # 删除 Title 列重复值对应的行
# df.drop_duplicates(subset=['Title'], inplace=True)
#
# # 输出
# print(">>>已删除重复项\n")
# print("-----------------------------------\n")

###############################################################

# 将修改后的数据保存
df.to_csv('数据/数据集/新数据集.csv', index=False)

# 输出
print(">>>数据已保存,新数据集已建立\n")
print("-----------------------------------\n")

###############################################################

# 加载需要修改的数字
K_Reviews = df['Number of Reviews']
K_Plays = df['Plays']
K_Playing = df['Playing']
K_Wishlist = df['Wishlist']

# 执行循环开始乘上1000
converted_data = []
for item in K_Reviews:
    if "K" in item:
        item = item.replace("K", "")
        item = float(item) * 1000
    converted_data.append(int(item))
dataframe = pd.DataFrame({'Number of Reviews': converted_data})
dataframe.to_csv("数据/特征/Number of Reviews.csv", index=False)

converted_data = []
for item in K_Plays:
    if "K" in item:
        item = item.replace("K", "")
        item = float(item) * 1000
    converted_data.append(int(item))
dataframe = pd.DataFrame({'Plays': converted_data})
dataframe.to_csv("数据/特征/Plays.csv", index=False)

converted_data = []
for item in K_Playing:
    if "K" in item:
        item = item.replace("K", "")
        item = float(item) * 1000
    converted_data.append(int(item))
dataframe = pd.DataFrame({'Playing': converted_data})
dataframe.to_csv("数据/特征/Playing.csv", index=False)

converted_data = []
for item in K_Wishlist:
    if "K" in item:
        item = item.replace("K", "")
        item = float(item) * 1000
    converted_data.append(int(item))
dataframe = pd.DataFrame({'Wishlist': converted_data})
dataframe.to_csv("数据/特征/Wishlist.csv", index=False)

# 输出
print(f"所有跟K相关的数字已修改到文件(./数据/特征)中")

###############################################################

# 将发布日期保存
Date = df['Release Date']
release_date_column = df['Release Date']
dataframe = pd.DataFrame(release_date_column, columns=['Release Date'])
dataframe.to_csv("数据/特征/Release Date.csv", index=False)

# 输出
print(f"Release Date已转到文件(../数据/特征)中")

###############################################################

# 将Rating保存
Rating = df['Rating']
release_Rating_column = df['Rating']
dataframe = pd.DataFrame(release_Rating_column, columns=['Rating'])
dataframe.to_csv("数据/特征/Rating.csv", index=False)

# 输出
print(f"Rating已转到文件(./数据/特征)中")
print(f"\n>>>所有特征均已保存完毕\n")
print("-----------------------------------\n")

###############################################################

# 定义文件路径和输出文件
path = '数据/特征/'
files = ['Rating.csv', 'Plays.csv', 'Playing.csv', 'Release Date.csv', 'Number of Reviews.csv', 'Wishlist.csv']
output = 'final.csv'

# 读取6个csv文件为DataFrame
df1 = pd.read_csv(os.path.join(path, files[0]))
df2 = pd.read_csv(os.path.join(path, files[1]))
df3 = pd.read_csv(os.path.join(path, files[2]))
df4 = pd.read_csv(os.path.join(path, files[3]))
df5 = pd.read_csv(os.path.join(path, files[4]))
df6 = pd.read_csv(os.path.join(path, files[5]))

# 确认df1列数后修改列名
col_nums1 = df1.shape[1]
new_cols1 = ['Release Date', 'Number of Reviews', 'Plays', 'Playing', 'Wishlist', 'Rating']
mapping1 = dict(zip(df1.columns, new_cols1))

# 其他df同样修改
frames = [df4, df5, df2, df3, df1, df6]

# 使用pandas concat()方法拼接6个df
result = pd.concat(frames, axis=1)

# 输出最终的csv文件
result.to_csv(os.path.join(path, output), index=False)

# 输出
print(f">>>所有特征集合于数据/特征/final.csv中\n")
print("-----------------------------------\n")

###############################################################



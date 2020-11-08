# -*- coding:utf-8 -*-
# @Time : 2020/10/23 11:46
# @Author: HuHuHu
# @File : test.py
# @Description : 房价预测

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import moxing as mox
mox.file.shift('os', 'mox')

# 显示所有列
pd.set_option('display.max_columns', None)
col_mappings = {
    'ocean_proximity': {
        'NEAR BAY': 0,
        '<1H OCEAN': 1,
        'INLAND': 2,
        'NEAR OCEAN': 3,
        'ISLAND': 4
    }
}


# 数据预处理
def data_processing():
    #读取csv文件
    try:
        with mox.file.File("s3://huhuhu/data/housing.csv", "r") as f:
            df = pd.read_csv(f)  # print(df)
    except mox.file.MoxFileReadException as e:
        print(e.resp)

    # df = pd.read_csv("housing.csv")

    # 方法一: 分类型特征编码
    # 字符串重新编码
    # 将映射结果映射到原始数据中
    # ocean_proximity_mapping = {label: idx for idx, label in enumerate(np.unique(df['ocean_proximity']))}
    # col_mappings['ocean_proximity'] = ocean_proximity_mapping

    # 方法二: One-Hot 编码
    # 将映射结果映射到原始数据中
    ocean_proximity_mapping = {label: idx for idx, label in enumerate(np.unique(df['ocean_proximity']))}
    col_mappings['ocean_proximity'] = ocean_proximity_mapping
    for col_mapping in col_mappings:
        df[col_mapping] = df[col_mapping].map(col_mappings[col_mapping])
    df = pd.get_dummies(df, columns=['ocean_proximity'], prefix_sep='_', dummy_na=False, drop_first=False)

    # 补充缺省值
    imputer = Imputer(strategy="median")
    df = imputer.fit_transform(df)
    df = pd.DataFrame(df)

    # 数据去均值和方差归一化
    standardscaler = StandardScaler()
    df = standardscaler.fit_transform(df)
    df = pd.DataFrame(df)
    data_partition(df)


# 数据集划分
def data_partition(df):
    x = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]]
    y = df.iloc[:, 8]
    x_train, x_varify, y_train, y_varify = train_test_split(x, y, test_size=0.2, random_state=1)
    # print(len(x_train), len(x_varify))
    Predict(x_train, x_varify, y_train, y_varify)


# 模型预测
def Predict(x_train, x_varify, y_train, y_varify):
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    house_predict = model.predict(x_varify)
    print("房价预测模型的权重矩阵为：", model.coef_)
    print("房价预测模型的偏置量为：", model.intercept_)
    print("lr的均方误差为：", mean_squared_error(y_varify,  house_predict))


if __name__ == '__main__':
    data_processing()

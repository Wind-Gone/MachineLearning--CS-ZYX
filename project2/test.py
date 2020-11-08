# -*- coding:utf-8 -*-
# @Time : 2020/11/3 23:11
# @Author: HuHuHu
# @File : test.py
# @Description :
import csv

import pandas as pd
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', None)
X = []
Y = []
Team_stat = []


# 15-16赛季球队实力数据特征初始化
def initialize_data(Tstat):
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat['Team'] = new_Tstat['Team'].str.replace('*', '')
    return new_Tstat.set_index('Team', inplace=False, drop=True)


# 15-16赛季数据处理
# @Description : 新增两列赢的球队和输的球队，将PTS列修改为客场得分和主场得分
def data_processing(ScheduleStat):
    df = pd.DataFrame(ScheduleStat)
    df.rename(columns={'PTS': 'VPTS'}, inplace=True)
    df.rename(columns={'PTS.1': 'HPTS'}, inplace=True)
    df['WTEAM'] = np.where(df['VPTS'] < df['HPTS'], df['Home/Neutral'], df['Visitor/Neutral'])
    df['LTEAM'] = np.where(df['VPTS'] > df['HPTS'], df['Home/Neutral'], df['Visitor/Neutral'])
    return df


# 建立数据集
def setData(data):
    print("正在建立数据集..")
    X = []
    for dataindex, row in data.iterrows():
        Wteam = row['WTEAM']
        Lteam = row['LTEAM']
        tip1 = []
        tip2 = []
        for key, value in Team_stat.loc[Wteam].items():
            tip1.append(value)
        for key, value in Team_stat.loc[Lteam].items():
            tip2.append(value)
        if random.random() > 0.5:
            X.append(tip1 + tip2)
            Y.append(0)
        else:
            X.append(tip2 + tip1)
            Y.append(1)
    return np.nan_to_num(X), Y


# 结果预测
def result_predition(team_1, team_2, model):
    tips = []

    # team 1，客场队伍
    for key, value in Team_stat.loc[team_1].items():
        tips.append(value)

    # team 2，主场队伍
    for key, value in Team_stat.loc[team_2].items():
        tips.append(value)

    tips = np.nan_to_num(tips)
    return model.predict_proba([tips])


# 主函数
if __name__ == '__main__':
    Tstat = pd.read_csv('data1.csv')
    Team_stat = initialize_data(Tstat)

    ScheduleStat = pd.read_csv('data2.csv')
    ScheduleStat = data_processing(ScheduleStat)
    X, Y = setData(ScheduleStat)

    # 训练网络模型
    print("正在训练网络模型，共有%d个比赛样例" % len(X))
    model = linear_model.LogisticRegression(solver='liblinear')
    model.fit(X, Y)

    # 利用10折交叉验证计算训练正确率
    print("利用10折交叉验证计算训练正确率")
    print(cross_val_score(model, X, Y, cv=10, scoring='accuracy', n_jobs=-1).mean())

    # 写入结果集
    result = []
    ResultStat = pd.read_csv('data3.csv')
    ResultStat = ResultStat.replace()
    # print(ResultStat)
    for index, row in ResultStat.iterrows():
        team1 = row['Visitor']
        team2 = row['Home']
        # print(team1)
        # print(team2)
        pred = result_predition(team1, team2, model)
        prob = pred[0][0]
        if pred[0][0] > 0.5:
            result.append([row["Date"], row["Start (ET)"], team1, team2, pred[0][0]])
        else:
            result.append([row["Date"], row["Start (ET)"], team2, team1, 1 - pred[0][0]])

    with open(r'data3.csv', 'a+', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Start (ET)', 'Winner', 'Loser', 'Probability'])
        writer.writerows(result)
        print('预测结束')

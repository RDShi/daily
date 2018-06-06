# -*- coding: utf-8 -*-
"""
Created on Tue May 13 20:57:52 2018

@author: li
该文件用于探索性分析，绘制各种用于描述数据的图像
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


if __name__ == '__main__':
    # 参数配置
    filename = 'data/train4.csv'
    place_list = [6, 7, 8] # 选取的地点
    all_place_list = list(range(1, 34)) # 所有地点的列表
    year = 2017         # 样本的年份
    date_start = 20170701  # 截取日期的起点
    is_save = True         # 是否保存图像
    
    # 判断是否为周末
    def is_weekend(x):
        """
        [in] x: int类型：20181013
        [out] bool类型：是否为周末
        """
        x = datetime.strptime(str(x), '%Y%m%d')
        if x.weekday() + 1 in (6, 7):
            return True
        else:
            return False
    
    # 读取数据
    data = pd.read_csv(filename, dtype=object, header=None)
    data[0] = data[0].apply(pd.to_numeric)
    data[2] = data[2].apply(pd.to_numeric)
    n_data = data.shape[0]
    place_1, place_2, place_3 = place_list[0], place_list[1], place_list[2]
    data_1 = pd.Series(data[data[0]==place_1][2])
    data_1.index = data[data[0]==place_1][1].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H'))
    data_2 = pd.Series(data[data[0]==place_2][2])
    data_2.index = data[data[0]==place_2][1].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H'))
    data_3 = pd.Series(data[data[0]==place_3][2])
    data_3.index = data[data[0]==place_3][1].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H'))
    
    
    # 图一：给定地点，绘制关于人数的时序图
    fig = plt.figure(figsize=(12, 5))
    fig.subplots_adjust(wspace=None, hspace=None)
    ax = fig.add_subplot(3, 1, 1)
    ax.set_title('Place-'+'6, 7, 8'+' Time Series', fontproperties='SimHei', fontsize=13)
    plt.subplot(311)
    data_1[:744].plot(label='Place'+str(place_1), style='r-', use_index=True, linewidth=0.3)
    data_1[744:1488].plot(style='g-', use_index=True, linewidth=0.3)
    data_1[1488:2208].plot(style='b-', use_index=True, linewidth=0.3)
    data_1[2208:2951].plot(style='y-', use_index=True, linewidth=0.3)
    data_1[2951:].plot(style='k-', use_index=True, linewidth=0.3)
    plt.xticks(['2017-07-01 00','2017-08-01 00','2017-09-01 00','2017-10-01 00','2017-11-01 00'],['7-1','8-1','9-1','10-1','11-1'])
    plt.tick_params(axis='x',width=2,colors='k')
    plt.tick_params(axis='y',width=2,colors='k')
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['bottom'].set_visible(False) #去掉下边框
    ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框
    # plt.legend(loc='upper left')
    
    ax = fig.add_subplot(3, 1, 2)
    # ax.set_title('Place-'+str(place_2)+' Time Series', fontproperties='SimHei', fontsize=12)
    data_2[:744].plot(label='Place'+str(place_2), style='r-', use_index=True, linewidth=0.3)
    data_2[744:1488].plot(style='g-', use_index=True, linewidth=0.3)
    data_2[1488:2208].plot(style='b-', use_index=True, linewidth=0.3)
    data_2[2208:2951].plot(style='y-', use_index=True, linewidth=0.3)
    data_2[2951:].plot(style='k-', use_index=True, linewidth=0.3)
    # plt.ylabel('人次/位', fontproperties='SimHei', fontsize=12)
    plt.xticks(['2017-07-01 00','2017-08-01 00','2017-09-01 00','2017-10-01 00','2017-11-01 00'],['7-1','8-1','9-1','10-1','11-1'])
    plt.tick_params(axis='x',width=2,colors='k')
    plt.tick_params(axis='y',width=2,colors='k')
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['bottom'].set_visible(False) #去掉下边框
    ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框
    
    ax = fig.add_subplot(3, 1, 3)
    # ax.set_title('Place-'+str(place_3)+' Time Series', fontproperties='SimHei', fontsize=13)
    data_3[:744].plot(label='Place'+str(place_3), style='r-', use_index=True, linewidth=0.3)
    data_3[744:1488].plot(style='g-', use_index=True, linewidth=0.3)
    data_3[1488:2208].plot(style='b-', use_index=True, linewidth=0.3)
    data_3[2208:2951].plot(style='y-', use_index=True, linewidth=0.3)
    data_3[2951:].plot(style='k-', use_index=True, linewidth=0.3)
    plt.xlabel('时间/小时', fontproperties='SimHei', fontsize=12)
    # plt.ylabel('人次/位', fontproperties='SimHei', fontsize=12)
    plt.yticks([0, 5000])
    plt.xticks(['2017-07-01 00','2017-08-01 00','2017-09-01 00','2017-10-01 00','2017-11-01 00'],['7-1','8-1','9-1','10-1','11-1'])
    plt.tick_params(axis='x',width=2,colors='k')
    plt.tick_params(axis='y',width=2,colors='k')
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['bottom'].set_visible(False) #去掉下边框
    ax.spines['left'].set_visible(False) #去掉左边框
    ax.spines['right'].set_visible(False) #去掉右边框
    if is_save:
        ts_path = 'fig/时序图{}-{}-{}'.format(str(place_1), str(place_2), str(place_3))
        if os.path.exists(ts_path):
            raise OSError('时序图文件已存在！')
        plt.savefig(ts_path)
    
    # 图二
    data_1.index = data[data[0]==place_1][1]
    data_2.index = data[data[0]==place_1][1]
    data_3.index = data[data[0]==place_1][1]

    for place in all_place_list:
        data1 = data[data[0]==place].reindex(columns=[1, 2])
        n_data1 = data1.shape[0]
        data1_1 = np.zeros([n_data1, 4], dtype=object)
        data1_1[:, 0:2] = data1
        for i in range(n_data1):
            data1_1[i, 2] = int(str(year) + data1_1[i, 0][:2] + data1_1[i, 0][2: 4])
            data1_1[i, 3] = int(data1_1[i, 0][4: 6])
        data1_2 = pd.DataFrame(data1_1[:, 1: 4], dtype='int').reindex(columns=[2, 1, 0])
        date1_list = data1_2[1].unique().astype('int')
        n_date1 = date1_list.shape[0]
        ss = list(date1_list)
        ss.insert(0, 'key')
        data1_3 = np.array(data1_2)
        data1_4 = pd.DataFrame(np.zeros((24, n_date1), dtype='int'), columns=date1_list, index=range(24))
        data1_4 = data1_4 - 1   # 默认空值为-1
        for i in range(n_data1):
            hour = data1_3[i, 0]
            date = data1_3[i, 1]
            data1_4.loc[hour, date] = data1_3[i, 2]
        new_date1_list = []
        for i in date1_list:
            if i > date_start:
                new_date1_list.append(i)
        data1_4 = data1_4.reindex(columns=new_date1_list)
        # 判断周末
        date1_weekend = list(map(lambda x: x if is_weekend(x) else None, new_date1_list))
        date_weekend = []
        for i in date1_weekend:
            if i:
                date_weekend.append(i)
        data_weekend = data1_4.reindex(columns=date_weekend)
        data_weekend = data_weekend[data_weekend>=0]
        date1_weekday = list(map(lambda x: x if not is_weekend(x) else None, new_date1_list))
        date_weekday = []
        for i in date1_weekday:
            if i:
                date_weekday.append(i)
        data_weekday = data1_4.reindex(columns=date_weekday)
        data_weekday = data_weekday[data_weekday>=0]
        data_weekend_mean = data_weekend.mean(axis=1)
        data_weekday_mean = data_weekday.mean(axis=1)

        # 画图2
        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(211)
        ax.set_title('节假日一天内各小时人流量情况, Place:{}'.format(str(place_1)), fontproperties='SimHei', fontsize=14)
        ax.set_ylabel('人数', fontproperties='SimHei')
        data_weekend.plot(ax=ax, label='weekends', use_index=False, grid=True, legend=False, style='c--', xticks=range(24))
        plt.plot(range(24), data_weekend_mean, 'r-', linewidth=5)
        ax = fig.add_subplot(212)
        # ax.set_title('非周末一天内各小时人流量情况, Place:{},RMSE:{}'.format(str(place_1), str(res1)), fontproperties='SimHei', fontsize=14)
        ax.set_title('工作日一天内各小时人流量情况, Place:{}'.format(str(place_1)), fontproperties='SimHei', fontsize=14)

        ax.set_ylabel('人数', fontproperties='SimHei')
        ax.set_xlabel('时间/h', fontproperties='SimHei')
        data_weekday.plot(ax=ax, label='weekdays', use_index=False, grid=True, legend=False, style='c--', xticks=range(24))
        plt.plot(range(24), data_weekday_mean, 'r-', linewidth=5)
        plt.show()
        
    # 对所有地点绘制图二   


        



    
        
    
    
    
    

        
    
    
    
    
    
    


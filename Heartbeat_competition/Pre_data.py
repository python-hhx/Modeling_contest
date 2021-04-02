# -*- coding: utf-8 -*-
# @Time    : 2021/4/1 10:50
# @Author  : HHX
# @FileName: Pre_data.py
# @Software: PyCharm

from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters
import pandas as pd
from tqdm import tqdm
import functools
import time
import os
from logger import Logger
import traceback


def print_time(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        now_time1 = int(time.time())
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
        print("开始时间为{}".format(now_time))
        func(*args, **kwargs)
        end_time1 = int(time.time())
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
        print("结束时间为{}".format(end_time))
        print("一共消耗{}分".format((end_time1 - now_time1) / 60))

    return inner


class Pre_df:
    """"
    此部分主要进行数据预处理，并进行特征工程
    """
    def __init__(self):
        self.df_train = pd.read_csv(r'./df_file/train.csv', encoding='utf-8')
        self.df_test = pd.read_csv(r'./df_file/testA.csv', encoding='utf-8')
        self.df_train.set_index('id', inplace=True)
        self.df_test.set_index('id', inplace=True)
        self.log = Logger('Heartbeat_competition').get_log()

    @print_time
    def data_processing(self):
        if 'Train_done.csv' not in os.listdir(r'./df_file'):
            all_columns_name = ['id']
            all_columns_name = all_columns_name + ['id_%s' % i for i in
                                                   range(len(self.df_train.iloc[0, 0].split(',')))] + ['label']
            all_columns_name_ = all_columns_name + ['id_%s' % i for i in
                                                    range(len(self.df_train.iloc[0, 0].split(',')))]
            df_train_new = pd.DataFrame(columns=(all_columns_name))
            df_test_new = pd.DataFrame(
                columns=(['id'] + ['id_%s' % i for i in range(len(self.df_train.iloc[0, 0].split(',')))]))

            for i in tqdm(range(205)):
                df_train_new['id_%s' % i] = self.df_train['heartbeat_signals'].apply(lambda x: float(x.split(',')[i]))
                df_test_new['id_%s' % i] = self.df_test['heartbeat_signals'].apply(lambda x: float(x.split(',')[i]))

            df_train_new['id'] = self.df_train.index.tolist()
            df_test_new['id'] = self.df_test.index.tolist()
            df_train_new['label'] = self.df_train['label']
            df_train_new.to_csv('./df_file/Train_done.csv')
            df_test_new.to_csv('./df_file/Test_done.csv')
        else:
            print('asd')
            df_train_new = pd.read_csv('./df_file/Train_done.csv', index_col=False)
            df_test_new = pd.read_csv('./df_file/Test_done.csv', index_col=False)
        return df_train_new, df_test_new

    @print_time
    def tsfresh_processing(self):
        try:
            df_train_new = pd.read_csv('./df_file/Train_done.csv')
            df_train_new.set_index('id', inplace=True)
            df_test_new = pd.read_csv('./df_file/Test_done.csv')
            df_test_new.set_index('id', inplace=True)
            print(df_train_new)
        except Exception as ex:
            traceback.print_exc()
            self.log.error('tsfresh_processing failed: %s' % (str(ex)))
        def batch_train_tsfresh(data):
            new_df = pd.DataFrame()
            all_data_len = len(data)
            n_splits = all_data_len // 10
            for i in range(10):
                print('*************************正在进行第%s批数据扩展,共10批*******************' % (i + 1))
                if i == 9:
                    df = data.loc[n_splits * i:, :]
                else:
                    df = data.loc[n_splits * i:n_splits * (i + 1), :]
                df = df.stack().reset_index()
                settings = ComprehensiveFCParameters()
                extracted_features_ = extract_features(df, column_id="id", column_sort="level_1",
                                                       default_fc_parameters=settings)
                new_df = pd.concat([new_df, extracted_features_])
            return new_df

        if 'extracted_features_train.csv' not in os.listdir('./df_file'):
            df_train_new_1 = df_train_new.iloc[:, :-1]
            extracted_features_1 = batch_train_tsfresh(df_train_new_1)
            df_col_ = extracted_features_1.isnull().sum().reset_index()
            all_need_col = df_col_.loc[df_col_[0] == 0, 'index'].tolist()
            extracted_features_train = extracted_features_1[all_need_col]

            df_test_new_1 = df_test_new
            df_test_new_1 = df_test_new_1.stack().reset_index()
            settings = ComprehensiveFCParameters()
            extracted_features_2 = extract_features(df_test_new_1, column_id="id", column_sort="level_1", default_fc_parameters=settings)
            extracted_features_test = extracted_features_2[all_need_col]

            extracted_features_train.to_csv('./df_file/extracted_features_train.csv')
            extracted_features_test.to_csv('./df_file/extracted_features_test.csv')
        else:
            extracted_features_train = pd.read_csv('./df_file/extracted_features_train.csv')
            extracted_features_test = pd.read_csv('./df_file/extracted_features_test.csv')
        return extracted_features_train, extracted_features_test


if __name__ == '__main__':
    pre_df = Pre_df()
   # pre_df.data_processing()
    pre_df.tsfresh_processing()


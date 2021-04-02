# -*- coding: utf-8 -*-
# @Time    : 2021/4/1 11:05
# @Author  : HHX
# @FileName: Build_model.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def result_compute(y_pred, y_true):
    return sum(sum(abs(np.array(y_pred) - np.array(y_true))))


def to_one_hot(labels, dimension=4):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, int(label)] = 1.0
    return results


def plot_plt(val_mae, ep_num):
    average_mae_history = [np.mean([x[i] for x in val_mae]) for i in range(ep_num)]
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.savefig('./mae.jpg')
    plt.show()


df_train_new = pd.read_csv('./df_file/Train_done.csv')
df_test_new = pd.read_csv('./df_file/Test_done.csv')


"""
然后再开多线程对模型进行对比
"""


class Deep_learning_model:

    def __init__(self, input_dim):
        self.model = models.Sequential()
        self.model.add(layers.Dense(1024, activation='relu', input_shape=(input_dim,)))
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(4, activation='softmax'))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])


    def run(self, epoch_num, batch_size):
        # 利用k折验证
        fold = 5
        seed = 2021

        kf = KFold(n_splits=fold, shuffle=True, random_state=seed)
        # all_mae_histories = []
        # all_mae_histories_train = []

        train_data = np.array(df_train_new.iloc[:, 1:-1])
        tran_targets = np.array(df_train_new.iloc[:, -1])

        test_result = np.zeros((len(df_test_new), 4))


        for i, (train_index, val_index) in enumerate(kf.split(train_data)):
            print('************************第{}次交叉************************'.format(i + 1))
            x_train = train_data[train_index]
            y_train = to_one_hot(tran_targets[train_index])
            x_val = train_data[val_index]
            y_val = to_one_hot(tran_targets[val_index])
            history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epoch_num, batch_size=batch_size, verbose=1)
            y_pred = self.model.predict(x_val)
            test_result += self.model.predict(np.array(df_test_new.iloc[:, 1:]))
            print('第%s轮得结果为%s' % (i+1, result_compute(y_pred, y_val)))
            # mae_history = history.history['val_accuracy']
            # mae_hitory_train = history.history['accuracy']
            # all_mae_histories_train.append(mae_hitory_train)
            # all_mae_histories.append(mae_history)

        # plot_plt(all_mae_histories, 200)
        test_result = test_result/5
        self.to_save(test_result)
        return test_result


    def to_save(self, final_result):
        new_df = pd.DataFrame(final_result)

        for i in range(len(new_df)):
            all_list = new_df.iloc[i, :].tolist()
            max_num = max(all_list)
            new_list = list(map(lambda x: 0 if x < max_num else 1, all_list))
            new_df.loc[i] = new_list

        new_df['4'] = new_df[0] + new_df[1] + new_df[2] + new_df[3]
        new_df.to_csv('./df_file/deep_learning_final_result_0401.csv')


class LightGBM_model:

    def f1_score_vali(self, preds, data_vali):
        labels = data_vali.get_label()
        preds = np.argmax(preds.reshape(4, -1), axis=0)
        score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
        return 'f1_score', score_vali, True

    def run(self):

        test_pred = np.zeros((len(df_test_new), 4))

        x_train = df_train_new.iloc[:, 1:-1]
        y_train = df_train_new.iloc[:, -1]

        # 5折交叉验证
        folds = 5
        seed = 2021
        # shuffle 进行数据打乱
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

        for i, (train_index, valid_index) in enumerate(kf.split(x_train, y_train)):
            print('************************第{}次交叉************************'.format(i + 1))
            x_train_split, y_train_split, x_val, y_val = x_train.iloc[train_index], y_train.iloc[train_index], x_train.iloc[valid_index], y_train.iloc[valid_index]
            train_matrix = lgb.Dataset(x_train_split, label=y_train_split)
            valid_matrix = lgb.Dataset(x_val, label=y_val)

            params = {
                "num_leaves": 128,
                "metric": None,
                "objective": "multiclass",
                "num_class": 4,
                "nthread": 10,
                "verbose": -1,
            }

            model = lgb.train(params,
                              train_set=train_matrix,
                              valid_sets=valid_matrix,
                              num_boost_round=2000,
                              verbose_eval=100,
                              early_stopping_rounds=200,
                              feval=self.f1_score_vali)

            val_pred = model.predict(x_val, num_iteration=model.best_iteration)
            test_predict = model.predict(df_test_new.iloc[:, 1:], num_iteration=model.best_iteration)
            test_pred = test_pred + test_predict
            print('第%s轮得结果为%s' % (i + 1, result_compute(val_pred, to_one_hot(y_val))))
        test_pred = test_pred / 5
        self.to_save(test_pred)
        return test_pred

    def to_save(self, final_result):
        new_df = pd.DataFrame(final_result)

        for i in range(len(new_df)):
            all_list = new_df.iloc[i, :].tolist()
            max_num = max(all_list)
            new_list = list(map(lambda x: 0 if x < max_num else 1, all_list))
            new_df.loc[i] = new_list

        new_df['4'] = new_df[0] + new_df[1] + new_df[2] + new_df[3]
        new_df.to_csv('./df_file/lgb_final_result_0401.csv')

if __name__ == '__main__':
    # model1 = Deep_learning_model(205)
    # model1.run(500, 128)

    model2 = LightGBM_model()
    model2.run()





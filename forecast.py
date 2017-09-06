# -*- coding:utf-8 -*-

# @Author zpf

import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn import svm
# from libsvm.python.svmutil import *
# from libsvm.python.svm import *


class Forecast:
    train_data_path = ""
    test_data_path = ""

    all_train_array = np.array([])
    all_val_array = np.array([])
    all_test_array = np.array([])

    def __init__(self, train, test):
        self.train_data_path = train
        self.test_data_path = test
        self.read_data(self.train_data_path, self.test_data_path)

    def read_data(self, train, test):
        # 数据维度(321675, 93)
        train_csv_read = np.loadtxt(open(train), delimiter=",", dtype=np.string_)
        # test_csv_read = np.loadtxt(open(test), delimiter=",", dtype=np.string_)

        tag = train_csv_read[0, -1]
        seg_index = []
        for row in train_csv_read:
            if row[-1] != tag:
                # row[0]的结果如下，每个index表示一个era,从1->20
                # b'0',b'16995',b'34215',b'51555',b'69259',b'87329',b'105763',b'124301',b'140514',b'156591',b'172474',b'188087',b'203449'
                # b'218724',b'233944',b'249142',b'264125',b'278990',b'293501',b'307668'
                seg_index.append(int(row[0]))
                tag = row[-1]

        # 按era划分为20组，然后堆叠为一个大的ndarray
        train_csv_array = []
        i = 0
        while i < len(seg_index):
            if i < len(seg_index) - 1:
                part_train = train_csv_read[seg_index[i] + 1:seg_index[i + 1] + 1, :]
                train_csv_array.append(part_train)
            else:
                part_train = train_csv_read[seg_index[i] + 1:, :]
                train_csv_array.append(part_train)
            i += 1
        all_train_csv = np.array(train_csv_array)

        # 按era随机划分训练集,验证集, train : val = 7 : 3
        seed = list(range(0, 20))
        random.shuffle(seed)
        count = 0
        print("1--------------")
        while count < len(seed):
            # print(all_train_csv[seed[count]].shape)
            if count < 6:
                if count == 0:
                    self.all_val_array = all_train_csv[seed[count]]
                else:
                    self.all_val_array = np.vstack((self.all_val_array, all_train_csv[seed[count]]))
            else:
                if count == 6:
                    self.all_train_array = all_train_csv[seed[count]]
                else:
                    self.all_train_array = np.vstack((self.all_train_array, all_train_csv[seed[count]]))
            count += 1

        print(self.all_val_array.shape)
        print(self.all_train_array.shape)
        # all_train_array[:, [-3, -2]] = all_train_array[:, [-2, -3]]
        # all_train_array[:, [-3, -2]] = all_train_array[:, [-2, -3]]

    def svm_train(self):
        print("2----------------")
        train_features_array = self.all_train_array[:, 1:-3]
        train_features = train_features_array.astype(np.float64)
        # train_label_array包括最后三列的label, group, era
        train_label_array = self.all_train_array[:, -3]
        train_label = train_label_array.astype(np.float64)
        print("train_label: " + str(train_label.shape))
        val_features_array = self.all_val_array[:, 1:-3]
        val_features = val_features_array.astype(np.float64)
        # val_label_array包括最后三列的label, group, era
        val_label_array = self.all_val_array[:, -3]
        val_label = val_label_array.astype(np.float64)
        print("val_label: " + str(val_label.shape))

        # # 测试集
        # test_features_array = test_csv_read[1:, 1:-1]
        # test_features = test_features_array.astype(np.float64)
        # # test_label_array包括最后一列的group
        # test_label_array = test_csv_read[1:, -1]
        # test_label = test_label_array.astype(np.float64)
        print("3------------")
        clf = svm.SVC(probability=True)
        y = train_label[:5000]
        X = train_features[:5000]
        clf.fit(X, y)
        X1 = val_features[:5]
        c = clf.predict(X1)
        print(c)
        print(val_label[:5])
        d = clf.predict_proba(X1)
        print(d)


if __name__ == "__main__":
    train_data = "./train/stock_train_data_20170901.csv"
    test_data = "./test/stock_test_data_20170901.csv"
    XXX = Forecast(train_data, test_data)
    XXX.svm_train()
    pass

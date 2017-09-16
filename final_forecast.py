# -*- coding:utf-8 -*-

# @Author zpf

import csv
import time
import math
import random
import numpy as np
from sklearn import *
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


class Forecast:
    train_data_path = ""
    test_data_path = ""

    train_features = np.array([])
    train_label = np.array([])
    train_weight = np.array([])

    val_features = np.array([])
    val_label = np.array([])
    val_label_predict = np.array([])
    val_label_pro = np.array([])
    val_weight = np.array([])

    test_features = np.array([])
    test_label = np.array([])
    test_label_pro = np.array([])
    test_id = np.array([])

    def __init__(self, train, test):
        self.train_data_path = train
        self.test_data_path = test

    def read_train_data(self, train, classifier_num):
        # 数据维度(321675, 93)
        train_csv_read = np.loadtxt(open(train), delimiter=",", dtype=np.string_)

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
        print("数据读取完毕，开始划分训练验证集")
        # 按era随机划分训练集,验证集, train : val = 7 : 3
        all_train_array = np.array([])
        all_val_array = np.array([])

        seed = list(range(0, 20))
        random_num = 0
        temp_label_pro = np.array([])
        while random_num < 3:
            random.shuffle(seed)
            count = 0
            while count < len(seed):
                # print(all_train_csv[seed[count]].shape)
                if count < 5:
                    if count == 0:
                        all_val_array = all_train_csv[seed[count]]
                    else:
                        all_val_array = np.vstack((all_val_array, all_train_csv[seed[count]]))
                else:
                    if count == 5:
                        all_train_array = all_train_csv[seed[count]]
                    else:
                        all_train_array = np.vstack((all_train_array, all_train_csv[seed[count]]))
                count += 1
            self.fetch_train_val_data(all_train_array, all_val_array)

            x = self.train_features[:]
            y = self.train_label[:]
            z = self.train_weight[:]

            x1 = self.val_features[:]
            y1 = self.val_label[:]

            print("正在预测第" + str(random_num + 1) + "次")
            print(seed)

            base_clf = RandomForestClassifier(n_estimators=classifier_num, max_depth=4)
            clf = BaggingClassifier(base_clf, n_estimators=7)
            clf.fit(x, y, sample_weight=z)

            # 验证集预测
            self.val_label_predict = clf.predict(x1)
            self.val_label_pro = clf.predict_proba(x1)
            self.calculate_loss()
            # 训练集预测20次取均值
            # self.test_label = clf.predict(self.test_features[:])

            self.test_label_pro = clf.predict_proba(self.test_features[:])
            if random_num == 0:
                temp_label_pro = self.test_label_pro
            else:
                temp_label_pro += self.test_label_pro

            random_num += 1

        self.test_label_pro = temp_label_pro / random_num
        self.test_label_pro = self.test_label_pro[:, 1]

        with open("./results1.txt", 'w') as f:
            count_num = 0
            res_str = "id,proba" + "\n"
            f.writelines(res_str)
            while count_num < len(self.test_label_pro):
                res_str = str(int(self.test_id[count_num])) + "," \
                          + str(self.test_label_pro[count_num]) + "\n"
                f.writelines(res_str)
                count_num += 1

    def fetch_train_val_data(self, all_train_data, all_val_data):
        # all_train_array, all_val_array最后四列为weight，label, group, era
        train_features_array = all_train_data[:, 1:-4]
        self.train_features = train_features_array.astype(np.float64)
        train_label_array = all_train_data[:, -3]
        self.train_label = train_label_array.astype(np.float64)
        train_weight_array = all_train_data[:, -4]
        self.train_weight = train_weight_array.astype(np.float64)

        val_features_array = all_val_data[:, 1:-4]
        self.val_features = val_features_array.astype(np.float64)
        val_label_array = all_val_data[:, -3]
        self.val_label = val_label_array.astype(np.float64)
        val_weight_array = all_val_data[:, -4]
        self.val_weight = val_weight_array.astype(np.float64)

        # # 对所有features作归一化处理
        # temp_max_train = np.max(self.train_features)
        # temp_min_train = np.min(self.train_features)
        # temp_y_train = 1 / (temp_max_train - temp_min_train)
        # self.train_features = (self.train_features - temp_min_train) * temp_y_train
        # temp_max_val = np.max(self.val_features)
        # temp_min_val = np.min(self.val_features)
        # temp_y_val = 1 / (temp_max_val - temp_min_val)
        # self.val_features = (self.val_features - temp_min_val) * temp_y_val

    def read_test_data(self, test):
        test_csv_read = np.loadtxt(open(test), delimiter=",", dtype=np.string_)
        test_features_array = test_csv_read[1:, 1:-1]
        self.test_features = test_features_array.astype(np.float64)

        # # 归一化
        # temp_max_test = np.max(self.test_features)
        # temp_min_test = np.min(self.test_features)
        # temp_y_test = 1 / (temp_max_test - temp_min_test)
        # self.test_features = (self.test_features - temp_min_test) * temp_y_test

        test_id_array = test_csv_read[1:, 0]
        self.test_id = test_id_array.astype(np.float64)

    def calculate_loss(self):
        y1 = self.val_label[:]
        z1 = self.val_weight[:]

        correct_num = 0
        num = 0
        while num < len(y1):
            if y1[num] == self.val_label_predict[num]:
                correct_num += 1
            num += 1
        print("预测准确率为：" + str(correct_num) + "/" + str(len(y1))
              + "=" + str(correct_num / len(y1)))

        # 计算logloss
        self.val_label_pro = self.val_label_pro[:, 1]
        results = []
        results.append(y1)
        results.append(z1)
        results.append(self.val_label_pro)
        results = np.array(results).T
        result = results.tolist()

        logloss = 0
        val_samples_num = 0
        while val_samples_num < len(y1):
            value_each = result[0]
            yti = value_each[0]
            wi = value_each[1]
            ypi = value_each[2]
            temp_a = math.log(ypi)
            temp_b = math.log((1 - ypi))
            logloss += wi * (yti * temp_a + (1 - yti) * temp_b)

            val_samples_num += 1
        logloss = -logloss
        print("logloss: " + str(logloss))


if __name__ == "__main__":
    localtime = time.asctime(time.localtime())
    print("程序开始运行时间：" + str(localtime))

    train_data = "./train/stock_train_data_20170910.csv"
    test_data = "./test/stock_test_data_20170910.csv"

    XXX = Forecast(train_data, test_data)
    XXX.read_test_data(XXX.test_data_path)
    XXX.read_train_data(XXX.train_data_path, 8)

    localtime = time.asctime(time.localtime())
    print("程序结束运行时间：" + str(localtime))

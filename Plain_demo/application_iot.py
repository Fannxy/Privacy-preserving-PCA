"""
Experiments - Iot attack
Dataset: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT#

This plain-text version is for demonstration about the data integration horizationally.
"""

import numpy as np
import pandas as pd
import time
import scipy.io as scio
import random

from plain_preprocess import *
from jacobi import *
from PCA_application import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def dataload(name):
    """Data loading
    """
    gaf_list = ['tcp', 'udp', 'scan', 'combo', 'junk']
    mir_list = ['ack', 'scan', 'syn', 'udp', 'udpplain']

    benign = pd.read_csv('./Data/Iot_Attack/'+name+'/benign_traffic.csv')
    benign = min_max_scaler(benign.values)

    data_dict = {
        'begign': np.concatenate([benign, np.zeros((len(benign), 1))], axis=1), # with labels 0
    }

    for gaf_name in gaf_list:
        print("loading %s ... " %(gaf_name))
        try:
            data = pd.read_csv('./Data/Iot_Attack/'+name+'/gafgyt_attacks/'+gaf_name+'.csv')
        except Exception as e:
            print("No ", gaf_name)
            continue;
        data = min_max_scaler(data.values)
        data_dict.update({'gaf_'+gaf_name : np.concatenate([data, np.ones((len(data), 1))], axis=1)})
    
    for mir_name in mir_list:
        print("loading %s ... " %(mir_name))
        try:
            data = pd.read_csv('./Data/Iot_Attack/'+name+'/mirai_attacks/'+mir_name+'.csv')
        except Exception as e:
            print("No ", mir_name)
            continue;
        data = min_max_scaler(data.values)
        data_dict.update({'mir_'+mir_name : np.concatenate([data, np.ones((len(data), 1))+1], axis=1)})
    
    return data_dict

def model_evaluation(X_train, y_train, X_test, y_test, k=16):
    """Evaluate models with different dataset
    """
    print(">>>>>>> x.shape", X_train.shape)
    p_matrix, X_reduce = dimension_reduction(X_train, k=k)
    print("model training ...")
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=30, learning_rate=1)
    bdt.fit(X_reduce, y_train)
    print("fit succeed")

    X_test = np.dot(X_test, p_matrix)
    y_pred = bdt.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['benign', 'gafgyt', 'miari'], digits=4))


def model_evaluation_np(X_train, y_train, X_test, y_test, k=16):
    """Evaluate models with different dataset
    """
    print(">>>>>>> x.shape", X_train.shape)
    p_matrix, X_reduce = dimension_reduction_np(X_train, k=k)
    print("model training ...")
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=30, learning_rate=1)
    bdt.fit(X_reduce, y_train)
    print("fit succeed")

    X_test = np.dot(X_test, p_matrix)
    y_pred = bdt.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['benign', 'gafgyt', 'miari'], digits=4))


if __name__ == '__main__':

    # data loading
    name_list = ['Baby_monitor', 'Danmini_doorbell', 'Ecobee', 'Ennio_doorbell', 'Samsung_Webcam', 'PT_737E_Camera', 'PT_838_Security_Camera', 'SH_XCS7_1002_Security_Camera', 'SH_XCS7_1003_Security_Camera']
    L = len(name_list)
    train_list = {item:[] for item in name_list}
    test_list = {item:[] for item in name_list}

    # # # data preprocess <- used to generate data for cipher-text platform
    # R_list = []
    # v_list = []
    # N_list = []
    # for item in data_list:
    #     X = data_list[item][:, :-1]
    #     time_begin = time.time()
    #     R, v, N = find_rv(X)
    #     R_list.append(R)
    #     v_list.append(v)
    #     N_list.append(N)
    #     print("Dataset %s with shape: %d" %(item, N))
    #     time_end = time.time()
    #     time_preprocess_list[item] = time_end - time_begin
    #     print("Time : %.8f" %(time_preprocess_list[item]))

    #     data_save = np.concatenate([R, np.reshape(v, (len(v), 1))], axis=1)
    #     np.savetxt('./Data/Iot_Attack/Rv_data/'+item+'.txt', data_save, delimiter=',', fmt='%.4f')

    # # load from files
    # for item in name_list:
    #     data = np.loadtxt('./Data/Iot_Attack/Rv_data/'+item+'.txt', delimiter=',')
    #     R_list.append(data[:, :-1])
    #     v_list.append(data[:, -1])
    # N_list = [1098677, 1018298, 835876]

    # # Data fusion
    # cov_f = join_cov(R_list, v_list, N_list)
    # print(">>>>>>>>>>>>", np.min(cov_f*cov_f))
    # print("sums: ", np.sum(cov_f**2))
    # evals, evecs, iter_num = jacobi_loop(cov_f)
    # # print(evals)
    # print(np.argsort(evals)[:10])
    # print(evals[np.argsort(evals)])


    # Demonstration about the data integration
    data_list = scio.loadmat('./Data/Iot_Attack/raw_data.mat')
    # split train and test data
    for item in name_list:
        data_train, data_test = train_test_split(data_list[item], test_size=0.05)
        train_list.update({item:data_train})
        test_list.update({item:data_test})
    test_set = np.concatenate([test_list[item] for item in name_list])
    X_test_final, y_test = test_set[:, :-1], test_set[:, -1]


    # benchmark - fused p_matrix
    print(">>>>> benchmark ")
    train_set = np.concatenate([train_list[item] for item in name_list])
    X_train, y_train = train_set[:, :-1], train_set[:, -1]
    np.savetxt('./Data/Iot_Attack/9-party.txt', X_train)
    #model_evaluation_np(X_train, y_train, X_test_final, y_test, k=20)


    # print(">>>>>>>>>>>> Single parties")
    # for item in name_list:  
    #     print(">>>>> ITEM: ", item)
    #     print(train_list[item].shape)
    #     X_train, y_train = train_list[item][:, :-1], train_list[item][:, -1]
    #     #model_evaluation_np(X_train, y_train, X_test_final, y_test, k=20)
    

    # print(">>>>>>>>>> Three parties")
    # S = 3
    # for i in range(1):
    #     print("===== party-", i)
    #     parties_list = random.sample(name_list, S)
    #     data = np.concatenate([train_list[item] for item in parties_list])
    #     print(data.shape)
    #     X, y = data[:, :-1], data[:, -1]
    #     np.savetxt('./Data/Iot_Attack/3-party.txt', X)
    #     #model_evaluation_np(X, y, X_test_final, y_test, k=20)


    # print(">>>>>>>>>> Five parties")
    # S = 5
    # for i in range(1):
    #     print("===== party-", i)
    #     parties_list = random.sample(name_list, S)
    #     data = np.concatenate([train_list[item] for item in parties_list])
    #     X, y = data[:, :-1], data[:, -1]
    #     #model_evaluation_np(X, y, X_test_final, y_test, k=20)
    #     np.savetxt('./Data/Iot_Attack/5-party.txt', X, fmt='%.4f')


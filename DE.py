import os
import platform
import sys
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.utils import shuffle
from GMM import GMM_RSMOTE, load_data, show_sample, load_bug_data
from Python_ELM.elm_orgign import GenELMClassifier
from Python_ELM.random_layer import MLPRandomLayer
from sklearn.preprocessing import label_binarize, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1)

if platform.system() == 'Linux':
    BASE_PATH = '/home/li/PycharmProjects/20-gmm/keel'
elif platform.system() == 'Windows':
    BASE_PATH = r'D:/ownCloud/share/newdata/UCI'


# def runELM(attr, label, attr_test, label_test):
#     rt = RandomForestClassifier(
#         random_state=0, n_estimators=1, max_depth=10)
#     rt.fit(attr, label)
#     score = rt.score(attr_test, label_test)
#     return score

def runELM(attr, label, attr_test, label_test, n_num):
    n1 = 0
    n0 = 0
    # print(label_test,len(label_test))
    for i in range(0, len(label)):
        if label[i] == 1:
            n1 += 1
        else:
            n0 += 1
    print("\n")
    print("训练集：   少数%d，多数%d" % (n1, n0))
    # print("test",label_test, len(label_test))
    n1 = 0
    n0 = 0
    for i in range(0, len(label_test)):
        if label_test[i] == 1:
            n1 += 1
        else:
            n0 += 1
    print("测试集：   少数%d，多数%d" % (n1, n0))
    onehotencoder = OneHotEncoder(sparse=False)
    label = onehotencoder.fit_transform(np.mat(label).T)
    label_test = onehotencoder.fit_transform(np.mat(label_test).T)
    # print(label.ravel(),len(label))
    # print(label_test.ravel())
    # w = np.random.uniform(-1,1,(N,n_num))
    # b = np.random.uniform(-0.6,0.6,(1,n_num))
    H_temp = np.dot(attr, w) + bias
    H_train = 1 / (1 + np.exp(-H_temp))
    # print(H_train)
    C = pow(2, 5)

    # print(NNN)
    if NNN < n_num:
        beta = np.dot(
            np.dot(H_train.T, np.linalg.inv((np.eye(np.shape(H_train)[0]) / float(C)) + np.dot(H_train, H_train.T))),
            label)

    else:
        beta = np.dot(
            np.dot(np.linalg.inv((np.eye(np.shape(H_train)[1]) / float(C)) + np.dot(H_train.T, H_train)), H_train.T),
            label)
    # print(beta,beta.shape)
    H_temp_test = np.dot(attr_test, w) + bias
    H_test = 1 / (1 + np.exp(-H_temp_test))
    # print(beta)
    y_pre = softmax(np.dot(H_test, beta))
    # print(y_pre)
    # print(bias)

    esq = Esq(w, beta)
    # print("esq",esq)
    score = accuracy(label_test, y_pre)

    mse0 = mse(label_test, y_pre)
    loss0 = np.sum(np.sum(-label_test * np.log(y_pre)))
    # print("loss",loss0)

    B = 1
    eta = 0.3
    epsilon = B * np.sqrt(np.log(eta) / (-2 * attr.shape[0]))
    # print(epsilon)

    # Rsq = mse(label_test,y_pre)
    # print("mse",Rsq)
    Rsq_l = np.square(np.sqrt(esq) + np.sqrt(loss0) + 1) + epsilon
    Rsq_m = np.square(np.sqrt(esq) + np.sqrt(mse0) + 1) + epsilon
    with open(os.path.join(base_path,data_dir+".csv"),'a') as f:
    #     f.write(str(score)+",")
        f.write(str(Rsq_m) + ",")
    # print(score,Rsq_l,Rsq_m)
    return score, Rsq_l


def accuracy(y_test, y_pre):

    # print(y_pre.ravel())
    MissClassify_test = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    y_p = []
    for i in range(len(y_test)):
        location_test = np.argmax(y_test[i, :])
        location_predict = np.argmax(y_pre[i, :])
        # print(location_test,location_predict)
        y_p.append(location_predict)
        if location_test != location_predict:
            MissClassify_test += 1
        if location_test == 1 and location_predict == 1:
            TP = TP + 1
        if location_test == 1 and location_predict == 0:
            FN = FN + 1
        if location_test == 0 and location_predict == 1:
            FP = FP + 1
        if location_test == 0 and location_predict == 0:
            TN = TN + 1
    print(TP, TN, FP, FN)

    Accuracy = (TP + TN) / (TP + TN + FP + FN)


    # print(y_p, len(y_p))
    n1 = 0
    n0 = 0
    for i in range(0, len(y_p)):
        if y_p[i] == 1:
            n1 += 1
        else:
            n0 += 1
    print("分类结果：  少数%d，多数%d" % (n1, n0))
    return Accuracy


def softmax(x):
    return np.divide(np.exp(x).T, np.sum(np.exp(x).T, axis=0)).T


def Esq(w, beta):
    Esq_tmp = np.zeros((beta.shape[0]))
    for j in range(beta.shape[1]):
        jth = beta[:, j]
        for k in range(w.shape[1]):
            tmp = (w[:, k] ** 2).sum()
        Esq_tmp += (jth ** 2) * tmp
    # print("Esq", Esq_tmp.sum())
    return Esq_tmp.sum()


def mse(y_test, y_pre):
    return ((y_pre - y_test) ** 2).mean()


def de(n, m_size, f, cr, iterate_times, x_l, x_u, o):
    # print(o)
    best = o
    x_all = np.ones((iterate_times, m_size, n))
    for i in range(m_size):  # 种群大小
        x_all[0][i] = x_l + np.random.random() * (x_u - x_l)

    print('finish init')

    counter = 1
    previous_max = 0
    previous_indics = 0

    for g in range(iterate_times - 1):
        print('当前第{}代-------------------------------'.format(g))
        for i in range(m_size):
            x_g_without_i = np.delete(x_all[g], i, 0)
            np.random.shuffle(x_g_without_i)
            h_i = x_g_without_i[1] + f * (x_g_without_i[2] - x_g_without_i[3])
            h_i = [h_i[item] if h_i[item] < x_u[item] else x_u[item] for item in range(n)]

            h_i = [h_i[item] if h_i[item] > x_l[item] else x_u[item] for item in range(n)]

            v_i = np.array([x_all[g][i][j] if (np.random.random() > cr) else h_i[j] for j in range(n)])

            temp = evaluate_func(v_i)

            if temp[0] >= best[0] and temp[1] <= best[1]:
                best = temp

            if evaluate_func(x_all[g][i]) > evaluate_func(v_i):
                x_all[g + 1][i] = v_i
            else:
                x_all[g + 1][i] = x_all[g][i]


        best = best
        print("best", best)
    print("----------原始-----------")
    print("原始", o)
    # with open(os.path.join(base_path, data_dir + ".csv"), 'a') as f:
    #     f.write(str(best)+",")

    return 0


def evaluate_func(x):
    # n, K, M, N = x
    n = x[0]
    K = x[1]
    M = x[2]
    N = x[3]
    # print(n,K,M,N)
    sample = GMM_RSMOTE(int(round(n)), K, int(
        round(M)), N)(dataset_tuple[0], dataset_tuple[1], dataset_tuple[4])
    new_attr = np.vstack([dataset_tuple[0], sample])
    s_label = np.ndarray(sample.shape[0])
    s_label.fill(dataset_tuple[2])
    new_label = np.hstack([dataset_tuple[1], s_label])
    new_attr, new_label = shuffle(new_attr, new_label, random_state=0)
    b = runELM(new_attr, new_label, dataset_tuple[5], dataset_tuple[6], n_num=L)
    with open(os.path.join(base_path,data_dir+".csv"),'a') as f:
        # f.write(str(b)+",")
        f.write(str(n) + ",")
        f.write(str(K) + ",")
        f.write(str(M) + ",")
        f.write(str(N) + ",")
        f.write("\n")
    print(b, n, K, M, N)
    # print(b)
    # print("\n")
    return b


def all(n, K, M, N):
    sample = GMM_RSMOTE(int(round(n)), K, int(
        round(M)), N)(dataset_tuple[0], dataset_tuple[1], dataset_tuple[4])
    new_attr = np.vstack([dataset_tuple[0], sample])
    s_label = np.ndarray(sample.shape[0])
    s_label.fill(dataset_tuple[2])
    new_label = np.hstack([dataset_tuple[1], s_label])
    new_attr, new_label = shuffle(new_attr, new_label, random_state=0)
    b = runELM(new_attr, new_label, dataset_tuple[5], dataset_tuple[6], n_num=L)
    print(b, n, K, M, N)
    with open(os.path.join(base_path, data_dir + ".csv"), 'a') as f:
        f.write(str(b) + ",")
        f.write(str(n) + ",")
        f.write(str(K) + ",")
        f.write(str(M) + ",")
        f.write(str(N) + ",")
        f.write("\n")
    print("\n")
    print(b)
    # print("\n")
    return 0


def main():
    # orgin = runELM(dataset_tuple[0], dataset_tuple[1],
    #                dataset_tuple[-2], dataset_tuple[-1], n_num=L)

    # all(4, 0.3, 5, 0.5)
    # print("原始",orgin)


    x_space = np.arange(1, 11, 1)
    y_space = np.arange(0.1, 1.1, 0.1)
    z_space = np.arange(1, 20, 1)
    w_space = np.arange(0.1, 1.1, 0.1)
    for n in x_space:
        for K in y_space:
            for M in z_space:
                for N in w_space:
                    all(n,K,M,N)

    # de(n=4, m_size=5, f=0.5, cr=0.5, iterate_times=5 ,x_l=np.array([1, 0.1, 1, 0.1]),
    #    x_u=np.array([10, 1.0, 20, 1.0]),o=orgin)


if __name__ == "__main__":
    L = 10

    data_list = ["vehicle0","vehicle1","yeast1","yeast4"]
    for data_dir in data_list:
        # for data_dir in os.listdir("./k1"):
        base_path = os.path.join("./keel", data_dir)
        print(data_dir)
        if "txt" in data_dir:
            continue
        dataset_tuple = load_data(data_dir)
        d = show_sample(dataset_tuple[1])
        NNN = int(np.shape(dataset_tuple[0])[1])
        w = np.random.uniform(-1, 1, (NNN, L))
        bias = np.random.uniform(-0.6, 0.6, (1, L))
        # w = np.array(pd.read_csv("./w_"+ str(L), header=None, index_col=None))
        # bias = pd.read_csv("./b_" + str(L), header=None, index_col=None).values
        main()

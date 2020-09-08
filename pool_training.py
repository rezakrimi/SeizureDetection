import numpy as np
import scipy.io as sio
from sklearn import preprocessing, model_selection, neighbors, svm, metrics
from sklearn.neural_network import MLPClassifier as nn
import pandas as pd
from os import listdir
from os.path import join
from sklearn.linear_model import LogisticRegression
import multiprocessing


def past_features(X):
    X_1 = X[0:-2, :]
    X_2 = X[1:-1, :]
    X = X[2:, :]
    X = np.concatenate((X_1, X_2, X), axis=1)
    return X


def train_svm():
    mypath = '/Users/reza/Desktop/EEG/Code to Give/SVM_TRAINING/3seconds_7.0Hz/'
    result_path = '/Users/reza/Desktop/EEG/rbf_results/shuffle/'
    files = [f for f in listdir(mypath)]

    for f in files:
        if f == '.DS_Store':
            continue
        with open(join(result_path, f + '_results.txt'), 'a') as result_file:
            print('=================================================================================')
            print(f)
            data = sio.loadmat(join(mypath, f))['data']
            data = np.array(data)
            np.random.shuffle(data)
            temp = data.shape[1]-1
            X = np.array(data[:, 0:temp])
            X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))

            y = np.array(data[:, temp])

            print('ys:' + str(np.sum(y)))

            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
            # X_test = X_train
            # y_test = y_train

            my_score = 0
            sensitivity = 0
            specificity = 0
            best_c = best_g = best_w = 0
            best_sens = best_spec = best_prec = 0

            count = 0
            test_range = np.logspace(start=0.0, stop=8.0, num=15, base=2, dtype=int)
            w_range = np.logspace(start=1.0, stop=7, base=2, num=10, dtype=int)
            test_range = np.concatenate(([0.01, 0.06, 0.1, 0.3, 0.8], test_range[1:]), axis=0)
            print(test_range)

            # test_range = [0.8, 1.5]
            # w_range = [8]
            for c in [.01,.05,.1,.5,.8,1.2,2,5,10,15,25,40,60,120]:  # 0.003, 0.01, 0.06, 0.1, 0.5, 0.8, 1, 3, 9, 20, 60
                for g in [.01,.05,.1,.5,.8,1.2,2,5,10,15,25,40,60,120]:
                    for w in [10]:

                        print('\n\n******************')
                        result_file.write('\n*********************')
                        count = count + 1
                        print(count)
                        result_file.write('\nIteration: {}'.format(count))
                        clf = svm.SVC(C=c, verbose=3, max_iter=10000, kernel='rbf', class_weight={1: w, 0: 1}, gamma=g)
                        clf.fit(X_train, y_train)
                        accuracy = clf.score(X_test, y_test)
                        test = clf.predict(X_test)

                        tp = tn = fp = fn = 0

                        for i in range(len(y_test)):
                            if y_test[i] == 1:
                                if test[i] == 1:
                                    tp = tp + 1
                                else:
                                    fn = fn + 1
                            else:
                                if test[i] == 1:
                                    fp = fp + 1
                                else:
                                    tn = tn + 1
                        print('\nC: {}, Gamma: {}, Weight: {}'.format(c, g, w))
                        result_file.write('\nC: {}, Gamma: {}, Weight: {}'.format(c, g, w))
                        print('tp: {}'.format(tp))
                        result_file.write('\ntp: {}'.format(tp))
                        print('tn: {}'.format(tn))
                        result_file.write('\ntn: {}'.format(tn))
                        print('fp: {}'.format(fp))
                        result_file.write('\nfp: {}'.format(fp))
                        print('fn: {}'.format(fn))
                        result_file.write('\nfn: {}'.format(fn))
                        sensitivity = tp / (tp + fn)
                        if (tp + fp) != 0:
                            precision = tp / (tp + fp)
                        else:
                            precision = 0
                        specificity = tn / (tn + fp)
                        print('sensitivity: {}'.format(sensitivity))
                        result_file.write('\nsensitivity: {}'.format(sensitivity))
                        print('precision: {}'.format(precision))
                        result_file.write('\nprecision: {}'.format(precision))
                        print('specificity: {}'.format(specificity))
                        result_file.write('\nspecificity: {}'.format(specificity))

                        f1 = metrics.f1_score(y_test, test, average='macro')

                        print('f1 macro score: {}'.format(f1))
                        result_file.write('\nf1 macro score: {}'.format(f1))
                        #
                        # print('accuracy')
                        # print(accuracy)
                        # my_metric = 0
                        # if fp+tp != 0:
                        #     my_metric = tp / (fp + tp)

                        if (sensitivity + precision) != 0 and (
                            (2 * sensitivity * precision) / (sensitivity + precision)) > my_score:
                            my_score = (2 * sensitivity * precision) / (sensitivity + precision)
                            best_c = c
                            best_g = g
                            best_w = w
                            best_sens = sensitivity
                            best_spec = specificity
                            best_prec = precision
                            print('best Score so far: {}'.format(my_score))
                            result_file.write('\nbest Score so far: {}'.format(my_score))

                        print('F1 Score: {}'.format(my_score))
                        result_file.write(('\nF1 Score: {}'.format(my_score)))
                        print('\n\n******************')
                        result_file.write('\n*****************************\n')

            print('\n Final Results for {}:'.format(f))
            result_file.write('\nFinal Results for {}:'.format(f))
            print('best score: {}'.format(my_score))
            result_file.write('\nbest score: {}'.format(my_score))
            print('C: {}, Gamma: {}, Weight: {}'.format(best_c, best_g, best_w))
            result_file.write('\nC: {}, Gamma: {}, Weight: {}'.format(best_c, best_g, best_w))
            print('sensitivity: {}'.format(best_sens))
            result_file.write('\nsensitivity: {}'.format(best_sens))
            print('precision: {}'.format(best_prec))
            result_file.write('\nprecision: {}'.format(best_prec))
            print('specificity: {}'.format(best_spec))
            result_file.write('\nspecificity: {}'.format(best_spec))
            print('=================================================================================\n\n\n')


def train_nn():
    mypath = '/Users/reza/Desktop/EEG/Code to Give/SVM_TRAINING/3seconds_7.0Hz/'
    result_path = '/Users/reza/Desktop/EEG/rbf_results/shuffle/'
    files = [f for f in listdir(mypath)]

    for f in files:
        if f == '.DS_Store':
            continue
        with open(join(result_path, f + '_results.txt'), 'a') as result_file:
            print('=================================================================================')
            print(f)
            data = sio.loadmat(join(mypath, f))['data']
            data = np.array(data)
            np.random.shuffle(data)
            X = np.array(data[:, 0:40])
            kernel = np.random.uniform(low=-1, high=1, size=(40, 500))
            X = X @ kernel
            X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))

            y = np.array(data[:, 40])

            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
            # X_test = X_train
            # y_test = y_train
            test_range = np.logspace(start=-100, stop=-20, num=30, base=2, dtype=float)

            my_score = 0
            sensitivity = 0
            specificity = 0
            best_a = 0

            for a in test_range:

                print('\n\n******************')
                result_file.write('\n*********************')
                print('alpha: {}'.format(a))
                clf = nn(solver='lbfgs', alpha=a, hidden_layer_sizes=(15))
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_test, y_test)
                test = clf.predict(X_test)

                tp = tn = fp = fn = 0

                for i in range(len(y_test)):
                    if y_test[i] == 1:
                        if test[i] == 1:
                            tp = tp + 1
                        else:
                            fn = fn + 1
                    else:
                        if test[i] == 1:
                            fp = fp + 1
                        else:
                            tn = tn + 1
                print('tp: {}'.format(tp))
                result_file.write('\ntp: {}'.format(tp))
                print('tn: {}'.format(tn))
                result_file.write('\ntn: {}'.format(tn))
                print('fp: {}'.format(fp))
                result_file.write('\nfp: {}'.format(fp))
                print('fn: {}'.format(fn))
                result_file.write('\nfn: {}'.format(fn))
                sensitivity = tp / (tp + fn)
                if (tp + fp) != 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0
                specificity = tn / (tn + fp)
                print('sensitivity: {}'.format(sensitivity))
                result_file.write('\nsensitivity: {}'.format(sensitivity))
                print('precision: {}'.format(precision))
                result_file.write('\nprecision: {}'.format(precision))
                print('specificity: {}'.format(specificity))
                result_file.write('\nspecificity: {}'.format(specificity))

                if (sensitivity + precision) != 0 and (
                    (2 * sensitivity * precision) / (sensitivity + precision)) > my_score:
                    my_score = (2 * sensitivity * precision) / (sensitivity + precision)
                    best_a = a
                    best_sens = sensitivity
                    best_spec = specificity
                    best_prec = precision
                    print('best Score so far: {}'.format(my_score))
                    result_file.write('\nbest Score so far: {}'.format(my_score))

                    #
                    # print('accuracy')
                    # print(accuracy)
                    # my_metric = 0
                    # if fp+tp != 0:
                    #     my_metric = tp / (fp + tp)


def train_svm_pool(f):
    mypath = '/Users/reza/Desktop/EEG/Code to Give/SVM_TRAINING/temp/'
    result_path = '/Users/reza/Desktop/EEG/rbf_results/shuffle/'
    if f == '.DS_Store':
        return
    with open(join(result_path, f + '_results.txt'), 'a') as result_file:

        print('=================================================================================')
        print(f)
        data = sio.loadmat(join(mypath, f))['data']
        data = np.array(data)
        np.random.shuffle(data)
        temp = data.shape[1] - 1
        X = np.array(data[:, 0:temp])
        X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        y = np.array(data[:, temp])

        print('ys:' + str(np.sum(y)))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        # X_test = X_train
        # y_test = y_train

        my_score = 0
        sensitivity = 0
        specificity = 0
        best_c = best_g = best_w = 0
        best_sens = best_spec = best_prec = 0

        count = 0
        test_range = np.logspace(start=0.0, stop=8.0, num=15, base=2, dtype=int)
        w_range = np.logspace(start=1.0, stop=7, base=2, num=10, dtype=int)
        test_range = np.concatenate(([0.01, 0.06, 0.1, 0.3, 0.8], test_range[1:]), axis=0)
        print(test_range)

        # test_range = [0.8, 1.5]
        # w_range = [8]
        for c in [ 1.2, 2, 5,]:  # 0.003, 0.01, 0.06, 0.1, 0.5, 0.8, 1, 3, 9, 20, 60
            for g in [2, 5, 10]:
                for w in [10]:

                    print('\n\n******************')
                    result_file.write('\n*********************')
                    count = count + 1
                    print(count)
                    result_file.write('\nIteration: {}'.format(count))
                    clf = svm.SVC(C=c, verbose=3, max_iter=10000, kernel='rbf', class_weight={1: w, 0: 1}, gamma=g)
                    clf.fit(X_train, y_train)
                    accuracy = clf.score(X_test, y_test)
                    test = clf.predict(X_test)

                    tp = tn = fp = fn = 0

                    for i in range(len(y_test)):
                        if y_test[i] == 1:
                            if test[i] == 1:
                                tp = tp + 1
                            else:
                                fn = fn + 1
                        else:
                            if test[i] == 1:
                                fp = fp + 1
                            else:
                                tn = tn + 1
                    print('\nC: {}, Gamma: {}, Weight: {}'.format(c, g, w))
                    result_file.write('\nC: {}, Gamma: {}, Weight: {}'.format(c, g, w))
                    print('tp: {}'.format(tp))
                    result_file.write('\ntp: {}'.format(tp))
                    print('tn: {}'.format(tn))
                    result_file.write('\ntn: {}'.format(tn))
                    print('fp: {}'.format(fp))
                    result_file.write('\nfp: {}'.format(fp))
                    print('fn: {}'.format(fn))
                    result_file.write('\nfn: {}'.format(fn))
                    sensitivity = tp / (tp + fn)
                    if (tp + fp) != 0:
                        precision = tp / (tp + fp)
                    else:
                        precision = 0
                    specificity = tn / (tn + fp)
                    print('sensitivity: {}'.format(sensitivity))
                    result_file.write('\nsensitivity: {}'.format(sensitivity))
                    print('precision: {}'.format(precision))
                    result_file.write('\nprecision: {}'.format(precision))
                    print('specificity: {}'.format(specificity))
                    result_file.write('\nspecificity: {}'.format(specificity))

                    f1 = metrics.f1_score(y_test, test, average='macro')

                    print('f1 macro score: {}'.format(f1))
                    result_file.write('\nf1 macro score: {}'.format(f1))
                    #
                    # print('accuracy')
                    # print(accuracy)
                    # my_metric = 0
                    # if fp+tp != 0:
                    #     my_metric = tp / (fp + tp)

                    if (sensitivity + precision) != 0 and (
                                (2 * sensitivity * precision) / (sensitivity + precision)) > my_score:
                        my_score = (2 * sensitivity * precision) / (sensitivity + precision)
                        best_c = c
                        best_g = g
                        best_w = w
                        best_sens = sensitivity
                        best_spec = specificity
                        best_prec = precision
                        print('best Score so far: {}'.format(my_score))
                        result_file.write('\nbest Score so far: {}'.format(my_score))

                    print('F1 Score: {}'.format(my_score))
                    result_file.write(('\nF1 Score: {}'.format(my_score)))
                    print('\n\n******************')
                    result_file.write('\n*****************************\n')

        print('\n Final Results for {}:'.format(f))
        result_file.write('\nFinal Results for {}:'.format(f))
        print('best score: {}'.format(my_score))
        result_file.write('\nbest score: {}'.format(my_score))
        print('C: {}, Gamma: {}, Weight: {}'.format(best_c, best_g, best_w))
        result_file.write('\nC: {}, Gamma: {}, Weight: {}'.format(best_c, best_g, best_w))
        print('sensitivity: {}'.format(best_sens))
        result_file.write('\nsensitivity: {}'.format(best_sens))
        print('precision: {}'.format(best_prec))
        result_file.write('\nprecision: {}'.format(best_prec))
        print('specificity: {}'.format(best_spec))
        result_file.write('\nspecificity: {}'.format(best_spec))
        print('=================================================================================\n\n\n')

def pool_training():
    mypath = '/Users/reza/Desktop/EEG/Code to Give/SVM_TRAINING/temp/'
    result_path = '/Users/reza/Desktop/EEG/rbf_results/shuffle/'
    files = [f for f in listdir(mypath)]
    p = multiprocessing.Pool()
    p.map(train_svm_pool, files)

pool_training()

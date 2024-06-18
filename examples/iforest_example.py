# -*- coding: utf-8 -*-
"""Example of using Isolation Forest for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.iforest import IForest
from pyod.utils.data import generate_data

from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

if __name__ == "__main__":


    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    X_train, X_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42)

    # train IForest detector
    clf_name = 'IForest_1'
    clf = IForest()
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores
    th = clf.threshold_
    print("-----train_1-----")
    print("scores", y_train_scores[-20:-1])
    print("th", th)
    print("labels", y_train_pred)
    evaluate_print(clf_name, y_train, y_train_scores)


    # y_train_pred_2 = clf.predict(X_train)  # outlier labels (0 or 1)
    # y_train_scores_2 = clf.decision_function(X_train)  # outlier scores
    # print("-----train_2-----")
    # print("scores", y_train_scores_2[-20:-1])
    # print("th", th)
    # print("labels", y_train_pred_2)
    # evaluate_print(clf_name, y_train, y_train_scores_2)



    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores
    print("-----test_1-----")
    print("scores", y_test_scores[-20:-1])
    print("th", th)
    print("labels", y_test_pred)
    eval_test = evaluate_print(clf_name, y_test, y_test_scores)




    clf_name_2 = 'IForest_2'
    clf_2 = IForest()
    clf_2.fit(X_train)
    y_train_pred_2 = clf_2.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores_2 = clf_2.decision_scores_  # raw outlier scores
    th_2 = clf_2.threshold_
    print("-----train_2-----")
    print("scores", y_train_scores_2[-20:-1])
    print("th", th_2)
    print("labels", y_train_pred_2)
    evaluate_print(clf_name_2, y_train, y_train_scores_2)



    clf_name_3 = 'IForest_3'
    clf_3 = IForest()
    clf_3.fit(X_test)
    y_test_pred_3 = clf_3.labels_  # binary labels (0: inliers, 1: outliers)
    y_test_scores_3 = clf_3.decision_scores_  # raw outlier scores
    th_3 = clf_3.threshold_
    print("-----test_3-----")
    print("scores", y_test_scores_3[-20:-1])
    print("th", th_3)
    print("labels", y_test_pred_3)
    evaluate_print(clf_name_3, y_test, y_test_scores_3)







    # # example of the feature importance
    # feature_importance = clf.feature_importances_
    # print(type(feature_importance))
    # print("Feature importance", feature_importance)

    # # visualize the results
    # visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
    #           y_test_pred, show_figure=True, save_figure=False)


    # from joblib import dump, load
    # # save the model
    # dump(clf, 'iforest.joblib')
    # # load the model
    # clf_load = load('iforest.joblib')
    # print(clf_load.feature_importances_)




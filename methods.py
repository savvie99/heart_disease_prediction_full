from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Function for fitting and evaluating the models
def run_evaluate(model, x_train, y_train, x_test, y_test):
    train_start = time.time() # model training starts
    model.fit(x_train, y_train) # the model is fitted on the training set
    train_stop = time.time() # model training stops
    test_start=time.time() # testing starts
    y_pred = model.predict(x_test) # model predicts the labels of the features in the test set
    test_stop=time.time() # testing stops
    print("Training time: ", train_stop - train_start) # print training time
    print("Testing time: ", test_stop - test_start) # print testing time
    print(classification_report(y_test, y_pred)) # print classification report
    print("Recall:", recall_score(y_test,y_pred))

    conf_matrix=confusion_matrix(y_test, y_pred) # calculate confusion matrix
    ax = sns.heatmap(conf_matrix, annot=True, fmt = "g")
    # plot the confusion matrix
    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');


    ## Display the visualization of the Confusion Matrix.
    plt.rcParams["figure.figsize"] = (10,6) # set size
    plt.show()

def training_accuracy(model, x_train, y_train, x_test, y_test):
    train_yhat = model.predict(x_train)
    train_acc = accuracy_score(y_train, train_yhat)
    print('Accuracy on training set:', train_acc)

def cross_validate(model, x_train, y_train):
    scores=cross_val_score(model, x_train, y_train, cv=5) # calculate cross validation scores
    print("Cross Validation Scores:\n")
    print("Mean score: ", scores.mean()) # print the mean
    print("Standard deviation: ", scores.std()) # print standard deviation
    return scores

#%% FUNCTION TO PLOT ROC-AUC CURVE (with help from https://machinelearningmastery.com)
def ROC_AUC(model , y_test, X_test):
    ns_probs = [0 for _ in range(len(y_test))] # generate a no skill prediction
    lr_probs = model.predict_proba(X_test) # predict probabilities
    lr_probs = lr_probs[:, 1] # keep probabilities for the positive outcome only
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Model: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Model')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

def find_best_k(x_train, y_train, x_test, y_test):
    error = []
    # Calculating error for K values between 1 and 30
    for i in range(1, 30):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train) # fit the classifier on the training set
        pred_i = knn.predict(x_test) # predict the classes
        error.append(np.mean(pred_i != y_test)) # append the error on the list
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10) # plot the error list 
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    print("Minimum error:-",min(error),"at K =",error.index(min(error))+1)

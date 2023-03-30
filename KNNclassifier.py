from collections import Counter
from pickle import FALSE
import numpy as np
import seaborn as sn
import pandas as pd
from keras.datasets import mnist
from matplotlib import pyplot as plt
import math as m
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# load mnist data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# reduce values in order to work better
train_X1 = train_X[:600]
train_y1 = train_y[:600]
test_X1 = test_X[:100]
test_y1 = test_y[:100]

def CheckDataset(train_x,train_y, test_x, test_y):
    if len(train_x) != len(train_y) or len(test_x) != len(test_y):
        print('ERROR: different lengths between training and test')
        exit()
    if len(train_x) == 0 or len(test_x) == 0:
        print("ERROR: trainset or testset can't be empty")
        exit()

def Check_K_value(train_x, k):
    if k > len(train_x) or k < 1:
        print('ERROR: invalid k value')
        exit()
    if int(k) == FALSE:
        print('ERROR: k is not an integer')
        exit()

def KNNclassifier(train_x, train_y, test_x, test_y, k):
    # check data 
    CheckDataset(train_x, train_y, test_x, test_y)
    Check_K_value(train_x, k)

    # apply reshape function in order to work in 2-Dimensions
    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)
   
    # initialize lists
    list_distance = []
    ind_counter = []
    index_list = []
    res_list = []
    pred_list=[]

    # calculating prediction list
    for i in range(len(test_x)):    
        for j in range(len(train_x)):
            distance = m.dist(test_x[i], train_x[j]) # Euclidean distance
            list_distance.append(distance)
            ind_counter.append(j)
            distance = 0

        # dictionary to store all the results  
        d = {'index':ind_counter, 'distance': list_distance}
        # convert dictionary to dataframe
        df = pd.DataFrame(d, columns = ['index', 'distance'])
        # sort in ascending order by euclidean distance
        df_sorted = df.sort_values(by = 'distance')
       
        index_list = df_sorted.iloc[:k]
        for u in range(len(index_list)):
            res_list.append(train_y[int(index_list.iloc[u][0])])

        # now get the count of the max class in result list
        pred_value = Counter(res_list).most_common(1)[0][0]
        # storing every prediction in a list
        pred_list.append(pred_value)

        #reinitialize lists
        list_distance = []
        ind_counter = []
        index_list = []
        res_list = []

    # calculating accuracy and error rate
    accuracy = 0
    number_errors = 0
    if len(test_y)!=0:
        for i in range(len(test_y)):     
            if pred_list[i]==test_y[i]:
                accuracy += 1
            else:
                number_errors += 1
        accuracy=accuracy/len(test_y)
        error_rate = number_errors/len(test_y)
    else:
        accuracy=0
    return round(accuracy*100,2), round(error_rate*100,2), pred_list


# confusion matrix for knn classifier
def ConfusionMatrix(test_conf, pred_conf):
    # initialize matrix
    matrix = [[0 for x in range(10)] for y in range(10)]
    for i in range(len(test_conf)):
        matrix[test_conf[i]][pred_conf[i]] += 1
    return matrix

# classification quality indexes of confusion matrix
def ClassificationQualityIndexes(matrix):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                TP += matrix[i][i]
            else:
                FN += matrix[i][j]
                FP += matrix[j][i]
    TN = len(matrix)*sum(sum(matrix, [])) - TP - FN - FP

    sensivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP) 
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    f1_score = 2*TP/(2*TP+FP+FN)
    sd = np.std(matrix)

    return round(sensivity*100,2), round(specificity*100,2), round(precision*100,2), round(accuracy*100,2), round(f1_score*100,2), round(sd,2)
   
# list of several values of K
k1_values = [1,2,3,4,5,10,15,20,30,40,50]
k2_values = [1,3,5,10,25,50]

# plotting the accuracy for several values of k
accuracy_list = []
error_list = []
k1_string_list = []
pred_list = []
for i in k1_values:
    accuracy, error, pred = KNNclassifier(train_X1, train_y1, test_X1, test_y1, i)
    accuracy_list.append(accuracy)
    error_list.append(error)
    pred_list.append(pred)
    k1_string_list.append(str(i))
plt.subplot(1,2,1)
plt.bar(k1_string_list,accuracy_list, align='center', color='g')
plt.xlabel('K')
plt.ylabel('Accuracy(%)')
plt.title("KNN's accuracy for several values of k")
plt.subplot(1,2,2)
plt.bar(k1_string_list,error_list, align='center', color='r')
plt.xlabel('K')
plt.ylabel('Error-Rate(%)')
plt.title("KNN's Error-Rate for several values of k")
plt.show()

# plotting the confusion matrix for some values of k
for i in range(len(k2_values)):
    plt.subplot(2,3,i+1)
    plt.subplots_adjust(wspace = 0.5, hspace=0.5)
    plt.title('Confusion matrix for k = '+str(k2_values[i]))
    matrix = ConfusionMatrix(test_y1, pred_list[i])
    df_cm = pd.DataFrame(matrix, range(len(matrix)), range(len(matrix)))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.suptitle('Examples of Confusion matrix for different values of k')
plt.show()

# calculating the quality indexes of confusion matrix
sensivity_list = []
specificity_list = []
precision_list = []
accuracy_list = []
f1_score_list = []
standard_deviation_list = []
for i in range(len(pred_list)):
    sensitivity,specificity,precision,Accuracy,f1,sd= ClassificationQualityIndexes(ConfusionMatrix(test_y1, pred_list[i]))
    sensivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    accuracy_list.append(Accuracy)
    f1_score_list.append(f1)
    standard_deviation_list.append(sd)

#create a table with the quality indexes for each value of k
columns_labels = ['Precision', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'Standard Deviation']
rows_labels = k1_string_list
data_cells = []
for i in range(len(k1_values)):
    data_cells.append([precision_list[i], accuracy_list[i], sensivity_list[i], specificity_list[i], f1_score_list[i], standard_deviation_list[i]])

plt.axis('tight')
plt.axis('off')
plt.title('Quality Indexes for different values of k')
plt.table(cellText=data_cells, rowLabels=rows_labels, colLabels=columns_labels, rowColours=['pink']*len(rows_labels), colColours=['green']*len(columns_labels), loc='center')
plt.show()

# plotting the accuracy of each digit classes
accuracy_classes=[]
pred_classes=[]
test_list=[]
J=1
for i in range(10):
    for j in k1_values:
        try:
            test_x2=test_X1[test_y1==i]
            test_y2=test_y1[test_y1==i]
            accuracy, error, pred = KNNclassifier(train_X1, train_y1, test_x2, test_y2, j)
            accuracy_classes.append(accuracy)
            pred_classes.append(pred)
            test_list.append(test_y2)
        except ValueError:
            print('ValueError: cannot reshape array of size 0 into shape (0,newaxis)','\n')
            print('Please, try again with a larger datasets')
            exit()

    plt.subplot(4,3,i+1)
    plt.subplots_adjust(wspace = 0.3, hspace=0.7)
    plt.bar(k1_string_list, accuracy_classes, align='center', color='g')
    plt.xlabel('K')
    plt.ylabel('Class '+str(i))
    accuracy_classes=[]
plt.suptitle('Accuracy of each digit vs the remaining 9 for different values of k')
plt.show()

# calculating the quality indexes of confusion matrix
sensivity_list = []
specificity_list = []
precision_list = []
accuracy_list = []
f1_score_list = []
standard_deviation_list = []
index = 4
for i in range(10):
    sensitivity,specificity,precision,Accuracy,f1,sd= ClassificationQualityIndexes(ConfusionMatrix(test_list[index], pred_classes[index]))
    sensivity_list.append(sensitivity)
    specificity_list.append(specificity)
    precision_list.append(precision)
    accuracy_list.append(Accuracy)
    f1_score_list.append(f1)
    standard_deviation_list.append(sd)
    index += 10

#create a table with the quality indexes for each value of k
columns_labels = ['Accuracy','Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'Standard Deviation']
rows_labels = [x for x in range(10)]
data_cells = []
for i in range(10):
    data_cells.append([accuracy_list[i], precision_list[i], sensivity_list[i], specificity_list[i], f1_score_list[i], standard_deviation_list[i]])

plt.axis('tight')
plt.axis('off')
plt.title('Quality Indexes for each digit vs the remaining 9 with k = 5')
plt.table(cellText=data_cells, rowLabels=rows_labels, colLabels=columns_labels, rowColours=['pink']*len(rows_labels), colColours=['green']*len(columns_labels), loc='center')
plt.show()
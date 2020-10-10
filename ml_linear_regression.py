import math
import numpy as np
import csv
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import re

def select_features(train_arr, test_arr, feature_list):
    n_train_arr = np.array(train_arr[:, feature_list[0]])
    n_train_arr.shape = (len(n_train_arr), 1)
    n_test_arr = np.array(test_arr[:, feature_list[0]])
    n_test_arr.shape = (len(n_test_arr), 1)
    for i in range(1, len(feature_list)):
        add_train_vect = np.array(train_arr[:, feature_list[i]])
        add_train_vect.shape = (len(add_train_vect), 1)
        add_test_vect = np.array(test_arr[:, feature_list[i]])
        add_test_vect.shape = (len(add_test_vect), 1)
        n_train_arr = np.append(n_train_arr, add_train_vect, axis=1)
        n_test_arr = np.append(n_test_arr, add_test_vect, axis=1)
    return n_train_arr, n_test_arr

def predict_outputs(weight_vector, arr):
    tw_vector = weight_vector.transpose()
    return np.dot(arr, tw_vector)

def get_cost(weight_vector, labels, arr):
    outputs = predict_outputs(weight_vector, arr)
    error = np.subtract(labels, outputs)
    squared = np.power(error, 2)
    mean_squared = np.divide(squared, 2*(len(arr)))
    return mean_squared

def get_new_weights(weight_vector, labels, learning_rate, arr):
    outputs = predict_outputs(weight_vector, arr)
    error_vector = np.subtract(labels, outputs)
    grad_arr = np.transpose(arr)
    slope_vector = np.dot(grad_arr, error_vector)
    slope_vector = np.multiply(np.divide(slope_vector, len(arr)), learning_rate)
    weight_vector = np.add(weight_vector, slope_vector)
    return weight_vector

def gradient_descent(weight_vector, labels, learning_rate, arr, epochs):
    error_record = []

    for _ in range(epochs):
        error = get_squared_error(predict_outputs(weight_vector, arr), labels)
        error_record.append(error)
        weight_vector = get_new_weights(weight_vector, labels, learning_rate, arr)
    
    return weight_vector, error_record

def get_test_fit(weight_vector, training_labels, learning_rate, train_arr, epochs, test_arr, test_labels):
    trained_weight_vector, error_record = gradient_descent(weight_vector, training_labels, learning_rate, train_arr, epochs)
    predictions = predict_outputs(trained_weight_vector, test_arr)
    fit = get_squared_error(predictions, test_labels)
    return fit, predictions, error_record

def get_squared_error(predictions, labels):
    error = np.sum(np.power(np.subtract(predictions, labels), 2))
    return (1 / (2 * len(predictions)) * error)

def convert_categorical(arr, idx):
    cat_vector = arr[:, idx]
    categories = sorted(list(set(cat_vector)))
    cat_dict = {}
    for i in range(len(categories)):
        cat_dict[categories[i]] = i
    new_arr = [np.zeros(len(cat_vector)) for i in range(len(categories))]
    for i in range(len(cat_vector)):
        new_arr[cat_dict[cat_vector[i]]][i] = 1
    new_arr = np.array([i for i in new_arr])
    new_arr = new_arr.transpose()
    new_arr = np.append(arr, new_arr, axis=1)
    return new_arr

def convert_binary(arr, idx):
    bin_vector = arr[:, idx]
    bins = list(set(bin_vector))
    bin_dict = {}
    for i in range(0, 2):
        bin_dict[bins[i]] = i
    return np.array([bin_dict[i] for i in bin_vector])

def scale_feature(feature):
    column_vector = np.array(feature, dtype=float)
    column_vector = np.subtract(column_vector, min(column_vector))
    denominator = max(column_vector) - min(column_vector)
    if denominator != 0:
        return column_vector / denominator
    else:
        return np.zeros(len(column_vector))

def normalize(feature):
    column_vector = np.array(feature)
    column_vector = np.delete(column_vector, np.argwhere(column_vector == 0))
    vector_mean = np.mean(column_vector)
    return np.where(feature == 0, vector_mean, feature)

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path + '\\student-por.csv', 'r', newline='') as iris_file:
    ptg_data = [i[0].split(';') for i in list(csv.reader(iris_file))]

for i in range(1, len(ptg_data)):
    ptg_data[i][-2] = re.sub('\"', '', ptg_data[i][-2])
    ptg_data[i][-3] = re.sub('\"', '', ptg_data[i][-3])

ptg_header = np.array(ptg_data[0][:-3])
ptg_arr = np.array(ptg_data[1:])
ptg_labels = np.array(ptg_arr[:, -1])
ptg_arr = np.array(ptg_arr[:, :-3])

binary_types = [i for i in range(len(ptg_arr[0])) if len(set(ptg_arr[:,i])) == 2]
categorical_types = [i for i in range(8, 12)]
post_expansion_end = len(ptg_arr[0]) - len(categorical_types)

for i in binary_types:
    ptg_arr[:, i] = convert_binary(ptg_arr, i)
for i in categorical_types:
    ptg_arr = convert_categorical(ptg_arr, i)
    ptg_header = np.append(ptg_header, np.array(sorted(list(set(ptg_arr[:, i])))))

ptg_arr = np.delete(ptg_arr, categorical_types, axis=1)
ptg_header = np.delete(ptg_header, categorical_types, axis=0)

binary_types = [i for i in range(0, post_expansion_end) if len(set(ptg_arr[:,i])) == 2]
categorical_types = [i for i in range(post_expansion_end, len(ptg_arr[0]))]
continuous_types = [i for i in range(len(ptg_arr[0])) if i not in binary_types and i not in categorical_types]

bias_vector = np.array(np.ones(shape = (len(ptg_arr), 1)))
ptg_arr = np.append(ptg_arr, bias_vector, axis=1)

ptg_training = ptg_arr[:int(.8*len(ptg_arr)), :]
ptg_training_labels = ptg_labels[:int(.8*len(ptg_arr))]
ptg_testing = ptg_arr[int(.8*len(ptg_arr)):, :]
ptg_testing_labels = ptg_labels[int(.8*len(ptg_arr)):]

for i in continuous_types:
    ptg_training[:, i] = scale_feature(ptg_training[:, i])
    ptg_testing[:, i] = scale_feature(ptg_testing[:, i])

ptg_training = ptg_training.astype(float)
ptg_training_labels = ptg_training_labels.astype(float)
ptg_testing = ptg_testing.astype(float)
ptg_testing_labels = ptg_testing_labels.astype(float)

feature_group1 = [9, 10, 11, 13, 15, 25]
feature_group2 = [6, 7, 11, 12, 13, 17]
feature_group3 = [8, 14, 16, 21, 39, 40, 41, 42]

f_train1, f_test1 = select_features(ptg_training, ptg_testing, feature_group1)
f_train2, f_test2 = select_features(ptg_training, ptg_testing, feature_group2)
f_train3, f_test3 = select_features(ptg_training, ptg_testing, feature_group3)

train_list = [f_train1, f_train2, f_train3, ptg_training]
test_list = [f_test1, f_test2, f_test3, ptg_testing]

epoch_compare = []
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

for k in range(len(epoch_list)):
    compare_error = []
    for i in range(len(train_list)):
        weight_vector = np.array([0.5 for i in range(len(train_list[i][0]))])
        compare_error.append(get_test_fit(weight_vector, ptg_training_labels, 0.05, train_list[i], epoch_list[k], test_list[i], ptg_testing_labels)[0])
    epoch_compare.append(compare_error)

test_error_compare = np.array(epoch_compare)

"""
with open(dir_path + '\\testing_mse.csv', 'w') as np_save_testing:
    np.savetxt(np_save_testing, test_error_compare, delimiter=',')
"""

print(test_error_compare)

plt.plot(test_error_compare)
plt.title("Test Set Performance - MSE on Testing Data by Epoch")
plt.legend(["Educational", "Financial", "Motivational", "All"])
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.xticks(ticks=[i for i in range(0, 10)], labels=[str(i) for i in epoch_list])
plt.show()

compare_descent = []
for i in range(len(train_list)):
    weight_vector = np.array([0.5 for i in range(len(train_list[i][0]))])
    compare_descent.append(gradient_descent(weight_vector, ptg_training_labels, 0.05, train_list[i], 50)[1])

descent_compare = np.array(compare_descent)
descent_compare = np.transpose(descent_compare)
print(descent_compare)

"""
with open(dir_path + '\\gradient_descent.csv', 'w') as np_save:
    np.savetxt(np_save, descent_compare, delimiter=',')
"""

plt.plot(descent_compare)
plt.title("Algorithm Convergence - MSE on Training Data by Epoch")
plt.legend(["Educational", "Financial", "Motivational", "All"])
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()
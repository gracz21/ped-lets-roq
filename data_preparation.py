from collections import Counter
import numpy as np
import pandas as pd
import time

start_x = time.time()
x_data = pd.read_csv("data/X_train.csv", header=None, delimiter=",")
start_y = time.time()
y_data = pd.read_csv("data/y_train.csv", header=None, delimiter=",")
finish_y = time.time()

start_preparation = time.time()
x_data_non_zero_features = x_data.loc[:, (x_data.columns[x_data.apply(lambda x: min(x) != max(x))])]
end_preparation = time.time()

start_writing = time.time()
x_data_non_zero_features.to_csv('data/X_train_prepared.csv', header=False, index=False)
end_writing = time.time()

with open("out/1_2.txt", "w") as output:
    output.write('Times:\n')
    output.write('Time to read x: ' + str(start_y - start_x) + ' s\n')
    output.write('Time to read y: ' + str(finish_y - start_y) + ' s\n')
    output.write('Time to prepare x: ' + str(end_preparation - start_preparation) + 's\n')
    output.write('Time to save prepared x: ' + str(end_writing - start_writing) + 's\n')

    row_num, col_num = x_data.shape

    output.write('\nBasic features statistics:\n')
    output.write('Num of features: ' + str(col_num) + '\n')
    output.write('Num of features after preparation: ' + str(x_data_non_zero_features.shape[1]) + '\n')
    output.write('Num of examples: ' + str(row_num) + '\n')
    output.write('% of non-zero feature values: ' + str(np.count_nonzero(x_data)/x_data.size) + '\n')

    y_counter = Counter(y_data[0])
    key, val = y_counter.most_common(1)[0]
    output.write('\nBasic classes statistics:\n')
    output.write('Number of classes: ' + str(len(y_counter)) + '\n')
    output.write('Dominant class: ' + str(key) + '\n')
    output.write('Probability of dominant class: ' + str(val/len(y_data)) + '\n')

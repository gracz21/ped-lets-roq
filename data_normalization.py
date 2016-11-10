import pandas as pd

x_data = pd.read_csv('data/X_train_prepared.csv', header=None, delimiter=",")

with open('out/1_4.txt', 'w') as output:
    x_data_z_score = x_data.copy(deep=True)
    x_data_min_max = x_data.copy(deep=True)

    for col in range(x_data.shape[1]):
        x_data_z_score[col] = (x_data_z_score[col] - x_data_z_score[col].mean()) / x_data_z_score[col].std()
        x_data_min_max[col] = (x_data_min_max[col] - x_data_min_max[col].min()) / \
                              (x_data_min_max[col].max() - x_data_min_max[col].min())

    output.write('Name\tAvg\tStd\tMin\tMax\tGlobal min\tGlobal max\n')

    output.write('Before:\t')
    output.write(str(x_data[0].mean()) + '\t')
    output.write(str(x_data[0].std()) + '\t')
    output.write(str(x_data[0].min()) + '\t')
    output.write(str(x_data[0].max()) + '\t')
    output.write(str(x_data.values.min()) + '\t')
    output.write(str(x_data.values.max()) + '\n')

    output.write('Z-score:\t')
    output.write(str(x_data_z_score[0].mean()) + '\t')
    output.write(str(x_data_z_score[0].std()) + '\t')
    output.write(str(x_data_z_score[0].min()) + '\t')
    output.write(str(x_data_z_score[0].max()) + '\t')
    output.write(str(x_data_z_score.values.min()) + '\t')
    output.write(str(x_data_z_score.values.max()) + '\n')

    output.write('Min-max:\t')
    output.write(str(x_data_min_max[0].mean()) + '\t')
    output.write(str(x_data_min_max[0].std()) + '\t')
    output.write(str(x_data_min_max[0].min()) + '\t')
    output.write(str(x_data_min_max[0].max()) + '\t')
    output.write(str(x_data_min_max.values.min()) + '\t')
    output.write(str(x_data_min_max.values.max()))

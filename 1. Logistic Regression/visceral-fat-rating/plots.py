# plots.py
# plotting visceral-fat-rating data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

LABELS_DICT = {
    '1'     : 'Subject id',
    '2'     : 'Sex',
    '3'     : 'Height',
    '4'     : 'Body weight',
    '5'     : 'Waist circumference',
    '6'     : 'Hip circumference',
    '7'     : 'Arm circumference',
    '8'     : 'Adipose tissue',
    '9'     : 'Muscle mass',
    '10'    : 'Bone mass',
    '11'    : 'BMI (Body Mass Index)',
    '12'    : 'DCI (Daily Calorie Intake)',
    '13'    : 'Metabolic age',
    '14'    : 'Body water',
    '15'    : 'Class'
}

def load_data(file_name):
    data_file = file_name
    data_labels = LABELS_DICT.keys()
    data_frame = pd.read_csv(data_file, header = None, names = data_labels)

    return data_frame

def plot_scatter_matrix(data_frame):
    filtered_df = data_frame.iloc[:, 2:14]
    color_wheel = {1: "#ff0000", 2: "#0000ff"}
    colors = data_frame['2'].map(lambda x: color_wheel.get(x))
    sm = scatter_matrix(filtered_df, alpha = 0.2, diagonal = 'kde', color = colors)
    # Hide all ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]

    # Save scatter matrix as PDF file
    #file_name = 'scatter_matrix'
    #plt.savefig(file_name + '.pdf')

    # Plot scatter matrix
    #plt.show()

def plot_each_figure(labels, x, y):
    df = pd.DataFrame(dict(x = x, y = y, label = labels))
    groups = df.groupby('label')

    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    ax.set_xlabel(LABELS_DICT[x.name])
    ax.set_ylabel(LABELS_DICT[y.name])
    for name, group in groups:
        ci = '#ff0000' if name == 1 else '#0000ff'
        ax.plot(group.x, group.y, marker = 'o', linestyle = '', ms = 4, label = name, color = ci)
    ax.legend()

    # Save figures as PDF files
    #file_name = LABELS_DICT[x.name] + '_' + LABELS_DICT[y.name]
    #plt.savefig(file_name + '.pdf')

    # Plot figures
    #plt.show()

def plot_each(data_frame):
    labels = data_frame['2']
    filtered_df = data_frame.iloc[:, 2:14]

    column_numbers = len(filtered_df.columns)
    i = 0
    while (i < column_numbers - 1):
        j = i
        while (j < column_numbers - 1):
            x = data_frame[str(i + 3)] # (i=0) + 3 = '3'
            y = data_frame[str(j + 4)] # (j=i) + 4 = '4'
            plot_each_figure(labels, x, y)
            j += 1
        i += 1   

def main():
    df = load_data('.\\visceral-fat-rating.data')
    plot_scatter_matrix(df)
    #plot_each(df)


if __name__ == '__main__':
    main()
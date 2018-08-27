# plots.py
# plotting visceral-fat-rating data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def load_data(file_name):
    data_file = file_name
    data_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    data_frame = pd.read_csv(data_file, header=None, names=data_labels)

    return data_frame

def plot_scatter_matrix(data_frame):
    filtered_df = data_frame.iloc[:, 2:14]
    sm = scatter_matrix(filtered_df, alpha=0.2, diagonal='kde')
    #Hide all ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]
    plt.show()

def plot_each(data_frame):

    labels = data_frame['2']
    x = data_frame['5']
    y = data_frame['14']

    new_df = pd.DataFrame(dict(x=x, y=y, label=labels))
    groups = new_df.groupby('label')

    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    ax.set_xlabel('Waist circumference')
    ax.set_ylabel('Body water')
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=3, label=name)
    ax.legend()
    plt.show()

def main():
    df = load_data('.\\visceral-fat-rating.data')
    #plot_scatter_matrix(df)
    plot_each(df)


if __name__ == '__main__':
    main()
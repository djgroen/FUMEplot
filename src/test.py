
from plot_uq import combine_data, data_statistics, plot_camps_uq

import csv

def test():

    data, n_runs = combine_data()
    #print(data.head())

    data = data_statistics(data)
    #data.to_csv('data_comb_stat.csv', index=True)
    #print(data.index)
    # print(data)

    plot_camps_uq(data, {"n_sim": n_runs},"sample_output")

if __name__ == "__main__":

    test()

import numpy as np
import pandas as pd


def fileread_tsv(FileName):
    tsv_reader = pd.read_csv(FileName + '.tsv', sep='\t', index_col=None, header=None)
    data_line_number = int(len(tsv_reader[1].values))
    timelong = data_line_number / 500
    t_list = list(np.linspace(0, timelong, int(data_line_number)))
    x_list = tsv_reader[2].values
    y_list = tsv_reader[3].values
    z_list = tsv_reader[4].values
    return t_list, x_list, y_list, z_list


def fileread_tsv_pro(FileName):
    tsv_reader = pd.read_csv(FileName + '.tsv', sep='\t', index_col=None, header=None)
    data_line_number = int(len(tsv_reader[1].values))
    timelong = data_line_number / 1000
    t_list = list(np.linspace(0, timelong, int(data_line_number)))
    x_list = tsv_reader[2].values
    y_list = tsv_reader[3].values
    z_list = tsv_reader[4].values
    return t_list, x_list, y_list, z_list


def fileread_csv(FileName):
    tsv_reader = pd.read_csv(FileName + '.csv', sep=',', index_col=None, header=None)
    data_line_number = int(len(tsv_reader[1].values) - 1)
    timelong = data_line_number / 500
    t_list = list(np.linspace(0, timelong, int(data_line_number)))
    x_list = tsv_reader[0].values[1:]
    y_list = tsv_reader[1].values[1:]
    z_list = tsv_reader[2].values[1:]
    x_list = list(map(float, x_list))
    y_list = list(map(float, y_list))
    z_list = list(map(float, z_list))
    return t_list, x_list, y_list, z_list


def fileread_csv_align(FileName):
    tsv_reader = pd.read_csv(FileName + '.csv', sep='\t', index_col=None, header=None)
    data_line_number = int(len(tsv_reader[0].values))
    timelong = data_line_number / 500
    t_list = list(np.linspace(0, timelong, int(data_line_number)))
    z_list = tsv_reader[0].values
    return t_list, z_list

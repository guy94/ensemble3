import pandas as pd
from os import walk


def read_file(file):
    """
    reads a csv file into pandas dataFrame.
    :param file:
    :return:
    """

    df = pd.read_csv(file)
    return df


def read_file_names(path):
    """
    reads all csv files names in a certain directory.
    :param path:
    :return:
    """

    files_list = []
    for (dirpath, dirnames, filenames) in walk(path):
        files_list = [dirpath+'/'+file_name for file_name in filenames if '.csv' in file_name]
        break

    return files_list

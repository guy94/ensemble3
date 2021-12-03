import collections

import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
pd.options.mode.chained_assignment = None


class DataPreparation:

    def __init__(self):
        self.df = pd.DataFrame
        self.label_encoder = LabelEncoder()
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.test_index = []
        self.train_index = []
        self.all_data_frames = []

    def fill_na(self):
        """
        fills NaN values for each of the split data sets.
        Categorical columns are filled with the most common value,
        while numerical columns are filed with the mean of the column.
        :return:
        """

        for df in self.all_data_frames:
            for col in df:
                if df[col].dtype == np.object:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)

    def discretization(self, df):
        """
         sklearn models can not handle categorical data, label-encoding (similar to dummy data) was applied.
         in addition, we used discretization for continuous-numerical data.
        :return:
        """

        for col in df:
            if df[col].dtype != np.object:
                df[col] = pd.cut(x=df[col], bins=3, include_lowest=True)
            df[col] = self.label_encoder.fit_transform(df[col])

    def partition_data_sets(self):
        """
        split data set into X_train, X_test, y_train, y_test sets.
        test set size - 30% of the data.
        train set size - 70% of the data.
        :return:
        """

        self.y = self.df['class']
        self.x = self.df.drop('class', axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=42)
        self.y_train = self.y_train.to_frame('class')
        self.y_test = self.y_test.to_frame('class')

        self.all_data_frames = [self.x_train, self.x_test, self.y_train, self.y_test]



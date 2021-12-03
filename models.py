from turtledemo import forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from rotation_forest import RotationTreeClassifier


class DTC:

    def __init__(self, max_depth=None, min_samples_split=2):

        self.model = DecisionTreeClassifier(random_state=0,
                                            max_depth=max_depth, min_samples_split=min_samples_split)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class ourRandomForest(DTC):

    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="auto"):

        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                            max_features=max_features)


class ourRotationForest(DTC):

    def __init__(self, n_features_per_subset=3, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1):

        super().__init__()
        self.model = RotationTreeClassifier(n_features_per_subset=n_features_per_subset, criterion=criterion,
                                            max_depth=max_depth, min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf)

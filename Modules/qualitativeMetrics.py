import os
import numpy as np
import pandas as pd
from Modules import datasets as d
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import distance as Di
from sklearn.metrics import mean_absolute_error


class Diversity:
    """ Compute the diversity between in the given dataset. We assume the dataset to be encoded. """

    def __init__(self, diversity_fn=None):
        """ Initialization
        :param diversity_fn: function to compute diversity with. If None, will use the default function """
        self.diversity_fn = diversity_fn if diversity_fn is not None else self.default_diversity

    def default_diversity(self, data, suffix=None):
        """ default diversity function."""
        if isinstance(data, d.Preprocessing):
            data = data.df
        try:
            distance = cdist(data, data, "euclidean").sum(1)
            distance /= np.sqrt(data.shape[1]) * (distance.shape[0] - 1)
        except MemoryError:
            distance = []
            for i in range(data.shape[0]):
                distance.append(np.sqrt(((data.iloc[i] - data) ** 2).sum(1)).sum())
            distance = np.array(distance) / ((data.shape[0] - 1) * np.sqrt(data.shape[1]))

        distance = distance.sum() / distance.shape[0]
        key = "diversity"
        if suffix is not None:
            key += "_{}".format(suffix)
        return {key: [distance]}

    def __call__(self, *args, **kwargs):
        return self.diversity_fn(*args, **kwargs)


class Damage:
    """ Compute the damage introduced both in categorical and numerical attributes """

    # attribute, type=num,cat, damage,

    def __init__(self, numerical_fn=None, categorical_in_list=False):
        """ Compute the damage between the two datasets.
        :param numerical_fn: function used to compute damage for numerical attributes
        :param categorical_in_list : if true, the value of the data is a list with the single value, instead of the value."""
        self.numerical_fn = numerical_fn if numerical_fn is not None else self.relative_change_resumed
        self.categorical_in_list = categorical_in_list

    def relative_change(self, original, transformed):
        """ Compute the relative change between the two given sets.
        Formula: |original - transformed| / f(original, transformed); with f(o,t) = 1/2 * (|o| + |t|) """

        return (original - transformed).abs() / ((original.abs() + transformed.abs()) / 2)

    def relative_change_resumed(self, original, transformed, quantiles=[0.5, 0.8]):
        """ Compute the relative change between the two given sets. Return the value resumed as points
        Formula: |original - transformed| / f(original, transformed); with f(o,t) = 1/2 * (|o| + |t|) """

        rc = self.relative_change(original, transformed)
        return rc.quantile(quantiles).T.to_dict()

    def damage_categorical(self, cat_orig, cat_transformed):
        """
        Compute the damage on categorical attributes
        :param cat_orig: categorical attributes. Original
        :param cat_transformed: categorical attributes. Transformed version
        :return: the damage computed
        """
        damage = {}
        wrapper = lambda x: x
        if self.categorical_in_list:
            wrapper = lambda x: [x]
        for c in cat_orig.columns:
            damage.update({c: wrapper((cat_orig[c].astype("object") != cat_transformed[c]).mean())})
        return damage

    def damage_numerical(self, num_orig, num_transformed):
        """
        Compute the damage on numerical attributes
        :param num_orig: numerical attributes original
        :param num_transformed: numerical attributes transformed
        :return: the damage in form of dataFrame.
        """
        return self.numerical_fn(num_orig, num_transformed)

    def __call__(self, original, transformed):
        if isinstance(original, d.Preprocessing):
            original = original.df
        if isinstance(transformed, d.Preprocessing):
            transformed = transformed.df

        cat_clm = original.select_dtypes(exclude=["int", "float", "double"]).columns.tolist()
        num_clm = original.select_dtypes(include=["int", "float", "double"]).columns.tolist()
        d_cat = self.damage_categorical(original[cat_clm], transformed[cat_clm])
        d_num = self.damage_numerical(original[num_clm], transformed[num_clm])

        return d_cat, d_num


class Distance:
    """ Compute the distance metric """

    def __init__(self, distance_metric=None):
        self._distance_metric = distance_metric
        self._distance_function = None
        if distance_metric is not None:
            self._distance_function = getattr(Di, distance_metric)
        self.euclidean_distance = getattr(Di, "euclidean")

    @staticmethod
    def _shaping(data):
        return data.values.reshape(-1) if isinstance(data, pd.DataFrame) else data.reshape(-1)

    def distance(self, o_test_set, test_set):
        """
        Compute the distance between the original test_set and the sanitized one
         We assume data are preprocessed
        :param o_test_set: the original test set
        :param test_set: the sanitized test set
        :return: the computed distance
        """
        if self._distance_metric is None:
            raise NotImplementedError
        o_test_set = self._shaping(o_test_set)
        test_set = self._shaping(test_set)
        d = self._distance_function(o_test_set, test_set)
        return {self._distance_metric: d}

    def mae(self, original, modified):
        """ return the mae between original and modified data """
        return mean_absolute_error(original, modified)

    def distance_mse(self, o_test_set, test_set):
        """ Compute the distance between the original test set and the sanitized one.
         We assume data are preprocessed
        :param o_test_set: the original test set
        :param test_set: the sanitized test set
        :return: the computed distance """

        o_test_set = self._shaping(o_test_set)
        test_set = self._shaping(test_set)
        distance = self.euclidean_distance(o_test_set, test_set)
        d = distance ** 2 / test_set.size
        return d
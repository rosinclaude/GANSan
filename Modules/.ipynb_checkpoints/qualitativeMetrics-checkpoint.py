import os
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
from Modules import datasets as d
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import distance as Di


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

    def __init__(self, numerical_fn=None):
        """ Compute the damage between the two datasets.
        :param numerical_fn: function used to compute damage for numerical attributes """
        self.numerical_fn = numerical_fn if numerical_fn is not None else self.relative_change_resumed

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
        for c in cat_orig.columns:
            damage.update({c: (cat_orig[c].astype("object") != cat_transformed[c]).mean()})
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


class ScoreRank:
    """ Compute the ranking preservation score between random rows """

    def __init__(self, exp_folder, num_cols, sample_size=100):
        """
        :param num_cols: list of numerical columns
        :param sample_size: number of sample to work with. -1 to use all data
        """
        self.sample_size = sample_size
        self.num_cols = num_cols
        self.exp_folder = exp_folder

    @staticmethod
    def rank_eval(col_orig, col_mapped, col_name):
        """  Construct the heatmap of relative percentage of gap between probabilities
            The obtained rate x would be read as new value = old value + x% of old value + c * x/100
            we choose c = 1 as all probabilities and their respective diff would be less than < 1.
            Thus avoiding nan and inf values
         """
        number_of_pairs = col_orig.shape[0] * (col_orig.shape[0] - 1) / 2
        matrices = []
        for col in [col_mapped, col_orig]:
            p_col = col.reshape(-1, 1)
            p_row = col.reshape(-1)
            matrices.append(p_col - p_row)
            # Matrice[r][c] =  prb[r] - prb[c]
        # mat[0] is mapped gap, mat[1] orig gap
        rank_diff = matrices[0] - matrices[1]
        rank_diff = np.round(rank_diff, 1)
        unique_rank_diff, freqs = np.unique(rank_diff, return_counts=True)
        freqs = freqs / rank_diff.size
        rk_props = pd.DataFrame(np.concatenate((unique_rank_diff.reshape(-1, 1), freqs.reshape(-1, 1)), 1),
                                columns=["rk-diff", "frequency"]).sort_values(by=["rk-diff"])
        rk_props["cumul_freq"] = rk_props["frequency"].cumsum()
        # Using seaborn is faster to plot heatmap
        heatmap = sns.heatmap(np.tril(rank_diff), center=0).figure
        # Bar plot
        # bar_plot = (p9.ggplot(data=rk_props, mapping=p9.aes(x='rk_diff', y='frequency')) +
        #             p9.geom_col()).draw(return_ggplot=False)
        # CDF plot
        # cdf_plot = (p9.ggplot(data=rk_props, mapping=p9.aes(x='rk_diff', y='cumul_freq')) +
        #             p9.geom_line()).draw(return_ggplot=False)

        for i in range(2):
            matrices[i] = np.tril(matrices[i])
        superior_original_mask = matrices[1] > 0
        inferior_original_mask = matrices[1] < 0
        equals_original_mask = matrices[1] < 0

        rank_inversions = {
            "sup-2-sup": (matrices[0][superior_original_mask] > 0).sum() / number_of_pairs,
            "sup-2-inf": (matrices[0][superior_original_mask] < 0).sum() / number_of_pairs,
            "sup-2-eql": (matrices[0][superior_original_mask] == 0).sum() / number_of_pairs,

            "inf-2-inf": (matrices[0][inferior_original_mask] < 0).sum() / number_of_pairs,
            "inf-2-sup": (matrices[0][inferior_original_mask] > 0).sum() / number_of_pairs,
            "inf-2-eql": (matrices[0][inferior_original_mask] == 0).sum() / number_of_pairs,

            "eql-2-eql": (matrices[0][equals_original_mask] == 0).sum() / number_of_pairs,
            "eql-2-sup": (matrices[0][equals_original_mask] > 0).sum() / number_of_pairs,
            "eql-2-inf": (matrices[0][equals_original_mask] < 0).sum() / number_of_pairs,
        }
        rank_inversions.update({"inversion": 1 - (rank_inversions["sup-2-sup"] + rank_inversions["inf-2-inf"] +
                                                  rank_inversions["eql-2-eql"])})
        rank_inversions.update({"column": col_name, "number_pairs": number_of_pairs, "number_els": col_orig.shape[0]})
        return heatmap, rk_props, rank_inversions

    def numerical_ranks(self, numerical_original, numerical_mapped, suffix=""):
        """ Compute the score ranking on numerical columns
            Just as with complex probability estimation.
            Compute the difference between indiv for each col. The idea of complex mapping is that it summarises all
            columns unto a single vector.
        """
        if len(suffix) != 0:
            suffix = "-{}".format(suffix)
        rank_and_props = pd.DataFrame()
        inversions = pd.DataFrame()
        for c in self.num_cols:
            heatmap, rk_props, rank_inversions = self.rank_eval(col_orig=numerical_original[c].values,
                                                                col_mapped=numerical_mapped[c].values, col_name=c)
            rk_props["A"] = c
            rank_and_props = pd.concat([rank_and_props, rk_props], axis=0)
            heatmap.savefig("{}/heatmap-{}{}.png".format(self.exp_folder.fig_dir, c, suffix))
            plt.close("all")
            inversions = inversions.append(rank_inversions, ignore_index=True)
        # Bar plot
        # bar_plot = (p9.ggplot(data=rank_and_props, mapping=p9.aes(x='rk-diff', y='frequency')) + p9.geom_col() +
        #             p9.facet_wrap('A'))
        # bar_plot.save("{}/bar_plot-{}.png".format(self.exp_folder.fig_dir, suffix), width=10, height=10, dpi=300)

        # CDF plot
        cdf_plot = (p9.ggplot(data=rank_and_props, mapping=p9.aes(x='rk-diff', y='cumul_freq')) + p9.geom_line() +
                    p9.facet_wrap('A'))
        cdf_plot.save("{}/cdf_plot-{}.png".format(self.exp_folder.fig_dir, suffix), width=10, height=10, dpi=300)
        return inversions

    def compute(self, original, mapped, suffix=""):
        if self.sample_size != -1:
            sample_size = min(self.sample_size, original.shape[0])
            original = original.sample(sample_size)
            mapped = mapped.sample(sample_size)
        inversions = self.numerical_ranks(numerical_original=original[self.num_cols],
                                          numerical_mapped=mapped[self.num_cols], suffix=suffix)
        return inversions

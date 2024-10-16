import warnings
import numpy as np
import pandas as pd
from Modules import classifiers as C
from scipy.spatial import distance as Di
from scipy.special import kl_div
from scipy.spatial import distance

from Modules import qualitativeMetrics as qm


# Transformation metrics, the quality of the transformation:
# Reconstruction of profiles with same control code, distance
# Distance between only transformed part (minority transformed)
# Distance between original and transformed set (all with same control code). This would be a sum of previous above,
# i.e if we compute maj to min and as we have min to min, we can just sum both.
# Control code prediction for those reconstructed profiles
# Control code prediction for transformed profiles
#
# Fairness: Transform all to either maj or either min. Transform only the opposite group
# Distance between groups (original vs transformed set(only minority))
# All majority: Fairness: Acc prediction target, demo par, indiv fairness (consistency), eqod. All in 4 scenarios
# All minority: Same as all majority
#
# Explainability: Damage can/will be seen in attribute changes.
#  - On transformed data (directly at the output of the decoder): control code predicted, if decision changed,
#  attributes that have changed (can draw histogram of different change to obtain positive/negative decision,
# they are actionable inputs
#  - On scenario Local fairness: Same as transformed data.

# Explainability by setting the decision as the control code: See attribute that change when moving from
# negative to positive and conversely.
# For explainabilitY: Most attribute changed, most combi of 2, 3 attributes changed. (And for each attribute, most
# changed value to what ...)


# Handle only majority and minority for now. Do not bother with all possible combination for now.


class GeneralMetrics:
    """
    General class with methods for computing specific metrics.
    """

    def __init__(self, distance_metric=None):
        self._distance_metric = distance_metric
        self._distance_function = None
        self._shaping = lambda data: data.values.reshape(-1) if isinstance(data, pd.DataFrame) else data.reshape(
            -1)
        if distance_metric is not None:
            self._distance_function = getattr(Di, distance_metric)
        self.euclidean_distance = getattr(Di, "euclidean")
        self.has_computed = False

    @staticmethod
    def Accuracy(predicted, targets, *args, **kwargs):
        """ Compute the accuracy of the prediction
        :param predicted: the predicted target
        :param targets: the ground truth
        :param args: others args just to have some homogeneity with other metrics such as the BER
        :param kwargs: same as args
        :return: The computed accuracy
        """
        if predicted is np.NaN:
            return np.NaN
        return np.abs(predicted == targets).mean()

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


class TransformationMetrics(GeneralMetrics):
    """ Compute the metrics for the transformation quality """

    def __init__(self, transformation, prediction, data_info, binary_sensitive, sel_prv_control, most_disc_control,
                 prefix_sep="=", train_size=70):
        # Function to transform the data (input, control) and to make prediction (input)
        super().__init__()
        self.transformation = transformation
        self.prediction = prediction

        self.train_size = train_size
        self.binary_sensitive = binary_sensitive
        self.len_sensitive = len(data_info["GroupAttributes"])

        self.data_info = data_info
        self.prefix_sep = prefix_sep

        self.control_attribute_names = data_info["GroupAttributes"]
        self.majority_group = data_info["DefaultGroupValue"]
        self.minority_group = data_info["ProtectedGroupValue"]
        if binary_sensitive:
            self.majority_group = [[1] for i in range(self.len_sensitive)]
            self.minority_group = [[0] for i in range(self.len_sensitive)]
        self.sel_prv_control = sel_prv_control
        self.most_disc_control = most_disc_control

        # Transform majority and minority into column names.
        self.map_sensitive()
        self.select_value = 1

        # self.sk_metrics = MetricsSkLearn()  ###### Add Initialisations data here
        # Initialise the classifiers
        self.control_clfs = C.MultiOutputClassifiers()
        self.mc_clfs = C.MultiOutputClassifiers()
        self.y_clfs = C.Classifiers()
        self.d_clfs = C.Classifiers()

        self.damage = qm.Damage()

        self.results = None
        self.reset_results()

    @staticmethod
    def __filter_features__(df, columns, prefix_sep="="):
        """
        Overriding the filter function of pandas, since we might have one-hot encoded columns.
        :param df: dataframe from which to remove columns
        :param columns: the columns to remove
        :param prefix_sep: the prefix used to separate attributes and their respective values when columns are encoded.
        :return: the filtered columns
        """
        cls = []
        if isinstance(columns, str):
            columns = [columns]
        for c in columns:
            if isinstance(c, list):
                cls.append(TransformationMetrics.__filter_features__(df, c, prefix_sep))
                continue
            # Handling encoded columns. Encoded attribute name.
            cls.extend(df.filter(regex="^{}{}".format(c, prefix_sep)).columns.tolist())
            # Handling other columns while avoiding to mistook some. Attribute name only.
            cls.extend(df.filter(regex="^{}$".format(c, )).columns.tolist())
        return cls

    def map_sensitive(self):
        """ """
        mag = []
        mig = []
        for i, n in enumerate(self.control_attribute_names):
            l = []
            for v in self.majority_group[i]:
                l.append("{}{}{}".format(n, self.prefix_sep, v))
            mag.append(l)
            l = []
            for v in self.minority_group[i]:
                l.append("{}{}{}".format(n, self.prefix_sep, v))
            mig.append(l)

        self.majority_group = mag
        self.minority_group = mig

    def get_group_data(self, data, group):
        """ Get the data for a given group, for instance, return only the majority group or the minority one."""
        mask = []
        for gr in group:
            m = []
            for key in gr:
                m.append(data[key] == self.select_value)
            mask.append(np.logical_or.reduce(m))
        return data[np.logical_and.reduce(mask)]

    def split_x_control(self, data):
        """ Split the columns and return x and the control columns. """
        c_names = self.__filter_features__(data, self.control_attribute_names, prefix_sep=self.prefix_sep)
        control = data[c_names]
        x = data.drop(c_names, axis=1)
        return x, control

    def reset_results(self):
        self.results = {
            "Attribute": [],
            "Classifier": [],
            "Metric": [],
            "Type": [],
            "Result": [],
            "Category": [],
            "Scenario": [],
            "Control": [],
            "ResultType": [],
        }

    def process_damage(self, data_v1, data_v2, control, data_prep, type_, result_type):
        """ Compute the damage and changes between data_v1 and data_v2 """
        dp_v1 = data_prep.copy(True)
        dp_v1.df = pd.concat([data_v1, control], axis=1, sort=True)[dp_v1.df.columns]
        dp_v1.inverse_transform()
        dp_v2 = data_prep.copy(True)
        dp_v2.df = pd.concat([data_v2, control], axis=1, sort=True)[dp_v2.df.columns]
        dp_v2.inverse_transform()
        cat, num = self.damage(dp_v1, dp_v2)
        for q, n in num.items():
            for c, v in n.items():
                self.add_results(Attribute=c, Classifier="-", Metric="Rel. Change", Type=type_, Result=v,
                                 Category="{} Quant.".format(q), Scenario="-", Control="-", ResultType=result_type)
        for c, v in cat.items():
            self.add_results(Attribute=c, Classifier="-", Metric="Cat. Damage", Type=type_, Result=v,
                             Category="-", Scenario="-", Control="-", ResultType=result_type)

    def add_results(self, **kwargs):
        for k, v in kwargs.items():
            self.results[k].append(v)
        for k in list(set(self.results.keys()) - set(kwargs.keys())):
            self.results[k].append("-")

    @staticmethod
    def extract_group(data, group_code):
        """ Extract from the data the selected group code indexes """
        return data.loc[group_code.index], data.drop(group_code.index, axis=0)

    def _for_kl_repeat(self, data, length):
        """ Repeat the data n times until n*data.shape[0] <= length, complete the length with random selection """
        size = data.shape[0]
        if size < length:
            size, length = length, size
        n = length // size
        r = length % size
        d = [data for i in range(n)] + [data.iloc[np.random.choice(range(size), r)]]
        d = np.concatenate(d, axis=0)
        return d

    def kl_div(self, in_group, out_of_group):
        """ Compute the Kl divergence between each group.
        If group length are not equals, will measure between each subgroup independently """
        if in_group.shape[0] < out_of_group.shape[0]:
            in_group = self._for_kl_repeat(in_group, length=out_of_group.shape[0])
        else:
            out_of_group = self._for_kl_repeat(out_of_group, length=in_group.shape[0])
        mean_kl = kl_div(in_group, out_of_group).mean()
        return mean_kl

    @staticmethod
    def x_y_groups_distance(x_group, y_group, metric="minkowski", p=1):
        """ Compute the mean and max distances between all of the points in x_group w.r.t y """
        d = distance.cdist(x_group, y_group, metric, p=p)
        # Maximal distance possible between,
        # maximal mean distance between a point of group x and the group y,
        # maximal mean distance between a point of group y and a the group x (all of the points of group x)
        # mean possible distance
        return d.max(), d.mean(1).max(), d.mean(0).max(), d.mean()

    def mapping(self, test_set, latent=True):
        """ Map data and return all """
        # 1- Remove the data code
        test_set_x, test_set_c = self.split_x_control(test_set)
        # 2- Split into different groups: Majority vs not majority, minority vs not minority
        maj_c_indexes = self.get_group_data(test_set_c, self.majority_group)
        t_maj_x, t_not_maj_x = self.extract_group(test_set_x, maj_c_indexes)
        min_c_indexes = self.get_group_data(test_set_c, self.minority_group)
        t_min_x, t_not_min_x = self.extract_group(test_set_x, min_c_indexes)
        # 3- Transform the data
        # Mapping to majority
        t_maj2maj_mapped = self.transformation(t_maj_x, maj_c_indexes, latent=latent, prv_mapping=True)
        t_not_maj2maj_mapped = self.transformation(t_not_maj_x, maj_c_indexes, latent=latent, prv_mapping=True)
        # Mapping to minority
        t_min2min_mapped = self.transformation(t_min_x, min_c_indexes, latent=latent, prv_mapping=False)
        t_not_min2min_mapped = self.transformation(t_not_min_x, min_c_indexes, latent=latent, prv_mapping=False)

        return maj_c_indexes, t_maj_x, t_not_maj_x, t_maj2maj_mapped, t_not_maj2maj_mapped, \
               min_c_indexes, t_min_x, t_not_min_x, t_min2min_mapped, t_not_min2min_mapped

    def dissimilarities(self, test_set):
        """ Compute the group dissimilarities metrics """
        # 1- Remove the data code
        # 2- Split into different groups: Majority vs not majority, minority vs not minority
        # 3- Transform the data

        _, _, _, t_maj2maj_latent, t_not_maj2maj_latent, \
        _, _, _, t_min2min_latent, t_not_min2min_latent = self.mapping(test_set, latent=True)

        # 4- Compute the distance between groups
        # Mapped on majority group: t_maj2maj_latent vs t_notmaj2maj_latent
        mx, mj2nmj, nmj2mj, mean = self.x_y_groups_distance(t_maj2maj_latent, t_not_maj2maj_latent)
        self.add_results(Attribute="-", Metric="Minkowski-p=1", Classifier="-", Type="Max Maj vs N.Maj", Result=mx,
                         Category="Latent Diss.", ResultType="N.A")
        self.add_results(Attribute="-", Metric="Minkowski-p=1", Classifier="-", Type="Mean Maj vs N.Maj", Result=mean,
                         Category="Latent Diss.", ResultType="N.A")
        self.add_results(Attribute="-", Metric="Minkowski-p=1", Classifier="-", Type="Max Mean pt. Maj vs gr. N.Maj",
                         Result=mj2nmj, Category="Latent Diss.", ResultType="N.A")
        self.add_results(Attribute="-", Metric="Minkowski-p=1", Classifier="-", Type="Max Mean pt. N.Maj vs gr. Maj",
                         Result=nmj2mj, Category="Latent Diss.", ResultType="N.A")

        # Mapped on majority group: t_maj2maj_latent vs t_notmaj2maj_latent
        mx, mn2nmn, nmn2mn, mean = self.x_y_groups_distance(t_min2min_latent, t_not_min2min_latent)
        self.add_results(Attribute="-", Metric="Minkowski-p=1", Classifier="-", Type="Max Min vs N.Min", Result=mx,
                         Category="Latent Diss.", ResultType="N.A")
        self.add_results(Attribute="-", Metric="Minkowski-p=1", Classifier="-", Type="Mean Min vs N.Min", Result=mean,
                         Category="Latent Diss.", ResultType="N.A")
        self.add_results(Attribute="-", Metric="Minkowski-p=1", Classifier="-", Type="Max Mean pt. Min vs gr. N.Min",
                         Result=mn2nmn, Category="Latent Diss.", ResultType="N.A")
        self.add_results(Attribute="-", Metric="Minkowski-p=1", Classifier="-", Type="Max Mean pt. N.Min vs gr. Min",
                         Result=nmn2mn, Category="Latent Diss.", ResultType="N.A")

    def base_rate_change(self, test_set):
        """ Compute the base rate change when mapping a data onto another group """
        maj_c_ind, maj_x, not_maj_x, t_maj2maj_rec, t_not_maj2maj_mapped, \
        min_c_ind, min_x, not_min_x, t_min2min_rec, t_not_min2min_mapped = self.mapping(test_set, latent=False)

        xy, control_features = self.split_x_control(test_set)
        named_data = {
            "Baseline": {"Baseline": xy},
            "Mixed":
                {
                    "All Maj.": pd.concat([maj_x, t_not_maj2maj_mapped], axis=0, sort=True).loc[xy.index],
                    "All Min.": pd.concat([min_x, t_not_min2min_mapped], axis=0, sort=True).loc[xy.index],
                },
            "Recon":
                {
                    "All Maj.": pd.concat([t_maj2maj_rec, t_not_maj2maj_mapped], axis=0, sort=True).loc[xy.index],
                    "All Min.": pd.concat([t_min2min_rec, t_not_min2min_mapped], axis=0, sort=True).loc[xy.index],
                },
        }

        scn = "-"
        clf = "B. Rate"
        # Per attribute fairness
        for rt, dt in named_data.items():
            for type_, data in dt.items():
                x, y = self.split_decision(data)
                for c in self.control_attribute_names:
                    control = control_features[self.__filter_features__(control_features, c,
                                                                        prefix_sep=self.prefix_sep)]
                    self.positive_rates(predicted=y.values.argmax(1), control_features=control,
                                        control_attribute_names=[c], scn=scn,
                                        type_=type_, attribute=self.data_info['LabelName'], clf=clf,
                                        result_type=rt)
                # All subgroups
                self.positive_rates(predicted=y.values.argmax(1), control_features=control_features,
                                    control_attribute_names=self.control_attribute_names, scn=scn,
                                    type_=type_, attribute=self.data_info['LabelName'], clf=clf,
                                    result_type=rt)

    def _prepare_targets_multi_outputs_(self, target_names, *targets):
        """ Format the control data for the multioutput prediction """
        t_columns = []
        targets_gmx = [[] for i in range(len(targets))]
        for c in target_names:
            ext_cls = self.__filter_features__(targets[0], c, self.prefix_sep)
            t_columns.append(ext_cls)
            for i in range(len(targets)):
                targets_gmx[i].append(targets[i][ext_cls].values.argmax(1))

        targets_gmx = [np.array(gmx).T for gmx in targets_gmx]

        return t_columns, targets_gmx

    def y_from_latent(self, train_set, test_set):
        """ predict the decision attribute from the latent space , with and without control only used with original
         decision """
        xy_train, c_train = self.split_x_control(train_set)
        _, y_train = self.split_decision(xy_train)
        xy_test, c_test = self.split_x_control(test_set)
        _, y_test = self.split_decision(xy_test)
        # Decision attribute is already removed in transformation fn
        x_tr_latent = self.transformation(xy_train, None, latent=True)
        x_ts_latent = self.transformation(xy_test, None, latent=True)

        def cat_l_c(latent, control):
            latent = pd.DataFrame(latent, columns=list(range(latent.shape[1])))
            return pd.concat([latent, control], axis=1, sort=True).values

        # Train predict and compute metric
        def up_to_metric(x_tr, y_tr, x_ts, y_ts, t):
            self.fairness_fit_predict_metrics(x_train=x_tr, y_train=y_tr, x_test=x_ts, y_test=y_ts,
                                              control_features=c_test, scn="From Latent", type_=t, result_type="N.A",
                                              train_clfs=True)

        up_to_metric(x_tr_latent, y_train, x_ts_latent, y_test, "latent 2 y Wo c")
        up_to_metric(cat_l_c(x_tr_latent, c_train), y_train, cat_l_c(x_ts_latent, c_test), y_test, "latent 2 y Wi c")

    def controls_from_latent(self, train_set, test_set):
        """ Try to predict the sensitive attribute from the latent space

        If decision not included during the training, the latent space does not include it also.
        """
        x_train, c_train = self.split_x_control(train_set)
        x_test, c_test = self.split_x_control(test_set)

        z_train = self.transformation(x_train, None, latent=True)
        z_test = self.transformation(x_test, None, latent=True)

        c_columns, c_mx = self._prepare_targets_multi_outputs_(self.control_attribute_names, c_train, c_test)
        c_train_mx, c_test_mx = c_mx

        # Train to predict the control attribute.
        self.control_clfs.fit(z_train, c_train_mx)
        # predict the control attribute
        for clf, p in self.control_clfs.predict(z_test).items():
            for i, c in enumerate(self.control_attribute_names):
                acc = self.Accuracy(p.T[i], c_test_mx.T[i])
                self.add_results(Attribute=c, Metric="Accuracy", Classifier=clf, Type="-", Result=acc, Scenario="-",
                                 Category="Latent 2 Control", ResultType="N.A")

    def distinguish(self, majority_data, not_majority_data, minority_data, not_minority_data, train_size=70):
        """ Train to distinguish between original and transformed data
        Data are both train and test concatenated.
        Limit to each group independently, the objective will be 0.5.
        If both group are merged, then the accuracy will be dependent of the size of the group to which data is mapped.
        i.e: if all mapped to maj, as we assume the reconstruction to be very good, maj data will be undifferentiable
        while the minority (mapped to maj) ... proportions dependant. Easier to test each group separately"""

        majority_data = majority_data.reset_index(drop=True)
        not_majority_data = not_majority_data.reset_index(drop=True)
        minority_data = minority_data.reset_index(drop=True)
        not_minority_data = not_minority_data.reset_index(drop=True)

        def _concat_rand_split_(x, xg):
            y = np.ones(x.shape[0])
            yg = np.zeros(xg.shape[0])
            x["True"] = y
            xg["True"] = yg
            xa = pd.concat([x, xg], axis=0, sort=True)
            xa = xa.sample(frac=1)
            xa.reset_index(drop=True, inplace=True)
            y = xa["True"]
            x = xa.drop(["True"], axis=1)
            return x, y

        def _compute_(orig, mapped, type_, random=False):
            # orig = orig.reset_index(drop=True)
            orig = orig.copy(True)
            mapped = mapped.copy(True)
            train_x = orig.sample(frac=train_size / 100)
            test_x = orig.drop(train_x.index, axis=0)
            r_prefix = "Chain"
            train_xg = mapped.loc[train_x.index]  # Same rows as train, but their mapped versions
            test_xg = mapped.drop(train_x.index, axis=0)
            if random:
                r_prefix = "Rand"
                train_xg = mapped.sample(frac=train_size / 100)  # Random rows
                test_xg = mapped.drop(train_xg.index, axis=0)
            train_xa, train_ya = _concat_rand_split_(train_x, train_xg)
            test_xa, test_ya = _concat_rand_split_(test_x, test_xg)
            self.d_clfs.fit(train_xa, train_ya.values)
            predicted = self.d_clfs.predict(test_xa)
            for clf, p in predicted.items():
                acc = self.Accuracy(p, test_ya.values)
                self.add_results(Attribute="-", Metric="Accuracy", Classifier=clf, Type=type_, Result=acc,
                                 Category="{}-Data Diff.".format(r_prefix))

        def _compute2_(orig, mapped, type_, result_type):
            # orig = orig.reset_index(drop=True)
            orig = orig.copy(True)
            mapped = mapped.copy(True)
            train_x, train_y = _concat_rand_split_(orig, mapped)
            test_x = train_x.sample(frac=(100 - train_size) / 100)
            test_y = train_y.loc[test_x.index]
            train_x = train_x.drop(test_x.index, axis=0)
            train_y = train_y.drop(test_x.index, axis=0)
            self.d_clfs.fit(train_x, train_y.values)
            predicted = self.d_clfs.predict(test_x)
            for clf, p in predicted.items():
                acc = self.Accuracy(p, test_y.values)
                self.add_results(Attribute="-", Metric="Accuracy", Classifier=clf, Type=type_, Result=acc,
                                 Category="Data Diff.", ResultType=result_type)

        maj_x, maj_c = self.split_x_control(majority_data)
        not_maj_x, _ = self.split_x_control(not_majority_data)
        min_x, min_c = self.split_x_control(minority_data)
        not_min_x, _ = self.split_x_control(not_minority_data)

        # Not min (Maj + other combinations) to minority
        not_min_to_min = self.transformation(not_min_x, min_c, prv_mapping=False)
        _compute2_(min_x, not_min_to_min, "Min-vs-N.Min 2 Min", "Mixed")
        min_rec = self.transformation(min_x, min_c, prv_mapping=False)
        _compute2_(min_rec, not_min_to_min, "Min-vs-N.Min 2 Min", "Recon")
        _compute2_(min_x, not_min_x, "Min-vs-N.Min", "Baseline")

        # mapped to Maj
        not_maj_to_maj = self.transformation(not_maj_x, maj_c, prv_mapping=True)
        _compute2_(maj_x, not_maj_to_maj, "Maj-vs-N.Maj 2 Maj", "Mixed")
        maj_rec = self.transformation(maj_x, maj_c, prv_mapping=True)
        _compute2_(maj_rec, not_maj_to_maj, "Maj-vs-N.Maj 2 Maj", "Recon")
        _compute2_(maj_x, not_maj_x, "Maj-vs-N.Maj", "Baseline")

    def distortions_and_control(self, majority_data, not_majority_data, minority_data, not_minority_data,
                                prefix="Test-"):
        d_category = "{}-Data Dist.".format(prefix)
        c_category = "{}-Control P.".format(prefix)
        # Majority same code
        maj_x, maj_c = self.split_x_control(majority_data)
        not_maj_x, not_maj_c = self.split_x_control(not_majority_data)
        maj_xg = self.transformation(maj_x, maj_c,
                                     prv_mapping=True)  # Same control, should get a very low reconstruction
        # maj_xg = pd.DataFrame(maj_xg, columns=maj_x.columns)
        d_maj = self.distance_mse(maj_x, maj_xg)
        self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="Maj 2 Maj", Result=d_maj,
                         Category=d_category, ResultType="Recon")
        # Minority same code
        min_x, min_c = self.split_x_control(minority_data)
        not_min_x, not_min_c = self.split_x_control(not_minority_data)
        min_xg = self.transformation(min_x, min_c,
                                     prv_mapping=False)  # Same control, should get a very low reconstruction
        # min_xg = pd.DataFrame(min_xg, columns=min_x.columns)
        d_min = self.distance_mse(min_x, min_xg)
        self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="Min 2 Min", Result=d_min,
                         Category=d_category, ResultType="Recon")
        # Not Minority to minority
        not_min_to_min = self.transformation(not_min_x, min_c, prv_mapping=False)
        # maj_to_min = pd.DataFrame(maj_to_min, columns=maj_x.columns)
        # Must have the least amount of modif to make a maj predicted as min
        d_maj_to_min = self.distance_mse(not_min_x, not_min_to_min)
        self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="N.Min 2 Min:N.Min",
                         Result=d_maj_to_min, Category=d_category, ResultType="Recon")
        # Not the same size, better use distance between distribution, such as kl div (min2maj, maj) ...
        # d_maj_to_min = self.distance_mse(min_x, maj_to_min)
        # self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="Maj 2 Min:Min", Result=d_maj_to_min,
        #                  Category="Data Dist.")
        # not Majority to Majority
        not_maj_to_maj = self.transformation(not_maj_x, maj_c, prv_mapping=True)
        # min_to_maj = pd.DataFrame(min_to_maj, columns=min_x.columns)
        # Must have the least amount of modif to make a min predicted as maj
        d_min_to_maj = self.distance_mse(not_maj_x, not_maj_to_maj)
        self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="N.Maj 2 Maj:N.Maj",
                         Result=d_min_to_maj, Category=d_category, ResultType="Recon")
        # d_min_to_maj = self.distance_mse(maj_x, min_to_maj)
        # self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="Min 2 Maj:Maj", Result=d_min_to_maj,
        #                  Category="Data Dist.")

        # real, control = self.prediction(test_data.values)

        # All originals
        data = pd.concat([maj_x, not_maj_x], axis=0, sort=True)
        data.reset_index(drop=True, inplace=True)
        data = data.sample(frac=1)
        targets_c = pd.concat([maj_c, not_maj_c], axis=0, sort=True)
        targets_c.reset_index(drop=True, inplace=True)
        targets_c = targets_c.loc[data.index]

        # All to min
        all2min = pd.concat((min_x, not_min_to_min), axis=0, sort=True).reset_index(drop=True)
        a2mn_codes = pd.concat((min_c, not_min_c), axis=0).reset_index(drop=True)
        train_a2mn_x = all2min.sample(frac=1).sample(frac=self.train_size / 100)
        train_a2mn_y = a2mn_codes.loc[train_a2mn_x.index]
        test_a2mn_x = all2min.drop(train_a2mn_x.index, axis=0)
        test_a2mn_y = a2mn_codes.loc[test_a2mn_x.index]

        # All to maj
        all2maj = pd.concat((maj_x, not_maj_to_maj), axis=0, sort=True).reset_index(drop=True)
        a2mj_codes = pd.concat((maj_c, not_maj_c), axis=0).reset_index(drop=True)
        train_a2mj_x = all2maj.sample(frac=1).sample(frac=self.train_size / 100)
        train_a2mj_y = a2mj_codes.loc[train_a2mj_x.index]
        test_a2mj_x = all2maj.drop(train_a2mj_x.index, axis=0)
        test_a2mj_y = a2mj_codes.loc[test_a2mj_x.index]

        # Train on all original with all original control
        _, c_mx = \
            self._prepare_targets_multi_outputs_(self.control_attribute_names, targets_c, maj_c, min_c,
                                                 self.most_disc_control, self.sel_prv_control)
        targets_c_gmx, maj_c_gmx, min_c_gmx, most_disc_c_gmx, sel_prv_c_gmx = c_mx
        self.control_clfs.fit(data, targets_c_gmx)
        # original majority accuracy
        self.control_predict_accuracy(maj_x, maj_c_gmx, type_="Org. Maj", category=c_category, result_type="Baseline")
        # Predict majority reconstructed. Expect perfect acc or same as original
        self.control_predict_accuracy(maj_xg, maj_c_gmx, type_="Rec. Maj", category=c_category, result_type="Recon")
        # original minority accuracy
        self.control_predict_accuracy(min_x, min_c_gmx, type_="Org. Min", category=c_category, result_type="Baseline")
        # Predict minority reconstructed. Expect perfect acc or same as original. Must tell that all data is min
        self.control_predict_accuracy(min_xg, min_c_gmx, type_="Rec. Min", category=c_category, result_type="Recon")
        # Predict not minority transformed to min. Must tell that all data is min
        self.control_predict_accuracy(not_min_to_min,
                                      np.repeat(most_disc_c_gmx, not_min_to_min.shape[0], axis=0),
                                      type_="N.Min 2 Min:Min", category=c_category, result_type="Recon")
        self.mcontrol_fit_predict_accuracy(train_a2mn_x, train_a2mn_y, test_x=test_a2mn_x, test_y=test_a2mn_y,
                                           type_="N.Min 2 Min:N.Min", category=c_category, result_type="Mixed")
        # Predict not majority transformed to maj. Must tell that all data is maj
        self.control_predict_accuracy(not_maj_to_maj,
                                      np.repeat(sel_prv_c_gmx, not_maj_to_maj.shape[0], axis=0),
                                      type_="N.Maj 2 Maj:Maj", category=c_category, result_type="Recon")
        self.mcontrol_fit_predict_accuracy(train_a2mj_x, train_a2mj_y, test_x=test_a2mj_x, test_y=test_a2mj_y,
                                           type_="N.Maj 2 Maj:N.Maj", category=c_category, result_type="Mixed")

        # for c in self.control_attribute_names:
        #     c_columns = self.__filter_features__(maj_c, c, prefix_sep=self.prefix_sep)
        # Train on all original with all original control
        # targets = targets_c[c_columns]
        # self.control_clfs.fit(data, targets.values.argmax(1))
        # original majority accuracy
        # self.control_predict_accuracy(maj_x, maj_c[c_columns], type_="Org. Maj", attribute=c, category=c_category,
        #                               result_type="Baseline")
        # Predict majority reconstructed. Expect perfect acc or same as original
        # self.control_predict_accuracy(maj_xg, maj_c[c_columns], type_="Rec. Maj", attribute=c,
        #                               category=c_category, result_type="Recon")
        # original minority accuracy
        # self.control_predict_accuracy(min_x, min_c[c_columns], type_="Org. Min", attribute=c, category=c_category,
        #                               result_type="Baseline")
        # Predict minority reconstructed. Expect perfect acc or same as original. Must tell that all data is min
        # self.control_predict_accuracy(min_xg, min_c[c_columns], type_="Rec. Min", attribute=c,
        #                               category=c_category, result_type="Recon")
        # Predict not minority transformed to min. Must tell that all data is min
        # self.control_predict_accuracy(not_min_to_min,
        #                               np.repeat(self.most_disc_control.iloc[:1].values, not_min_to_min.shape[0],
        #                                         axis=0),
        #                               type_="N.Min 2 Min:Min", attribute=c, category=c_category,
        #                               result_type="Recon")
        # self.mcontrol_fit_predict_accuracy(train_a2mn_x, train_a2mn_y[c_columns], test_x=test_a2mn_x,
        #                                    test_y=test_a2mn_y[c_columns], type_="N.Min 2 Min:N.Min", attribute=c,
        #                                    category=c_category, result_type="Recon")
        # self.control_predict_accuracy(not_min_to_min, maj_c[c_columns], type_="Not Min 2 Min:Maj", attribute=c,
        #                               category=c_category)
        # Predict not majority transformed to maj. Must tell that all data is maj
        # self.control_predict_accuracy(not_maj_to_maj,
        #                               np.repeat(self.sel_prv_control.iloc[:1].values, not_maj_to_maj.shape[0],
        #                                         axis=0),
        #                               type_="N.Maj 2 Maj:Maj", attribute=c, category=c_category,
        #                               result_type="Recon")
        # self.control_predict_accuracy(not_maj_to_maj, min_c[c_columns], type_="Min 2 Maj:Min", attribute=c,
        #                               category=c_category)
        # self.mcontrol_fit_predict_accuracy(train_a2mj_x, train_a2mj_y[c_columns], test_x=test_a2mj_x,
        #                                    test_y=test_a2mj_y[c_columns], type_="N.Maj 2 Maj:N.Maj", attribute=c,
        #                                    category=c_category, result_type="Recon")

    def control_predict_accuracy(self, test_data, c_test_gmx, type_, category, result_type):
        """ Predict and compute the accuracy. Mostly used for the control prediction """
        # Predict
        p_controls = self.control_clfs.predict(test_data)
        for clf, p in p_controls.items():
            for i, c in enumerate(self.control_attribute_names):
                acc = self.Accuracy(p.T[i], c_test_gmx.T[i])
                self.add_results(Attribute=c, Metric="Accuracy", Classifier=clf, Type=type_, Result=acc,
                                 Category=category, Scenario="Trained.on.Orig", ResultType=result_type)

    def control_predict_accuracy_v1(self, test_data, test_target, type_, attribute, category, result_type):
        """ Predict and compute the accuracy. Mostly used for the control prediction """
        if isinstance(test_target, pd.DataFrame):
            test_target = test_target.values
        test_target = test_target.argmax(1)
        # Predict
        p_controls = self.control_clfs.predict(test_data)
        for clf, p in p_controls.items():
            acc = self.Accuracy(p, test_target)
            self.add_results(Attribute=attribute, Metric="Accuracy", Classifier=clf, Type=type_, Result=acc,
                             Category=category, Scenario="Trained.on.Orig", ResultType=result_type)

    def mcontrol_fit_predict_accuracy(self, train_x, train_y, test_x, test_y, type_, category, result_type):
        """ Train on original and mapped data, try to predict accurately the given control code.
         If the approach succeed, the accuracy should equal that of the orig control code. For each control,
         the accuracy should not be greater than the proportion of its mapped value in the dataset
         i.e: if mapped to majorty and control are attributes [A and B], the acc of A should not be greater than the
         proportion of value of A corresponding to the majority as all are mapped to majority, hence should be predicted
         as having the same value, which is only true for the not modified data. (The distinguish will tell us if real
         and fake are distinguishible.  Or the classifier can predict only the mst probable class in the set all the time."""

        _, c_mx = self._prepare_targets_multi_outputs_(self.control_attribute_names, train_y, test_y)
        c_train_gmx, c_test_gmx = c_mx
        # fit
        self.mc_clfs.fit(train_x, c_train_gmx)
        # Predict
        p_controls = self.mc_clfs.predict(test_x)
        for clf, p in p_controls.items():
            for i, c in enumerate(self.control_attribute_names):
                acc = self.Accuracy(p.T[i], c_test_gmx.T[i])
                self.add_results(Attribute=c, Metric="Accuracy", Classifier=clf, Type=type_, Result=acc,
                                 Category=category, Scenario="Trained.on.Mixed", ResultType=result_type)

    def mcontrol_fit_predict_accuracy_v1(self, train_x, train_y, test_x, test_y, type_, attribute,
                                         category, result_type):
        """ Train on original and mapped data, try to predict accurately the given control code.
         If the approach succeed, the accuracy should equal that of the orig control code. For each control,
         the accuracy should not be greater than the proportion of its mapped value in the dataset
         i.e: if mapped to majorty and control are attributes [A and B], the acc of A should not be greater than the
         proportion of value of A corresponding to the majority as all are mapped to majority, hence should be predicted
         as having the same value, which is only true for the not modified data. (The distinguish will tell us if real
         and fake are distinguishible.  Or the classifier can predict only the mst probable class in the set all the time."""

        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.values
        train_y = train_y.argmax(1)
        if isinstance(test_y, pd.DataFrame):
            test_y = test_y.values
        test_y = test_y.argmax(1)

        # fit
        self.mc_clfs.fit(train_x, train_y)
        # Predict
        p_controls = self.mc_clfs.predict(test_x)
        for clf, p in p_controls.items():
            acc = self.Accuracy(p, test_y)
            self.add_results(Attribute=attribute, Metric="Accuracy", Classifier=clf, Type=type_, Result=acc,
                             Category=category, Scenario="Trained.on.Mixed", ResultType=result_type)

    def split_decision(self, data):
        y_names = self.__filter_features__(data, self.data_info["LabelName"], prefix_sep=self.prefix_sep)
        y = data[y_names]
        x = data.drop(y_names, axis=1)
        return x, y

    def positive_rates__(self, predicted, group_features, scn, type_, attribute, control_attr, clf, result_type,
                         decision_positive=1):
        """ Compute the positive rate for the given attribute groups """
        positive_predictions = (predicted == decision_positive) * 1
        groups = group_features.values.argmax(1)
        mapping = dict(
            zip(
                list(range(group_features.shape[1])), group_features.columns
            )
        )
        d = pd.DataFrame.from_dict({"pos_pred": positive_predictions, "groups": groups})
        d = d.groupby(by="groups").mean()
        d.reset_index(inplace=True)
        d.replace({"groups": mapping}, inplace=True)
        d.set_index("groups", inplace=True)
        pos = d["pos_pred"].to_dict()
        for cat, res in pos.items():
            self.add_results(Attribute=attribute, Metric="Pos. Rate", Classifier=clf, Type=type_, Scenario=scn,
                             Result=res, Category=cat, Control=control_attr, ResultType=result_type)

    def get_suffix(self, name):
        return self.prefix_sep.join(name.split(self.prefix_sep)[1:])

    def groups_mapping_controls(self, control_features, control_attribute_names):
        # Identify all possible subgroups
        groups = control_features.groupby(by=self.__filter_features__(control_features, control_attribute_names,
                                                                      self.prefix_sep)).ngroup()
        mapping = {}
        for g in groups.value_counts().index:
            sample = control_features[groups == g].iloc[:1]
            v_name = ""
            for c in control_attribute_names:
                cols = sample[self.__filter_features__(sample, c, self.prefix_sep)]
                c_index = cols.values.argmax(1)
                col = cols.columns[c_index].tolist()[0]
                v_name += "{}-".format(self.get_suffix(col))
            v_name = v_name[:-1]
            mapping.update({g: v_name})
        a_name = "-".join(control_attribute_names)
        return groups, mapping, a_name

    def positive_rates(self, predicted, control_features, control_attribute_names, scn, type_, attribute, clf,
                       result_type, decision_positive=1):
        """ Compute the positive rate for all given subgroups """
        groups, mapping, a_name = self.groups_mapping_controls(control_features, control_attribute_names)
        if predicted is np.NaN:
            for g, cat in mapping.items():
                self.add_results(Attribute=attribute, Metric="Pos. Rate", Classifier=clf, Type=type_, Scenario=scn,
                                 Result=np.NaN, Category=cat, Control=a_name, ResultType=result_type)
        else:
            positive_predictions = (predicted == decision_positive) * 1
            positive_predictions = positive_predictions.reshape(groups.shape)
            d = pd.DataFrame.from_dict({"pos_pred": positive_predictions, "groups": groups})
            d = d.groupby(by="groups").mean()
            d.reset_index(inplace=True)
            d.replace({"groups": mapping}, inplace=True)
            d.set_index("groups", inplace=True)
            pos = d["pos_pred"].to_dict()
            for cat, res in pos.items():
                self.add_results(Attribute=attribute, Metric="Pos. Rate", Classifier=clf, Type=type_, Scenario=scn,
                                 Result=res, Category=cat, Control=a_name, ResultType=result_type)

    def tp_and_fp(self, predicted, targets, control_features, control_attribute_names, scn, type_, attribute,
                  clf, result_type, decision_positive=1):
        """ Compute the true positive and false positive rates. """

        groups, mapping, a_name = self.groups_mapping_controls(control_features, control_attribute_names)
        if predicted is np.NaN:
            for g, cat in mapping.items():
                self.add_results(Attribute=attribute, Metric="Tp. Rate", Classifier=clf, Type=type_, Scenario=scn,
                                 Result=np.NaN, Category=cat, Control=a_name, ResultType=result_type)
                self.add_results(Attribute=attribute, Metric="Fp. Rate", Classifier=clf, Type=type_, Scenario=scn,
                                 Result=np.NaN, Category=cat, Control=a_name, ResultType=result_type)
        else:
            positive_mask = targets == decision_positive
            predicted = predicted.reshape(targets.shape)
            tp = (predicted[positive_mask] == decision_positive) * 1
            fp = (predicted[~positive_mask] == decision_positive) * 1
            groups = groups.values.reshape(targets.shape)

            def add_missing_group(miss_v, df, c_name, g_name="groups"):
                if len(miss_v):
                    empty = pd.DataFrame.from_dict({g_name: miss_v, c_name: [np.NaN for i in miss_v]})
                    empty.set_index(g_name, inplace=True)
                    return pd.concat([df, empty], axis=0)
                return df

            # True positives
            g = groups[positive_mask]
            d_tp = pd.DataFrame.from_dict({"tp": tp, "groups": g})
            d_tp = d_tp.groupby(by="groups").mean()
            miss_v = np.setdiff1d(np.unique(groups), np.unique(g))
            d_tp = add_missing_group(miss_v, d_tp, "tp", "groups")
            d_tp.reset_index(inplace=True)
            d_tp.replace({"groups": mapping}, inplace=True)
            d_tp.set_index("groups", inplace=True)
            for cat, res in d_tp["tp"].items():
                self.add_results(Attribute=attribute, Metric="Tp. Rate", Classifier=clf, Type=type_, Scenario=scn,
                                 Result=res, Category=cat, Control=a_name, ResultType=result_type)

            # False positives
            g = groups[~positive_mask]
            d_fp = pd.DataFrame.from_dict({"fp": fp, "groups": g})
            d_fp = d_fp.groupby(by="groups").mean()
            miss_v = np.setdiff1d(np.unique(groups), np.unique(g))
            d_fp = add_missing_group(miss_v, d_fp, "fp", "groups")
            d_fp.reset_index(inplace=True)
            d_fp.replace({"groups": mapping}, inplace=True)
            d_fp.set_index("groups", inplace=True)
            for cat, res in d_fp["fp"].items():
                self.add_results(Attribute=attribute, Metric="Fp. Rate", Classifier=clf, Type=type_, Scenario=scn,
                                 Result=res, Category=cat, Control=a_name, ResultType=result_type)

    def fairness_fit_predict_metrics(self, x_train, y_train, x_test, y_test, control_features, scn, type_, result_type,
                                     train_clfs=True, arg_max=True):
        """ Fit predict and compute the fairness metrics. """
        y_train = y_train.values
        y_test = y_test.values
        if arg_max:
            y_train = y_train.argmax(1)
            y_test = y_test.argmax(1)
        if train_clfs:
            self.y_clfs.fit(x_train, y_train)
        pred_y = self.y_clfs.predict(x_test)
        y_test_arg = y_test
        for clf, p in pred_y.items():
            acc = self.Accuracy(p, y_test_arg)
            self.add_results(Attribute=self.data_info['LabelName'], Metric="Accuracy", Classifier=clf,
                             Type=type_, Result=acc, Scenario=scn, Category="-", ResultType=result_type)
            # Per attribute fairness
            for c in self.control_attribute_names:
                control = control_features[self.__filter_features__(control_features, c,
                                                                    prefix_sep=self.prefix_sep)]
                self.positive_rates(predicted=p, control_features=control, control_attribute_names=[c], scn=scn,
                                    type_=type_, attribute=self.data_info['LabelName'], clf=clf,
                                    result_type=result_type)
                self.tp_and_fp(predicted=p, targets=y_test_arg, control_features=control, control_attribute_names=[c],
                               scn=scn, type_=type_, attribute=self.data_info['LabelName'], clf=clf,
                               result_type=result_type)
            # All subgroups
            self.positive_rates(predicted=p, control_features=control_features,
                                control_attribute_names=self.control_attribute_names, scn=scn,
                                type_=type_, attribute=self.data_info['LabelName'], clf=clf,
                                result_type=result_type)
            self.tp_and_fp(predicted=p, targets=y_test_arg, control_features=control_features,
                           control_attribute_names=self.control_attribute_names,
                           scn=scn, type_=type_, attribute=self.data_info['LabelName'], clf=clf,
                           result_type=result_type)

    def fairness_and_y_accuracies(self, train_set, test_set, data_prep, original_y=False):
        """ Compute fairness metrics """

        # Baseline: Fit, predict, save.
        # Split x and control attributes
        train_set_xy, train_set_c = self.split_x_control(train_set)
        test_set_xy, test_set_c = self.split_x_control(test_set)
        # Split x and y, fit, predict and compute metrics
        train_set_x, train_set_y = self.split_decision(train_set_xy)
        test_set_x, test_set_y = self.split_decision(test_set_xy)
        # fit, predict and metrics
        self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y, x_test=test_set_x,
                                          y_test=test_set_y, control_features=test_set_c, scn="Baseline",
                                          type_="Baseline", result_type="Baseline")

        # Other scn Transform: min or maj, fit, predict
        # Get Majority/Minority control code
        train_set_c_maj = self.get_group_data(train_set_c, self.majority_group)
        test_set_c_maj = self.get_group_data(test_set_c, self.majority_group)
        # Get minority and majority data: data = data_min + data_maj
        # train_set_xy_maj = self.get_group_data(train_set_xy, self.majority_group)
        # train_set_xy_min = self.get_group_data(train_set_xy, self.minority_group)
        # test_set_xy_maj = self.get_group_data(test_set_xy, self.majority_group)
        # test_set_xy_min = self.get_group_data(test_set_xy, self.minority_group)
        train_set_xy_maj = train_set_xy.loc[train_set_c_maj.index]
        train_set_xy_not_maj = train_set_xy.drop(train_set_c_maj.index, axis=0)
        test_set_xy_maj = test_set_xy.loc[test_set_c_maj.index]
        test_set_xy_not_maj = test_set_xy.drop(test_set_c_maj.index, axis=0)

        # Minority
        train_set_c_min = self.get_group_data(train_set_c, self.minority_group)
        test_set_c_min = self.get_group_data(test_set_c, self.minority_group)
        train_set_xy_min = train_set_xy.loc[train_set_c_min.index]
        train_set_xy_not_min = train_set_xy.drop(train_set_c_min.index, axis=0)
        test_set_xy_min = test_set_xy.loc[test_set_c_min.index]
        test_set_xy_not_min = test_set_xy.drop(test_set_c_min.index, axis=0)

        # Order matter especially for scenario involving the original decision.

        # Mapping all to majority
        # train
        train_set_xy_not_maj_to_maj = self.transformation(train_set_xy_not_maj, train_set_c_maj, prv_mapping=True)
        train_set_xy_maj_to_maj = self.transformation(train_set_xy_maj, train_set_c_maj, prv_mapping=True)
        # Test
        test_set_xy_not_maj_to_maj = self.transformation(test_set_xy_not_maj, test_set_c_maj, prv_mapping=True)
        test_set_xy_maj_to_maj = self.transformation(test_set_xy_maj, test_set_c_maj, prv_mapping=True)

        # Mapping all to minority
        # train
        train_set_xy_not_min_to_min = self.transformation(train_set_xy_not_min, train_set_c_min, prv_mapping=False)
        train_set_xy_min_to_min = self.transformation(train_set_xy_min, train_set_c_min, prv_mapping=False)
        # Test
        test_set_xy_not_min_to_min = self.transformation(test_set_xy_not_min, test_set_c_min, prv_mapping=False)
        test_set_xy_min_to_min = self.transformation(test_set_xy_min, test_set_c_min, prv_mapping=False)

        # Re-form whole dataset
        for rt, mj, mn in zip(("Mixed", "Recon"),
                              ({"tr": train_set_xy_maj, "ts": test_set_xy_maj},
                               {"tr": train_set_xy_maj_to_maj, "ts": test_set_xy_maj_to_maj}),
                              ({"tr": train_set_xy_min, "ts": test_set_xy_min},
                               {"tr": train_set_xy_min_to_min, "ts": test_set_xy_min_to_min})):
            # Majority
            train_set_xy_all_maj = pd.concat([mj["tr"], train_set_xy_not_maj_to_maj], axis=0, sort=True)
            # Set the same order as the original data
            train_set_xy_all_maj = train_set_xy_all_maj.loc[train_set_xy.index]
            # Test
            test_set_xy_all_maj = pd.concat([mj["ts"], test_set_xy_not_maj_to_maj], axis=0, sort=True)
            # Set the same order as the original data
            test_set_xy_all_maj = test_set_xy_all_maj.loc[test_set_xy.index]

            # Minority
            # Train
            train_set_xy_all_min = pd.concat([mn["tr"], train_set_xy_not_min_to_min], axis=0, sort=True)
            # Set the same order as the original data
            train_set_xy_all_min = train_set_xy_all_min.loc[train_set_xy.index]
            # Test
            test_set_xy_all_min = pd.concat([mn["ts"], test_set_xy_not_min_to_min], axis=0, sort=True)
            # Set the same order as the original data
            test_set_xy_all_min = test_set_xy_all_min.loc[test_set_xy.index]

            # Compute damage
            self.process_damage(test_set_xy, test_set_xy_all_maj, test_set_c_maj, data_prep, type_="All Maj.",
                                result_type=rt)
            self.process_damage(test_set_xy, test_set_xy_all_min, test_set_c_min, data_prep, type_="All Min.",
                                result_type=rt)

            # Fairness metrics
            # Majority
            # Split decisions
            train_set_x_all_maj, train_set_y_all_maj = self.split_decision(train_set_xy_all_maj)
            test_set_x_all_maj, test_set_y_all_maj = self.split_decision(test_set_xy_all_maj)
            # Scenario
            # S1: Complete modifs: train on all data modif with modif decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_maj, y_train=train_set_y_all_maj,
                                              x_test=test_set_x_all_maj, y_test=test_set_y_all_maj,
                                              control_features=test_set_c, scn="S1", type_="All Maj.", result_type=rt)
            # S2: Partial modifs: train on data modif but with orig decisions.
            # If gan model is trained without decision, then we keep the decision as original. S2 == S1
            if not original_y:
                self.fairness_fit_predict_metrics(x_train=train_set_x_all_maj, y_train=train_set_y,
                                                  x_test=test_set_x_all_maj, y_test=test_set_y,
                                                  control_features=test_set_c, scn="S2", type_="All Maj.",
                                                  result_type=rt)
            # S3-y: Fair Classifier: train on modif data with orig dec and test on orig data
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_maj, y_train=train_set_y,
                                              x_test=test_set_x, y_test=test_set_y,
                                              control_features=test_set_c, scn="S3-y", type_="All Maj.", result_type=rt,
                                              train_clfs=original_y)
            # S3-y_hat: Fair Classifier: train modif data with modif dec and test on orig data
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_maj, y_train=train_set_y_all_maj,
                                              x_test=test_set_x, y_test=test_set_y,
                                              control_features=test_set_c, scn="S3-y_hat", type_="All Maj.",
                                              result_type=rt)
            # S4-y: Local San: train on orig data and test on modified data but with orig decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y,
                                              x_test=test_set_x_all_maj, y_test=test_set_y,
                                              control_features=test_set_c, scn="S4-y", type_="All Maj.", result_type=rt)
            # S4-y_hat: Local San: train on orig data and test on modified data but with modified decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y,
                                              x_test=test_set_x_all_maj, y_test=test_set_y_all_maj,
                                              control_features=test_set_c, scn="S4-y_hat", type_="All Maj.",
                                              result_type=rt, train_clfs=False)

            # Mapping all to minority
            # Split decisions
            train_set_x_all_min, train_set_y_all_min = self.split_decision(train_set_xy_all_min)
            test_set_x_all_min, test_set_y_all_min = self.split_decision(test_set_xy_all_min)
            # Scenario
            # S1: Complete modifs: train on all data modif with modif decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_min, y_train=train_set_y_all_min,
                                              x_test=test_set_x_all_min, y_test=test_set_y_all_min,
                                              control_features=test_set_c, scn="S1", type_="All Min.", result_type=rt)
            # S2: Partial modifs: train on data modif but with orig decisions
            # If gan model is trained without decision, then we keep the decision as original. S2 == S1
            if not original_y:
                self.fairness_fit_predict_metrics(x_train=train_set_x_all_min, y_train=train_set_y,
                                                  x_test=test_set_x_all_min, y_test=test_set_y,
                                                  control_features=test_set_c, scn="S2", type_="All Min.",
                                                  result_type=rt)
            # S3-y: Fair Classifier: train on modif data with orig dec and test on orig data
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_min, y_train=train_set_y,
                                              x_test=test_set_x, y_test=test_set_y,
                                              control_features=test_set_c, scn="S3-y", type_="All Min.", result_type=rt,
                                              train_clfs=original_y)
            # S3-y_hat: Fair Classifier: train modif data with modif dec and test on orig data
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_min, y_train=train_set_y_all_min,
                                              x_test=test_set_x, y_test=test_set_y,
                                              control_features=test_set_c, scn="S3-y_hat", type_="All Min.",
                                              result_type=rt)
            # S4-y: Local San: train on orig data and test on modified data but with orig decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y,
                                              x_test=test_set_x_all_min, y_test=test_set_y,
                                              control_features=test_set_c, scn="S4-y", type_="All Min.", result_type=rt)
            # S4-y_hat: Local San: train on orig data and test on modified data but with modified decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y,
                                              x_test=test_set_x_all_min, y_test=test_set_y_all_min,
                                              control_features=test_set_c, scn="S4-y_hat", type_="All Min.",
                                              result_type=rt, train_clfs=original_y)

    def compute_metrics(self, train_set, test_set, data_prep, stop_restart, original_y=False):
        """ Compute the relevant metrics """
        # Check only the last scenario as we compute it at last. If it saved, then all is computed
        # Always take care of indexes... Especially when cutting and cating data...
        self.has_computed = False
        if not stop_restart.computed(Scenario="S4-y_hat", Type="All Min."):
            train_set = train_set.reset_index(drop=True)
            test_set = test_set.reset_index(drop=True)
            all_data = pd.concat([train_set, test_set], axis=0, sort=True)
            all_data.reset_index(drop=True, inplace=True)
            majority_data = self.get_group_data(all_data, self.majority_group)
            minority_data = self.get_group_data(all_data, self.minority_group)
            not_maj_data = all_data.drop(majority_data.index, axis=0)
            not_min_data = all_data.drop(minority_data.index, axis=0)
            ts_majority_data = self.get_group_data(test_set, self.majority_group)
            ts_minority_data = self.get_group_data(test_set, self.minority_group)

            if original_y:
                self.y_from_latent(train_set, test_set)
            self.distinguish(majority_data, not_maj_data, minority_data, not_min_data, train_size=self.train_size)

            self.controls_from_latent(train_set, test_set)

            self.distortions_and_control(ts_majority_data, test_set.drop(ts_majority_data.index, axis=0),
                                         ts_minority_data, test_set.drop(ts_minority_data.index, axis=0),
                                         prefix="Test-")
            self.distortions_and_control(majority_data, not_maj_data, minority_data, not_min_data, prefix="All-")

            self.fairness_and_y_accuracies(train_set, test_set, data_prep, original_y)

            self.dissimilarities(test_set)

            self.base_rate_change(test_set)
            self.has_computed = True

    def explainability(self):
        """ Compute metrics and visualization for the explainability part """

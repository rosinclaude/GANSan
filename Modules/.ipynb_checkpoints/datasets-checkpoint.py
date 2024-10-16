import copy
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from sklearn.preprocessing import MinMaxScaler
import pickle


class Preprocessing:
    """
    Class to handle all the preprocessing steps before passing the dataset to pytorch
    """

    def __init__(self, csv, numeric_as_categoric_max_thr=5, group_and_decision_excluded=[],
                 scale=0, prep_excluded=None, prep_included=None):
        """

        :param csv: the path to the dataset
        :param numeric_as_categoric_max_thr: Maxiimum number of values for a numeric column to be considered as
        categoric
        :param group_and_decision_exlcuded: if group attribute and the decision ones are numeric, exclude them from
        being transformed as categorical using the numeric_as_categoric_max_thr
        :param scale: the lower bound of the feature scaling. Either 0 or -1
        :param prep_excluded: Columns to exclude from the preprocessing
        :param prep_included: Columns to include from the preprocessing. This overwrite the prep_excluded
        """
        if isinstance(csv, str):
            self.df = pd.read_csv(csv)
        elif isinstance(csv, pd.DataFrame):
            self.df = csv.copy(True)
        else:
            raise ValueError("Invalid input type: {}".format(csv))

        self.cols_order = self.df.columns
        self.prep_excluded = prep_excluded
        self.prep_included = prep_included
        if prep_included is not None:
            self.prep_excluded = list(set(self.cols_order) - set(prep_included))
        self.encoded_features_order = None
        self.categorical_was_numeric = {}
        self.num_as_cat = numeric_as_categoric_max_thr
        self.group_decision_excluded = group_and_decision_excluded
        p = []
        if self.prep_excluded is not None:
            p = self.prep_excluded[:]
        if self.group_decision_excluded is not None:
            p += self.group_decision_excluded[:]
        self.prep_excluded = None if len(p) == 0 else p
        self.scale = scale
        self.scaler = MinMaxScaler(feature_range=(scale, 1))
        self.cat_clm = None
        self.num_clm = None
        self.int_clm = None
        self.transformed = False

    def save_parameters(self, prmPath, prmFile="parameters.prm"):
        """
        Dump the parameters on the disc
        :param prmPath: location where to save params
        """
        o = {
            "cols_order": self.cols_order,
            "prep_excluded": self.prep_excluded,
            "prep_included": self.prep_included,
            "encoded_features_order": self.encoded_features_order,
            "categorical_was_numeric": self.categorical_was_numeric,
            "num_as_cat": self.num_as_cat,
            "group_decision_excluded": self.group_decision_excluded,
            "scale": self.scale,
            "cat_clm": self.cat_clm,
            "num_clm": self.num_clm,
            "int_clm": self.int_clm,
            "scaler": self.scaler,
        }
        prmLoc = "{}/{}".format(prmPath, prmFile)
        if prmFile is None or prmFile in prmPath:
            prmLoc = prmPath

        pickle.dump(o, open(prmLoc, "wb"))

    def load_parameters(self, prmPath=None, prmFile="parameters.prm", prm_dict=None):
        prmLoc = "{}/{}".format(prmPath, prmFile)
        if prmPath is not None:
            if prmFile is None or prmFile in prmPath:
                prmLoc = prmPath
            o = pickle.load(open(prmLoc, "rb"))
        elif prm_dict is not None:
            o = copy.deepcopy(prm_dict)
        else:
            raise ValueError("Either prmPath or prm_dict has to be set")

        self.cols_order = o["cols_order"]
        self.prep_excluded = o["prep_excluded"]
        self.prep_included = o["prep_included"]
        self.encoded_features_order = o["encoded_features_order"]
        self.categorical_was_numeric = o["categorical_was_numeric"]
        self.num_as_cat = o["num_as_cat"]
        self.group_decision_excluded = o["group_decision_excluded"]
        self.scale = o["scale"]
        self.cat_clm = o["cat_clm"]
        self.num_clm = o["num_clm"]
        self.int_clm = o["int_clm"]
        self.scaler = o["scaler"]

    def __filter_features__(self, columns, prefix_sep="="):
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
            # Handling encoded columns. Encoded attribute name.
            cls.extend(self.df.filter(regex="^{}{}".format(c, prefix_sep)).columns.tolist())
            # Handling other columns while avoiding to mistook some. Attribute name only.
            cls.extend(self.df.filter(regex="^{}$".format(c, )).columns.tolist())
        return cls

    def __find_categorical__(self):
        """
        List int and float columns
        :return: The list of num column
        """
        cat_clm = []
        num_clm = []
        for c in self.df.select_dtypes(include=["int", 'float', 'double']).columns:
            if (self.df[c].value_counts().shape[0] <= self.num_as_cat) and (c not in self.group_decision_excluded):
                self.categorical_was_numeric.update({c: self.df[c].dtype})
                cat_clm.append(c)
            else:
                num_clm.append(c)
        cat_clm.extend(self.df.select_dtypes(exclude=["int", 'float', 'double']).columns.tolist())
        self.cat_clm = cat_clm
        self.num_clm = num_clm
        self.int_clm = self.df.select_dtypes(include=["int"]).columns.tolist()

    def get_indexes(self, to_drop, prefix_sep="="):
        """ compute the indexes of the categorical and numerical columns and return them """
        columns = self.df.columns.copy()
        to_ignore_ = self.__filter_features__(to_drop, prefix_sep=prefix_sep)
        columns = columns.drop(to_ignore_)

        def compute(clm_list, seperate=True):
            save = []
            for c in clm_list:
                if c not in to_drop:
                    cls = self.__filter_features__(columns=c, prefix_sep=prefix_sep)
                    indexes = []
                    for cl in cls:
                        if cl not in to_ignore_:
                            indexes.append(columns.get_loc(cl))
                    if seperate:
                        save.append(indexes)
                    else:
                        save.extend(indexes)
            return save

        return compute(self.cat_clm, seperate=True), compute(self.num_clm, seperate=False)

    def __from_dummies__(self, prefix_sep='='):
        """
        Convert encoded columns into original ones
        """
        data = self.df
        categories = self.cat_clm
        cat_was_num = self.categorical_was_numeric
        out = data.copy()
        for l in categories:
            cols = data.filter(regex="^{}{}".format(l, prefix_sep), axis=1).columns
            labs = [cols[i].split(prefix_sep)[-1] for i in range(cols.shape[0])]
            out[l] = pd.Categorical(np.array(labs)[np.argmax(data[cols].values, axis=1)])
            out.drop(cols, axis=1, inplace=True)
            if l in cat_was_num.keys():
                out[l] = out[l].astype(cat_was_num[l])

        self.df = out

    def __squash_in_range__(self):
        """
        Squash df values between min and max from scaler
        """
        for c in self.num_clm:
            i = self.df.columns.get_loc(c)
            self.df.loc[self.df[c] > self.scaler.data_max_[i], c] = self.scaler.data_max_[i]
            self.df.loc[self.df[c] < self.scaler.data_min_[i], c] = self.scaler.data_min_[i]
            # Change non values to modes
            self.df[c].replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df[c].fillna(self.df[c].mode().iloc[0], inplace=True)

    def set_encoded_features_order(self, features_order=None, from_df=False):
        if from_df:
            self.encoded_features_order = self.df.columns.tolist()
        else:
            self.encoded_features_order = features_order

    def set_features_params(self, features_order, categorical_was_numeric, num_clm, cat_clm):
        """
        Set the original feature ordering, and the original type of numeric columns that have been considered
        as categorical because of they had a limited number of values
        """
        self.encoded_features_order = features_order
        self.categorical_was_numeric = categorical_was_numeric
        self.num_clm = num_clm
        self.cat_clm = cat_clm

    def __encoded_features_format__(self):
        """
        Add missing columns to have the same dataset structure
        scale should be the lower scale as given i
        """
        prep_excluded = self.prep_excluded if self.prep_excluded is not None else []
        if self.encoded_features_order is not None:
            for c in self.encoded_features_order:
                if c not in self.df.columns and c not in prep_excluded:
                    self.df[c] = self.scale

            self.df = self.df[self.encoded_features_order]

    def __round_integers__(self):
        """
        Round the columns that where of type integer to begin with
        """
        for c in self.int_clm:
            try:
                self.df[c] = self.df[c].round().astype("int")
            except ValueError as e:
                print("Value error: {}\n Column: {}".format(str(e), c))
                if "Cannot convert non-finite values (NA or inf) to integer" not in str(e):
                    raise ValueError(e)
                self.df[c].replace([np.inf, -np.inf], np.nan, inplace=True)
                self.df[c].fillna(self.df[c].mode().iloc[0], inplace=True)

    def fit_transform(self, prefix_sep='='):
        """
        Apply all transformation. Some attribute will be set. If you do not want to change values of those attributes,
        call transform
        """
        if not self.transformed:
            indexes = self.df.index
            excluded = pd.DataFrame()
            # if self.prep_included is not None and len(self.prep_included) > 0:
            #     excluded = self.df[self.prep_included] # Included
            #     self.df = self.df.drop(self.prep_included, axis=1) # Excluded
            #     excluded, self.df = self.df, excluded # Inverting both
            # else:
            if self.prep_excluded is not None:
                excluded = self.df[self.prep_excluded]
                self.df.drop(self.prep_excluded, axis=1, inplace=True)
            self.__find_categorical__()
            self.df = pd.get_dummies(self.df, columns=self.cat_clm, prefix_sep=prefix_sep)

            # Scale the data, then add missing columns and set a fixed order, then add the excluded columns.
            # Excluded is at the end of the processing as we are not supposed to touch them.
            # columns
            # Extract columns order as they are encoded
            self.set_encoded_features_order(from_df=True)
            # Scaler contains all columns except the excluded ones
            # self.df.iloc[:, :] = self.scaler.fit_transform(self.df.values)
            self.df = pd.DataFrame(self.scaler.fit_transform(self.df.values), columns=self.df.columns)
            self.df = pd.concat([self.df, excluded], axis=1, sort=True)

            # Complete missing columns
            # self.__features_formatting__()
            # Scaler contains all columns
            # self.df.iloc[:, :] = self.scaler.fit_transform(self.df.values)
            # self.df = pd.DataFrame(self.scaler.fit_transform(self.df.values), columns=self.df.columns)
            # self.df = pd.concat([self.df, excluded], axis=1)
            self.df.index = indexes
            self.transformed = True

    def transform(self, prefix_sep='='):
        """
        Transform dataset without setting values.
        """
        if not self.transformed:
            indexes = self.df.index
            excluded = pd.DataFrame()
            if self.prep_excluded is not None:
                excluded = self.df[self.prep_excluded]
                self.df.drop(self.prep_excluded, axis=1, inplace=True)
            self.df = pd.get_dummies(self.df, columns=self.cat_clm, prefix_sep=prefix_sep)
            # Complete missing columns, and give a standard column order.
            self.__encoded_features_format__()
            self.df = pd.DataFrame(self.scaler.transform(self.df.values), columns=self.df.columns)
            self.df = pd.concat([self.df, excluded], axis=1, sort=True)
            self.df.index = indexes
            self.transformed = True

    def inverse_transform(self, ignore_scale=False):
        """
        Recover the original data
        """
        if self.transformed:
            indexes = self.df.index
            excluded = pd.DataFrame()
            # if self.prep_included is not None and len(self.prep_included) > 0:
            #     excluded = self.df[self.prep_included] # Included
            #     self.df = self.df.drop(self.prep_included, axis=1) # Excluded
            #     excluded, self.df = self.df, excluded # Inverting both
            # else:
            if self.prep_excluded is not None:
                excluded = self.df[self.prep_excluded]
                self.df.drop(self.prep_excluded, axis=1, inplace=True)
            self.__encoded_features_format__()
            if not ignore_scale:
                self.df = pd.DataFrame(self.scaler.inverse_transform(self.df.values), columns=self.df.columns)
            self.__from_dummies__()
            # Scaler contains all columns
            self.__squash_in_range__()
            self.__round_integers__()
            self.df = pd.concat([self.df, excluded], axis=1, sort=True)[self.cols_order]
            self.df.index = indexes
            self.transformed = False

    def copy(self, deepcopy=False):
        if deepcopy:
            return copy.deepcopy(self)
        return copy.copy(self)


class FairnessPreprocessing(Preprocessing):
    """
    Extension of preprocessing class to the fairness preprocessing
    """

    def __init__(self, csv, sens_attr_names, privileged_class_identifiers, decision_names, positive_decision_values,
                 numeric_as_categoric_max_thr=5, group_and_decision_excluded=[], scale=0, prep_excluded=None,
                 prep_included=None, binary_sensitive=True, binary_decision=True):
        super().__init__(csv, numeric_as_categoric_max_thr, group_and_decision_excluded, scale, prep_excluded,
                         prep_included)
        self.sens_attr_names = sens_attr_names
        self.privileged_class_values = privileged_class_identifiers
        self.decision_names = decision_names
        self.positive_decision_values = positive_decision_values

        # Privilege class will be mapped to 1, unprivileged one will be mapped to 0
        privileged_values = 1
        unprivileged_values = 0
        if binary_sensitive:
            self.__map__(self.sens_attr_names, privileged_class_identifiers, [privileged_values, unprivileged_values],
                         "int64")

        # positive decision will be mapped to 1, while negative will be 0
        positive_dec = 1
        negative_dec = 0
        if binary_decision:
            self.__map__(self.decision_names, positive_decision_values, [positive_dec, negative_dec], "int64")

    def __map__(self, list_of_attrs, list_of_values, list_of_values_to_map_to, series_type):
        """
        Map the attribute values in the dataset to the given ones.
        :param list_of_attrs: list of attributes to map
        :param list_of_values: list of values of each of these attribute to map in the dataset
        :param list_of_values_to_map_to: values to map the given list of values to
        :param series_type: new type of the mapped series
        """

        # find all instances which match any of the attribute values
        for attr, vals in zip(list_of_attrs, list_of_values):
            priv = np.logical_or.reduce(np.equal.outer(vals, self.df[attr].values))
            self.df.loc[priv, attr] = list_of_values_to_map_to[0]
            self.df.loc[~priv, attr] = list_of_values_to_map_to[1]
            self.df[attr] = self.df[attr].astype(series_type)


class TorchCommon(data.Dataset):

    def __init__(self):
        self.length = None

    @staticmethod
    def __filter_features___(df, columns, prefix_sep="="):
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
            cls.extend(df.filter(regex="^{}{}".format(c, prefix_sep)).columns.tolist())
            # Handling other columns while avoiding to mistook some. Attribute name only.
            cls.extend(df.filter(regex="^{}$".format(c, )).columns.tolist())
        return cls

    def __len__(self):
        return self.length


class TorchGeneralDataset(TorchCommon):

    def __init__(self, csv, target_features, xTensor=torch.FloatTensor, yTensor=torch.LongTensor, transform=None,
                 to_drop=None, noise=None, noise_fn=None, drop_target=True, prefix_sep="="):
        """
        Do all the heavy data processing here.
        :param csv: the filename with data or the data as dataFrame or an instance of Preprocessing class
        :param target_features: the target feature name
        :param xTensor: data tensor type
        :param yTensor: target tensor type
        :param transform: transformations from pytorch to apply
        :param to_drop: list of features to drop
        :param noise: Number of nodes to use for the noise
        :param noise_fn: Torch function to use to generate noise
        :param drop_target: Should we remove the targe value in x ?
        :param prefix_sep: prefix separator.
        """
        super().__init__()
        if isinstance(target_features, str):
            target_features = [target_features]
        if isinstance(csv, Preprocessing):
            df = csv.df
        elif isinstance(csv, pd.DataFrame):
            df = csv
        else:
            df = pd.read_csv(csv)
        self.length = len(df)
        drop = [target_features] if not isinstance(target_features, list) else target_features[:]
        if to_drop is not None:
            drop.extend(to_drop)
        self.noise = noise
        self.noise_fn = noise_fn
        if self.noise is not None and self.noise > 0:
            self.cat_noise = lambda x: torch.cat((x, self.noise_fn(1, self.noise).view(-1)), 0)
        else:
            self.cat_noise = lambda x: x

        self.cat_target = lambda x, y: x
        if not drop_target:
            self.cat_target = lambda x, y: torch.cat((x, y), 0)

        self.transform = transform
        self.x = xTensor(df.drop(self.__filter_features___(df, drop, prefix_sep=prefix_sep), axis=1).values)

        self.targets_dim = []
        self.y = []
        self.arg_max = lambda x: x.argmax(0)
        t = df[self.__filter_features___(df, target_features[0], prefix_sep=prefix_sep)].values
        if (len(t.shape) == 1) or (t.shape[1] == 1):
            self.arg_max = lambda x: x
        self.yf = xTensor(df[self.__filter_features___(df, target_features, prefix_sep=prefix_sep)].values)
        for target in target_features:
            targets = df[self.__filter_features___(df, target, prefix_sep=prefix_sep)].values
            self.targets_dim.append(targets.shape[1])
            self.y.append(yTensor(targets))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.x[index]
        yf = self.yf[index]
        y = {}
        yArg = {}
        for i, t in enumerate(self.y):
            y.update({"y{}".format(i): t[index]})
            yArg.update({"y{}Arg".format(i): self.arg_max(t[index])})
        xyn = self.cat_noise(self.cat_target(x, yf))
        # sample = {'x': x, "xyn": xyn, "yf": yf, "yArg": torch.cat((list(yArg.values()))),
        #           **y, **yArg, }
        sample = {'x': x, "xyn": xyn, "yf": yf, **y, **yArg, }
        return sample


class TorchNoTargetDataset(TorchCommon):

    def __init__(self, csv, xTensor=torch.FloatTensor, transform=None, to_drop=None, noise=None, noise_fn=None,
                 prefix_sep="="):
        """
        Do all the heavy data processing here.
        :param csv: the filename with data or the data as dataFrame or an instance of Preprocessing class
        :param xTensor: data tensor type
        :param yTensor: target tensor type
        :param transform: transformations from pytorch to apply
        :param to_drop: list of features to drop
        :param noise: Number of nodes to use for the noise
        :param noise_fn: Torch function to use to generate noise
        :param prefix_sep: prefix separator.
        """
        super().__init__()
        if isinstance(csv, Preprocessing):
            df = csv.df
        elif isinstance(csv, pd.DataFrame):
            df = csv
        else:
            df = pd.read_csv(csv)
        self.length = len(df)
        drop = []
        if to_drop is not None:
            drop.extend(to_drop)
        self.noise = noise
        self.noise_fn = noise_fn
        if self.noise is not None and self.noise > 0:
            self.cat_noise = lambda x: torch.cat((x, self.noise_fn(1, self.noise).view(-1)), 0)
        else:
            self.cat_noise = lambda x: x

        self.transform = transform
        self.x = xTensor(df.drop(self.__filter_features___(df, drop, prefix_sep=prefix_sep), axis=1).values)

    @staticmethod
    def __filter_features___(df, columns, prefix_sep="="):
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
            cls.extend(df.filter(regex="^{}{}".format(c, prefix_sep)).columns.tolist())
            # Handling other columns while avoiding to mistook some. Attribute name only.
            cls.extend(df.filter(regex="^{}$".format(c, )).columns.tolist())
        return cls

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.x[index]
        sample = {'x': x}
        return sample

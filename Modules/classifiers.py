import torch
import joblib
import numpy as np
from sklearn.utils import _testing as _t

from sklearn import multiclass as mc
from sklearn import multioutput as mo
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn import ensemble, neural_network, svm, gaussian_process, linear_model, tree


class PyToSkWrapper:
    """ Pytorch to sklearn wrapper. Just adding fit and predict functions. """

    def __init__(self, pytorch_model, already_fitted, fit_fn=None, argmax=True):
        """
        Initialisation
        :param pytorch_model: the model to wrap
        :param already_fitted: Tells if the model is already fitted. No need to train again.
        :param fit_fn: the function to fit the model. It should take as input the model plus the same
        arguments as any fit fn of sklearn
        :param argmax: the output of the predict fn will be << argmaxed >> before being returned
        """

        self.model = pytorch_model
        self.fitted = already_fitted
        self.fit_fn = fit_fn
        self.argmax = argmax

    def fit(self, *args, **kwargs):
        """ train the model or pass if already trained """
        if not self.fitted:
            self.fit_fn(self.model, *args, **kwargs)

    def predict(self, *args, **kwargs):
        out = self.model.predict(torch.FloatTensor(*args, **kwargs))
        if self.argmax:
            out = out.argmax(1)
        return out.data.numpy()


class ClassifiersBase:
    def __init__(self, verbose):
        self.listClfs = {}
        self.raise_nan = False
        self.v = verbose
        self.verbose = lambda s: print(s) if self.v else None
        self.fitted = False

    def add_classifiers(self, dict_of_clfs):
        """ Add the classifiers to the lists. Make sure there is no duplicate key to avoid erasal. """
        self.listClfs.update(dict_of_clfs)

    @_t.ignore_warnings(category=ConvergenceWarning)
    @_t.ignore_warnings(category=DataConversionWarning)
    def fit(self, *args, **kwargs):
        self.raise_nan = False
        try:
            for clf in self.listClfs.items():
                self.verbose("--- Training {}\n".format(clf[0]))
                clf[1].fit(*args, **kwargs)
        except ValueError as e:
            if not (("a minimum of 2 classes are required" in str(e)) or
                    ("samples of at least 2 classes" in str(e)) or
                    ("data contains only one class" in str(e))
            ):
                raise ValueError(e)
            self.raise_nan = True

    @_t.ignore_warnings(category=DataConversionWarning)
    def predict(self, *args, **kwargs):
        if self.raise_nan:
            return self.return_np_nan()
        p = {}
        for clf in self.listClfs.items():
            self.verbose("--- Predicting {}\n".format(clf[0]))
            p.update({clf[0]: clf[1].predict(*args, **kwargs)})

        return p

    def return_np_nan(self):
        p = {}
        for clf in self.listClfs.items():
            self.verbose("--- Setting NaN to {}\n".format(clf[0]))
            p.update({clf[0]: np.NaN})
        return p

    def dump(self, path):
        """ Dump classifiers on disk """
        joblib.dump(self.listClfs, path)

    def load(self, path):
        """ Load classifiers from path """
        try:
            self.listClfs = joblib.load(path)
            return True
        except FileNotFoundError:
            return False


class Classifiers(ClassifiersBase):
    """
    Class list of all classifier used to make prediction.
    Simple wrapper for all classifiers used.
    """

    def __init__(self, addition=None, addition_names=None, default=True, seed=42, verbose=False,
                 multiclass_wrapper=None, full_name=False):
        """
        :param addition: New classifiers to add in the list for making prediction
        :param addition_names: Names of the added classifiers
        :param default: Use the default list of classifiers. If set to False,
         addition must be provided
        :param multiclass_wrapper: sklearn multiclass wrapper. Can either be OneVsRestClassifier or OneVsOneClassifier,
        etc. If None, will not apply any multiclass wrapper.
        """
        super().__init__(verbose=verbose)
        assert (
            default or ((addition is not None) and (addition_names is not None)),
            "No classifiers provided. At least set "
            "default to True")
        if default:
            tc = [
                # ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                #                                     random_state=seed),
                neural_network.MLPClassifier(random_state=seed),
                # svm.SVC(class_weight="balanced", gamma="scale"),
                tree.DecisionTreeClassifier(),
                # ensemble.RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed),
                linear_model.LogisticRegression(class_weight='balanced', solver='liblinear')
                # autosklearn.classification.AutoSklearnClassifier(),
            ]
            if multiclass_wrapper is None:
                if full_name:
                    for c in tc:
                        self.listClfs.update({c.__class__.__name__: c})
                else:
                    for c in tc:
                        self.listClfs.update({"".join([char for char in c.__class__.__name__ if char.isupper()]): c})

            else:
                if full_name:
                    for c in tc:
                        self.listClfs.update({c.__class__.__name__: multiclass_wrapper(c)})
                else:
                    for c in tc:
                        self.listClfs.update(
                            {"".join([char for char in c.__class__.__name__ if char.isupper()]): multiclass_wrapper(c)})

        if (addition_names is not None) and (addition is None):
            self.listClfs.update(dict(zip(addition_names, addition)))

        self.verbose("List of Classifiers:")
        self.verbose(self.listClfs)


class ClassifiersValidation(ClassifiersBase):
    """
    Class list of all classifier used to make prediction.
    Simple wrapper for all classifiers used.
    """

    def __init__(self, addition=None, addition_names=None, default=True, seed=42, verbose=False,
                 multiclass_wrapper=None, full_name=False):
        """
        :param addition: New classifiers to add in the list for making prediction
        :param addition_names: Names of the added classifiers
        :param default: Use the default list of classifiers. If set to False,
         addition must be provided
        :param multiclass_wrapper: sklearn multiclass wrapper. Can either be OneVsRestClassifier or OneVsOneClassifier,
        etc. If None, will not apply any multiclass wrapper.
        """
        super().__init__(verbose=verbose)
        assert (
            default or ((addition is not None) and (addition_names is not None)),
            "No classifiers provided. At least set "
            "default to True")
        if default:
            tc = [
                ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                    random_state=seed),
                neural_network.MLPClassifier(random_state=seed),
                svm.SVC(class_weight="balanced", gamma="scale"),
                tree.DecisionTreeClassifier(),
                ensemble.RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed),
                # linear_model.LogisticRegression(class_weight='balanced', solver='liblinear')
                # autosklearn.classification.AutoSklearnClassifier(),
            ]
            if multiclass_wrapper is None:
                if full_name:
                    for c in tc:
                        self.listClfs.update({c.__class__.__name__: c})
                else:
                    for c in tc:
                        self.listClfs.update({"".join([char for char in c.__class__.__name__ if char.isupper()]): c})

            else:
                if full_name:
                    for c in tc:
                        self.listClfs.update({c.__class__.__name__: multiclass_wrapper(c)})
                else:
                    for c in tc:
                        self.listClfs.update(
                            {"".join([char for char in c.__class__.__name__ if char.isupper()]): multiclass_wrapper(c)})

        if (addition_names is not None) and (addition is None):
            self.listClfs.update(dict(zip(addition_names, addition)))

        self.verbose("List of Classifiers:")
        self.verbose(self.listClfs)


class OneVsRestClassifiers(Classifiers):
    """
    Class list of all classifier used to make prediction. Implementing the one-vs-rest strategy from sklearn
    Simple wrapper for all classifiers used.
    """

    def __init__(self, addition=None, addition_names=None, default=True, seed=42, verbose=False):
        """
        :param addition: New classifiers to add in the list for making prediction
        :param addition_names: Names of the added classifiers
        :param default: Use the default list of classifiers. If set to False,
         addition must be provided
        """
        super().__init__(addition, addition_names, default, seed, verbose, multiclass_wrapper=mc.OneVsRestClassifier)


class MultiOutputClassifiers(Classifiers):
    """
    Class list of all classifier used to make prediction. Implementing the one-vs-rest strategy from sklearn
    Simple wrapper for all classifiers used.
    """

    def __init__(self, addition=None, addition_names=None, default=True, seed=42, verbose=False):
        """
        :param addition: New classifiers to add in the list for making prediction
        :param addition_names: Names of the added classifiers
        :param default: Use the default list of classifiers. If set to False,
         addition must be provided
        """
        super().__init__(addition, addition_names, default, seed, verbose, multiclass_wrapper=mo.MultiOutputClassifier)


class Regressors(Classifiers):
    """
    Class list of all classifier used to make prediction.
    Simple wrapper for all classifiers used.
    """

    def __init__(self, addition=None, addition_names=None, default=True, seed=42, verbose=False, ):
        """
        :param addition: New classifiers to add in the list for making prediction
        :param addition_names: Names of the added classifiers
        :param default: Use the default list of classifiers. If set to False,
         addition must be provided
        :param multiclass_wrapper: sklearn multiclass wrapper. Can either be OneVsRestClassifier or OneVsOneClassifier,
        etc. If None, will not apply any multiclass wrapper.
        """
        super().__init__(addition, addition_names, default, seed, False, None)
        assert (
            default or ((addition is not None) and (addition_names is not None)),
            "No classifiers provided. At least set "
            "default to True")
        self.listClfs = {}
        if default:
            tc = [
                ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
                # ensemble.HistGradientBoostingRegressor(random_state=seed),
                neural_network.MLPRegressor(random_state=seed),
                tree.DecisionTreeRegressor(),
                ensemble.RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, criterion="mae"),
                linear_model.ElasticNet(random_state=seed)

            ]

        for c in tc:
            self.listClfs.update({c.__class__.__name__: c})
        if (addition_names is not None) and (addition is None):
            self.listClfs.update(dict(zip(addition_names, addition)))

        self.v = verbose
        self.verbose("List of Classifiers:")
        self.verbose(self.listClfs)
        self.fitted = False

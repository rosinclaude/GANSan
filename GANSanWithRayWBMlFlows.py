"""
    When experimenting, cross validation is used to evaluate the generalisation capability of the model.
    We train on k-1 data and test on the k-th one.
    Therefore when looking for hyperparameters, we cannot look for hyperparameters for each fold.
    But rather we should find the best hyperparameters, then we test the generalisation of the model built with such
    hyperparameters.

    The easiest way to do both is to split the dataset using k-val, the number of folds will define the ratio between
     the training, the validation and the test set.

     We can thus
      A- look for the best hyperparameters using the training set, and the validation set
     and evaluate its performances on the test set.
     Afterward, we perform the k-val.

     Or
     B- We can ignore the validation set and use the training and the test set. The number of splits
     used for the training set will thus include the validation set.

     Advantage of method A is that we can straightforwardly evaluate the performances using the test set to have a real
     idea of the model behaviour. Drawback is that we have less data for the training set, and since we will conduct the
     cross validation anyway, it might increase the computing time.

     B will require that we perform the cross validation to ensure that we have the best set of hyperparameters, but
     will have more data to train with and might be faster. So, we can not use the model built with such hyperparameters
     without cross validation.


     We use wandb, mlflow and Ray Tune.
     Ray Tune is used to tune the hyperparameters of the model.
     Mlflow and wandb are used to follow the training and final results of each training.
     The thing is, for Ray tune, we need to report all metrics we track, but often, the metric tracked is done at the
     end of the training (especially in MO optimisation), thus, metrics used during the training differ from what is
     computed and tracked during the validation set. Since train report expect the same metrics every time, we cannot
     track the model training progress and the hyperparameters tuning.
"""

import argparse
import copy
import os
import socket
import tempfile
import time
from functools import partial

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from torch.utils import data as torchd

import Datasets
from Modules import classifiers as clfs
from Modules import datasets as d
from Modules import models
from Modules import optimizers2 as o
from Modules import qualitativeMetrics as qm
from Modules import setup as s
from Modules import std_metric as std_m
from Modules import utils

from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import mlflow
import wandb

# Datasets.get_config()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=False, help="Dataset",
                    default='Lipton')
parser.add_argument("-y", "--decision", type=str, default="mapped", required=False,
                    choices=["original", "transformed", "generated"],
                    help="How should we process the decision attribute."
                         "orig: Untouched, removed from processing; transformed: Mapped from input to output,"
                         "gen: removed from input, but generated at output")
# parser.add_argument("--group_loss", type=str, default="Ber", choices=["Ber", "Acc"],
#                     help="Criterion for Fairness to use")
parser.add_argument("-s", "--redo_split", required=False, help="Re Do the dataset split", action="store_true")
parser.add_argument("-p", "--concurrent_trial", required=False, help="Number of concurrent trials", default=5, type=int)
parser.add_argument("-t", "--ntrials", required=False, help="Number of trials", default=100, type=int)
parser.add_argument("-r", "--experiment_root", required=False, help="Root path to save experiments",
                    default="/home/hun13r/Devel/Uqam/FairMapping/FinalTests/")
parser.add_argument("-b", "--ccrency_batches", required=False,
                    help="Finish one batch of concurrent trials before sampling the next", action="store_true")
parser.add_argument("--resume", required=False,
                    help="Resume running experiments if it exists", action='store_true')
parser.add_argument("--final_validation", required=False,
                    help="Compute final metrics and save in results folder", action="store_true")
parser.add_argument("--checkpointing",
                    help="Use model checkpointing to save GPU resources. See https://pytorch.org/docs/stable/checkpoint.html",
                    required=False,
                    action="store_true")
parser.add_argument("--ray_address", help="ray cluster address", default="auto")
parser.add_argument("--debug", help="are you debugging ? Do not start ray", action="store_true", required=False)
parser.add_argument("--model_save_rate", help="frequency of saving models during the training of epochs."
                                              "Note that models will only be available based on the given frequency "
                                              "(every x epoch). So checkpoint directory might not always exist.",
                    default=1,
                    type=int, required=False, )
parser.add_argument("--max_epochs", help="Maximum epochs to train Gansan with", default=2500, type=int,
                    required=False, )
# parser.add_argument("--max_failures", help="Try to recover a trial at least this many times. "
#                                            "Ray will recover from the latest checkpoint if present. "
#                                            "Setting to -1 will lead to infinite recovery retries. "
#                                            "Setting to 0 will disable retries. Defaults to 3.", default=3,
#                     type=int, required=False, )
parser.add_argument("--gpu", help="Use GPU if available", required=False, action="store_true")
parser.add_argument("--num_workers", required=False, help="Number of trials", default=0, type=int)
parser.add_argument("--suffix", required=False, help="Suffix to append to differentiate runs",
                    default='')
parser.add_argument("--mo", type=str, default="A-MO", required=False,
                    choices=["A-MO", "A-SC", "FP", "FPL"],
                    help="How should we optimise ? "
                         "A-MO - Use only heuristic A, with optuna, like multi-objective optimization"
                         "A-SC - Use only heuristic A, with a scalar optimisation and early stopping."
                         "FP - Use Fidelity and Protection"
                         "FPL - Use Fidelity, Protection and Losses ")

args, leftovers = parser.parse_known_args()

runtime_env = {
    "working_dir": "./",
    "excludes": [".ipynb_checkpoints", ".idea", "*.ipynb", ".git", "Core", "Storage"]
}
#
if not args.debug:
    try:
        ray.init(address=args.ray_address, runtime_env=runtime_env)
    except ConnectionError:
        ray.init(runtime_env=runtime_env)
else:
    args.num_workers = 0

RandomState = 42

# FOLD CONFIGURATION
KFolds = 5
# COMPUTATION DEVICES
DEVICE = torch.device("cpu")
RAY_RESSOURCES_PER_TRIALS = {"cpu": 1}  # Set to 1, each trial will be run on a single cpu

CUDA = False
if args.gpu and torch.cuda.is_available():
    # Percentage of gpu to use per trial
    gpu = 1 / args.concurrent_trial
    RAY_RESSOURCES_PER_TRIALS.update({"gpu": gpu})
    DEVICE = torch.device("cuda:0")  ##
    CUDA = True
CPU_DEVICE = torch.device("cpu")
# MODELS AND PLOTS

# -------------------------------------------------------------------------------------------------------------------- #
# STARTING
torch.manual_seed(RandomState)
np.random.seed(RandomState)
# experiment_suffix = "-".join([args.dataset, "y={}".format(args.decision), args.group_loss])
experiment_suffix = "-".join([args.dataset, "y={}".format(args.decision)])
ExperimentDir = "{}/{}" \
    .format(args.experiment_root, experiment_suffix)


def set_most_privileged_group(data_info):
    """ Take the first value of default group value and set it as the privileged group """
    priv_values = data_info["DefaultGroupValue"]
    most_priv = []
    for i, values in enumerate(data_info["DefaultGroupValue"]):
        most_priv.append([values[0]])
        data_info['ProtectedGroupValue'][i].extend(values[1:])
    data_info['DefaultGroupValue'] = most_priv
    return data_info


def set_group_variable_and_info(data_info, dataprep, most_prv_value_idx=0):
    """ Create an index of groups in the dataset, based on the different combination of sensitive values.
    Update data_info accordingly """

    # Groups ids
    group_by = dataprep.df.groupby(by=data_info['GroupAttributes']).ngroup()
    dataprep.df["groups"] = group_by
    # Get the mask for the most priv group
    priv_mask = np.logical_and.reduce(
        [dataprep.df[v] == data_info["DefaultGroupValue"][i][most_prv_value_idx] for i, v in
         enumerate(data_info['GroupAttributes'])])
    prv_indx = dataprep.df[priv_mask]["groups"].iloc[0]
    # Set most priv group to 1 and other values != 1
    dataprep.df["groups"] = dataprep.df["groups"] + 1 - prv_indx
    dataprep.df["groups"] = dataprep.df["groups"].astype("str")
    # Get the list of other groups ids
    groups_list = dataprep.df["groups"].drop_duplicates()
    # Get the mapping of group indexes
    group_mapping = {}
    for g in groups_list:
        map_ = dataprep.df.loc[dataprep.df["groups"] == g, data_info['GroupAttributes']].iloc[0].to_dict()
        group_mapping.update({"group {}".format(g): map_})

    # Update data info
    data_info.update({"OriginalGroupAttributes": data_info['GroupAttributes'],
                      "OriginalDefaultGroupValue": data_info["DefaultGroupValue"],
                      "OriginalProtectedGroupValue": data_info["ProtectedGroupValue"]})
    data_info['GroupAttributes'] = ["groups"]
    data_info["DefaultGroupValue"] = [[1]]
    data_info["ProtectedGroupValue"] = [groups_list[groups_list != 1].values.tolist()]
    data_info.update(dict(PropertiesSet=True))

    return group_mapping


def init(data_name="Lipton", approach_name="GANSan", re_split=False):
    """ Initialise folders and split dataset according to the given configuration """
    # dataset
    data_info = Datasets.get_config(data_name)

    exp_folders = s.ExperimentsFolders(base_dir=ExperimentDir, set_name=data_info["SetName"], data_key_name=data_name,
                                       approach_name=approach_name, prm_dir="/Prm/",
                                       prepared_data_dir="{}/Csv/".format(ExperimentDir), is_k_val=True,
                                       exp_number=1, root_base=ExperimentDir)
    # Check if the properties have already been set with a first call
    if not "PropertiesSet" in data_info.keys() or not data_info["PropertiesSet"]:
        # Read data and prepare first. Extract parameters
        # Transformations are done here.
        data = d.FairnessPreprocessing(csv="{}/{}.csv".format(data_info["Location"],
                                                              data_info["SetName"]),
                                       sens_attr_names=data_info['GroupAttributes'],
                                       privileged_class_identifiers=data_info['DefaultGroupValue'],
                                       decision_names=[data_info['LabelName']],
                                       positive_decision_values=[data_info['PosOutcomes']],
                                       numeric_as_categoric_max_thr=data_info['NumericAsCategoric'],
                                       scale=data_info['Scale'],
                                       prep_excluded=data_info['PreprocessingExcluded'],
                                       prep_included=data_info['PreprocessingIncluded'],
                                       binary_sensitive=True, binary_decision=True)
        # Set the new values if the binary bool have been set. Necessary for assigning the new data groups
        # Set the values as binaries
        data_info["DefaultGroupValue"] = [[1] for dfv in data_info["DefaultGroupValue"]]
        data_info["ProtectedGroupValue"] = [[0] for dfv in data_info["ProtectedGroupValue"]]

        data_info["PosOutcomes"] = [1 for po in data_info["PosOutcomes"]]

        # Set the most privileged group to which data will be mapped
        set_most_privileged_group(data_info)
        # Set the different groups of the dataset. The most privileged group will have index of '1'
        group_mapping = set_group_variable_and_info(data_info, data, most_prv_value_idx=0)
        # Re-set the data prep based on the new modifications
        if not exp_folders.k_validation_folders(nb_splits=KFolds) or re_split:
            data = d.Preprocessing(csv=data.df, numeric_as_categoric_max_thr=data_info['NumericAsCategoric'],
                                   scale=data_info['Scale'],
                                   prep_excluded=data_info['PreprocessingExcluded'],
                                   prep_included=data_info['PreprocessingIncluded'])
            data.fit_transform()
            data.save_parameters(prmPath=exp_folders.prm_dir)
            data.inverse_transform()
            data = data.df
            exp_folders.k_splits(data=data, nb_splits=KFolds, random_state=RandomState,
                                 stratify_feature=data_info["GroupAttributes"], similar_as=None)
        with open("{}/group_mapping.txt".format(exp_folders.exp_sets_dir), "w") as g_fp:
            g_fp.write(str(group_mapping))

    return data_info, exp_folders


# Classifiers use for some metric computation : mostly sensitive attribute BER and Accuracy.
classifiers = clfs.ClassifiersValidation(seed=RandomState)  # Output protection classifiers or for anything else


def number_rotation(current, max_, min_=1):
    """ Avoid getting past the maximum, return the minimum when above (>) given maximum value """
    c = current
    if current > max_:
        c = min_
    return c


def load_train_set(val_index, test_index, k_folds, prm_path, sets_dir, name_fn=None, transform=True,
                   **preprocess_kwargs):
    """ Load the training set based on the fold number. Ensure that columns are of the correct type """
    if name_fn is None:
        n = lambda x: "{}/{}.csv".format(sets_dir, x)
    else:
        n = lambda x: "{}/{}.csv".format(sets_dir, name_fn(x))
    types = pd.read_csv(n(1))
    train_d = pd.DataFrame(columns=types.columns)
    types = types.dtypes
    types[types == "object"] = "str"
    # types = dict(types)
    for i in range(1, k_folds + 1):
        if i != val_index and i != test_index:
            tmp = pd.read_csv(n(i))
            types.loc[tmp.dtypes[types.index] != types] = "str"
            t = dict(types)
            # train_d = pd.concat([train_d.astype(types), pd.read_csv(name_fn(i)).astype(types)], axis=0)
            train_d = pd.concat([train_d.astype(t), tmp.astype(t)], axis=0)
    train_d.reset_index(drop=True, inplace=True)
    train_d = d.Preprocessing(csv=train_d, **preprocess_kwargs)
    train_d.load_parameters(prmPath=prm_path)
    if transform:
        train_d.transform()
    return train_d


def load_index(csv_index, prm_path, sets_dir, name_fn=None, transform=True, **preprocess_kwargs):
    """ Load dataset split (cross-validation) corresponding to the given index """
    if name_fn is None:
        n = lambda x: "{}/{}.csv".format(sets_dir, x)
    else:
        n = lambda x: "{}/{}.csv".format(sets_dir, name_fn(x))
    data = d.Preprocessing(csv="{}".format(n(csv_index)), **preprocess_kwargs)
    data.load_parameters(prmPath=prm_path)
    if transform:
        data.transform()
    return data


def get_datasets(val_index, test_index, exp_folders, data_info, k_folds=KFolds, transform=True,
                 ignore_validation=False, ignore_test=False):
    """ Get the training, validation and test sets
     :param transform : make the dataset model ready
     :param ignore_validation : Do not load the validation set, but only the training and test set
     :param ignore_test : Do not load the test set, but only the training and validation set
     :param val_index : Index of the validation set
     :param test_index : Index of the test set
     :param exp_folders: Path to the folder containing the experiments
     :param data_info: Dictionary of information about the dataset
     :param k_folds: Number of folds for cross validation"""

    valid_prep = None
    if not ignore_validation:
        valid_prep = load_index(val_index, exp_folders.prm_dir, exp_folders.exp_sets_dir, name_fn=None,
                                transform=transform, numeric_as_categoric_max_thr=data_info['NumericAsCategoric'],
                                scale=data_info['Scale'],
                                prep_excluded=data_info['PreprocessingExcluded'],
                                prep_included=data_info['PreprocessingIncluded'])

    train_prep = load_train_set(val_index * (not ignore_validation), test_index * (not ignore_validation), k_folds,
                                exp_folders.prm_dir, exp_folders.exp_sets_dir, name_fn=None, transform=transform,
                                numeric_as_categoric_max_thr=data_info['NumericAsCategoric'], scale=data_info['Scale'],
                                prep_excluded=data_info['PreprocessingExcluded'],
                                prep_included=data_info['PreprocessingIncluded'])
    test_prep = None
    if not ignore_test:
        test_prep = load_index(test_index, exp_folders.prm_dir, exp_folders.exp_sets_dir, name_fn=None,
                               transform=transform, numeric_as_categoric_max_thr=data_info['NumericAsCategoric'],
                               scale=data_info['Scale'],
                               prep_excluded=data_info['PreprocessingExcluded'],
                               prep_included=data_info['PreprocessingIncluded'])

    return train_prep, valid_prep, test_prep


class AccuracyLoss(nn.Module):
    """
    Compute the accuracy loss, in a compatible way as nn.Module (Mostly for CUDA)
    """

    def __init__(self, device="cpu"):
        super(AccuracyLoss, self).__init__()

    def forward(self, input_, target, loss_target_value=1, *args, **kwargs):
        """
        Compute the accuracy
        """
        # Selecting the outputs that should be maximized (equal to 1)
        if len(target.size()) > 1:
            target = target.argmax(1)
        out = input_[range(len(target)), target]
        # computing accuracy loss (distance to 1
        return torch.abs(loss_target_value - out.mean())


class BalancedErrorRateLoss(nn.Module):
    """
    Compute the balanced error rate loss.
    """

    def __init__(self, device="cpu"):
        """
        :param targetBer: the value of the BER to be closed to
        """
        super(BalancedErrorRateLoss, self).__init__()
        self.device = device

    def get_true_value(self, computed_ber):
        """
        Return the true value of the computed BER, not the distance from target. BER is between [0,1/2]. Impossible to
        go beyond that interval unless wrongly implemented.
        :param computed_ber: the distance from the target BER. Must be of type numpy array.
        """
        if isinstance(computed_ber, list):
            computed_ber = np.array(computed_ber)
        b = -computed_ber + self.targetBer
        return np.abs(b)

    def forward(self, input_, target, group, loss_target_value=1 / 2, *args, **kwargs):
        """
        Comput the balanced error rate
        :param input_: the input
        :param target: the target values
        :param group: the attribute whose values allow group constructions
        :param args: the argument to ignore
        :param kwargs: the keyword arguments to ignore
        :return: the computed loss
        """
        if len(target.size()) > 1:
            target = target.argmax(1)
        if len(group.size()) > 1:
            group = group.argmax(1)

        # Selecting the right predictions,
        out = input_[range(len(target)), target]
        # Computing errors
        out = torch.abs(1 - out)
        # Reshaping data
        group.to(self.device, non_blocking=True)
        group = group.view(-1)
        out = out.view(-1, 1)
        l = len(out)
        # Summing by mask values
        # From: https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
        m = torch.zeros(group.max() + 1, l).to(self.device, non_blocking=True)
        m[group, torch.arange(l)] = 1
        m = nn.functional.normalize(m, p=1, dim=1)
        # Computing the mean of each group\
        out = torch.mm(m, out)
        out = out.mean()
        # k = torch.abs(input.argmax(1) - target).type(torch.FloatTensor).to(self.device)
        # k = (k[sens==1].mean() + k[sens==0].mean())/2
        return torch.abs(loss_target_value - out)

    def __str__(self):
        return "{}(\n Device: {}\n)".format(self.__class__.__name__, self.device)


class MILoss(nn.Module):
    """ Compute an estimate of the Mutual Information """

    def __init__(self, device="cpu", mi_safe_value=1e-20):
        """
        :param targetBer: the value of the BER to be closed to
        """
        super(MILoss, self).__init__()
        self.device = device
        self.mi_safe_value = mi_safe_value

    def non_zero_vector(self, vector):
        """ Ensures that the vector has no value of zero. As we are dealing with probabilities a value of zero could be
        to some extend equal to an extremely low value """

        for i in range(vector.size(0)):
            if vector[i] == 0:
                vector[i] += self.mi_safe_value
        return vector

    def forward(self, input_, target, *args, **kwargs):
        """
            Comput the balanced error rate in a way that can be used during the model training.
            :param input_: the input
            :param target: the target values
            :param args: the argument to ignore
            :param kwargs: the keyword arguments to ignore
            :return: the computed loss
        """
        return_as_np = False
        if not isinstance(input_, torch.Tensor):
            input_ = torch.Tensor(input_)
            return_as_np = True
        if not isinstance(target, torch.Tensor):
            target = torch.Tensor(target)
            return_as_np = True

        p_s = self.non_zero_vector(target.mean(0))
        p_s_hat = self.non_zero_vector(input_.mean(0))

        length = target.size(0)
        target_argmax = target.argmax(1)
        target_groups = target_argmax.unique()
        mi = 0
        # We only consider groups that are presents in the target data, which represent the real data.
        # Input_ contains the distribution of proba of each columns, therefore, all columns have a values.
        # Moreover, as we consider input_ as probabilities instead of real group indicator, we cannot use it as
        # reference of possible groups that are present in the given batch of data. Groups not presents in target will
        # be ignored, as the joint distribution does not exists, thus mi for those groups will be 0.
        for s in target_groups:
            # Fix s
            s_mask = target_argmax == s
            for s_hat in target_groups:
                # Fix the predicted s
                p_s_s_hat = input_[s_mask][:, s_hat].sum() / length
                if p_s_s_hat == 0:
                    # Add an infinitely small value to avoid nan value with the log
                    # Needed, since some columns of in input_ might have nil probability
                    p_s_s_hat += self.mi_safe_value
                mi += p_s_s_hat * torch.log(p_s_s_hat / (p_s[s] * p_s_hat[s_hat]))
        if return_as_np:
            return mi.item()
        return mi

    def numpy_mi(self, input_, target):
        """ Compute the MI when input_ and targets are vector, not matrices """
        target_groups = np.unique(target)
        input_groups = np.unique(input_)
        common_groups = np.intersect1d(input_groups, target_groups)
        length = target.shape[0]
        p_s = {}
        p_s_hat = {}
        # For columns that are not among the possible values of input or target, the mi will always be 0, as there are
        # no joint distribution. We only proceed with groups that are both presents in input_ and target
        mi = 0
        for s in common_groups:
            p_s.update({s: (target == s).sum() / length})
            p_s_hat.update({s: (input_ == s).sum() / length})
        for s in common_groups:
            mask_target_s = target == s
            for s_hat in common_groups:
                mask_input_s_hat = input_ == s_hat
                intersection = np.logical_and.reduce([mask_target_s, mask_input_s_hat])
                p_intersection = intersection.sum() / length
                if p_intersection == 0:
                    # Add an infinitely small value to avoid nan value with the log
                    # No need, since we ensure that we only work with values that have an intersection
                    p_intersection += self.mi_safe_value
                mi += p_intersection * np.log(p_intersection / (p_s[s] * p_s_hat[s_hat]))

        return mi

    def __call__(self, input_, target, *args, **kwargs):
        if isinstance(input_, torch.Tensor):
            return self.forward(input_, target, *args, **kwargs)
        else:
            return self.numpy_mi(input_, target)


class Logs:
    """
        Class to handle wandb and mlflow logging procedure, with necessary initialization, so that we do not have
        to pass objects everywhere.
    """
    def __init__(self, wandb_init, mlflow_init):
        self.wandb = setup_wandb(**wandb_init)
        if not args.debug:
            self.mlflow = setup_mlflow(**mlflow_init)

    def log_metrics(self, metrics_dictionary, step=None, save_directory=None, **to_save):
        """
        Log the given metrics to wandb, mlflow and tune.
        For the beginning only use ray tune. But if there is an issue with ray tune calling report multiple time,
        uncomment mlflow and w&b
        Args:
            metrics_dictionary: Dictionary containing the set of metrics to log
            step: current step
            save_directory : location where to save models and data.

        Returns:

        """
        if not args.debug:
            for m_name, m_value in metrics_dictionary.items():
                mlflow.log_metric(m_name, m_value, step=step)

        if step is not None:
            metrics_dictionary.update({"step": step})

        # In debug mode, tune is not running. Thus, log to wandb and mlflow only.
        self.wandb.log(metrics_dictionary)
        if not args.debug:
            if to_save:
                if save_directory is not None:
                    save_models(save_directory, epoch=step, **to_save)
                else:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        save_models(temp_dir, epoch=step, **to_save)
                        train.report(metrics_dictionary, checkpoint=train.Checkpoint.from_directory(temp_dir))
            else:
                train.report(metrics_dictionary)

    def finish(self):
        """
        Call wandb finish to end.
        Returns:

        """
        self.wandb.finish()

def get_target_mask(dataframe, data_info):
    """
    Extract the boolean mask representing the data belonging only to the privileged group
    Args:
        dataframe: dataset from which to extract the target mask
        data_info: Informations about the dataset, namely sensitive columns and values

    Returns:

    """
    mask = []
    for i, n in enumerate(data_info["GroupAttributes"]):
        # We should only have a single value in data_info["DefaultGroupValue"][i], as we take the most priv
        # for v in data_info["DefaultGroupValue"][i]:
        # s = "{}={}".format(n, v)
        s = "{}={}".format(n, data_info["DefaultGroupValue"][i][0])
        mask.append(np.logical_or.reduce([dataframe[s] == 1, dataframe[s] == '1']))
    return np.logical_and.reduce(mask)


def duplicate_rows(row, n):
    """
    Duplicate given rows
    Args:
        row: row to duplicate
        n: number of repeat

    Returns:

    """
    return np.repeat(row, n, axis=0)


def to_np(x):
    """
    Convert given tensor x into numpy array
    Args:
        x:

    Returns:

    """
    return x.data.cpu().numpy()


def batch_target_mask(batch, target):
    """
    For the given batch, extract a mask identifying rows belonging only to the target group
    Args:
        batch: data batch
        target: value representing the target group

    Returns:

    """
    return (batch == torch.FloatTensor(target).to(batch.device, non_blocking=True)).all(1)


def make_targets(prv_data, enc_drop_cols, enc_group_cols, prt_data=None):
    """ create x and y for models.
        Drop necessary columns to form x.
         suppose that enc_drop_cols already contains enc_group_cols
         If prt_data is supplied, split both prv and prt into respective x and y, then concatenate into a single x and y
         respectively """

    if prt_data is None:
        return prv_data.drop(enc_drop_cols, axis=1).values, prv_data[enc_group_cols].values.argmax(1)
    x = []
    y = []
    for data in [prv_data, prt_data]:
        x.append(data.drop(enc_drop_cols, axis=1).values)
        y.append(data[enc_group_cols].values.argmax(1))
    return np.concatenate(x), np.concatenate(y)


class MetricComputation:
    """
    Class to compute metrics that will be used for both validation and test.
    """
    def __init__(self, exp_folders, eval_set_index, noise_dim, train_prep, eval_set_prep,
                 train_loader_single_batch, eval_set_loader, train_target_mask, eval_set_target_mask,
                 enc_dropped_cols, enc_group_names, to_drop, dist_module, div_module, dam_mod,
                 fair_mod, mi_criterion, is_testing_set):
        """

        Args:
            exp_folders: Experiment folder object
            eval_set_index: Index corresponding to the evaluation set
            noise_dim: dimension of the noise to consider
            train_prep: training dataset preparation module
            eval_set_prep: testing dataset preparation module
            train_loader_single_batch: train set as a single batch
            eval_set_loader: evaluation set loader
            train_target_mask: target mask for the training set
            eval_set_target_mask: target mask for the test set
            enc_dropped_cols: columns to drop in encoded format. Contains enc_group_names
            enc_group_names: sensitive group names in encoded format
            to_drop: additional columns to drop
            dist_module: module to compute distance metrics
            div_module: module to compute diversity metrics
            dam_mod: module to compute damage in the dataset
            fair_mod: module to compute the fairness metrics
            mi_criterion: criterion to compute the mutual information
            is_testing_set: Compute all metrics, including fairness, only for when we are dealing with the test set. Do
            not use tune to report metrics, but rather return them.
        """

        self.exp_folders = exp_folders
        self.set_index = eval_set_index
        self.noise_dim = noise_dim
        self.train_prep = train_prep
        self.train_loader_single_batch = train_loader_single_batch
        self.train_target_mask = train_target_mask
        self.eval_set_prep = eval_set_prep
        self.eval_set_loader = eval_set_loader
        self.eval_set_target_mask = eval_set_target_mask
        self.enc_dropped_cols = enc_dropped_cols
        self.enc_group_names = enc_group_names
        self.to_drop = to_drop
        self.dist_module = dist_module
        self.div_module = div_module
        self.dam_mod = dam_mod
        self.fair_mod = fair_mod
        self.mi_criterion = mi_criterion
        self.is_testing_set = is_testing_set

    def change_evaluation_set(self, new_set_prep, new_set_loader, new_set_target_mask):
        """
        Change the dataset on which to compute the evaluation
        Args:
            new_set_prep: new dataset prepared
            new_set_loader: new dataset loader
            new_set_target_mask: new dataset target mask.

        Returns:

        """
        self.eval_set_prep = new_set_prep
        self.eval_set_loader = new_set_loader
        self.eval_set_target_mask = new_set_target_mask

    @staticmethod
    def format_transformed_data(transformed_data, data_prep, enc_dropped_cols, transform=True):
        """
        Reshape the transformed data and transformed it back in the similar format as the original data.
        data_prep.df contains the original data
        Args:
            transformed_data: transformed dataset to shape back as the original one
            data_prep: data preparation object to hold the transformed data and shape it back as the original one
            enc_dropped_cols: columns that have been dropped from the transformed data (most probably during the training)
            transform: Should the data be set into shape for ml input ?
        Returns:
            Nothing, all processing will be carried out in the respective objects.
        """

        df = pd.DataFrame(to_np(transformed_data), columns=data_prep.df.columns.drop(enc_dropped_cols))
        df = pd.concat([df, data_prep.df[enc_dropped_cols]], axis=1, sort=True)
        data_prep.df = df
        data_prep.inverse_transform()
        if transform:
            data_prep.transform()

    def compute_diversities(self, data_prep, data_target_mask, enc_dropped_cols):
        """
        Compute the diversity values
        Args:
            data_prep: Dataset encapsulated in its preparation module (prepared data)
            data_target_mask: Mask identifying the data belonging only to the privileged group
            enc_dropped_cols: columns dropped from the experiment and training

        Returns:

        """
        data_div = self.div_module(data_prep.df.drop(enc_dropped_cols, axis=1))["diversity"][0]
        privileged_grp_div = self.div_module(data_prep.df[data_target_mask].drop(enc_dropped_cols,
                                                                                 axis=1))["diversity"][0]
        protected_group_div = self.div_module(data_prep.df[~data_target_mask].drop(enc_dropped_cols,
                                                                                   axis=1))["diversity"][0]

        return data_div, privileged_grp_div, protected_group_div

    def transform_data(self, g_map, data_loader, data_prep, transform=True):
        """
        Transform the dataset using the trained model.
        Args:
            g_map: model use for transformation
            data_loader: loader of the dataset
            data_prep: preparation module for the dataset
            transform: Should the data be set into shape for ml input ?
        Returns:
            Nothing, all processing will be carried out in the respective objects.
        """
        g_map.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                # 4 - Update
                x = batch["x"].to(g_map.device, non_blocking=True)
                noise = get_noise(self.noise_dim, x.size(0), device=g_map.device)
                x_trans = g_map(x, noise=noise).to(CPU_DEVICE)

        # Format the data back into the original shape
        self.format_transformed_data(x_trans, data_prep, self.enc_dropped_cols, transform=transform)

    def compute_test_metrics_ext_val(self, g_map, invert_is_testing_set=False, ignore=False):
        """
        Compute the necessary metrics using the external validation, meaning external classifiers.
        Args:
            g_map: model that performs transformation
            invert_is_testing_set: Invert the value of the boolean is testing set.
            ignore: Ignore the test metric computation

        Returns:

        """

        reported_metrics = {}
        if ignore:
            return reported_metrics


        if invert_is_testing_set:
            self.is_testing_set = not self.is_testing_set
        # Do all test computation on the cpu
        g_map.to(CPU_DEVICE)

        tr_p_trans = copy.deepcopy(self.train_prep)
        self.transform_data(g_map=g_map, data_loader=self.train_loader_single_batch, data_prep=tr_p_trans,
                            transform=True)
        ts_p_trans = copy.deepcopy(self.eval_set_prep)
        self.transform_data(g_map=g_map, data_loader=self.eval_set_loader, data_prep=ts_p_trans,
                            transform=True)
        g_map.to(DEVICE)
        # ----------------------------------Protection------------------------------------ #
        for tr_prep, ts_prep, type_ in zip([tr_p_trans, self.train_prep], [ts_p_trans, self.eval_set_prep], ["R-M", "O-O"]):
            if type_ == "O-O":
                x_train, y_train = make_targets(tr_prep.df[self.train_target_mask], self.enc_dropped_cols,
                                                self.enc_group_names,
                                                prt_data=self.train_prep.df[~self.train_target_mask])
                x_test, y_test = make_targets(ts_prep.df[self.eval_set_target_mask], self.enc_dropped_cols,
                                              self.enc_group_names,
                                              prt_data=self.eval_set_prep.df[~self.eval_set_target_mask])
            else:
                x_train, y_train = make_targets(tr_prep.df[self.train_target_mask], self.enc_dropped_cols,
                                                self.enc_group_names,
                                                prt_data=tr_p_trans.df[~self.train_target_mask])
                x_test, y_test = make_targets(ts_prep.df[self.eval_set_target_mask], self.enc_dropped_cols,
                                              self.enc_group_names,
                                              prt_data=ts_p_trans.df[~self.eval_set_target_mask])

            # Train external classifiers
            classifiers.fit(x_train, y_train)
            test_pred = classifiers.predict(x_test)
            acc_max = 0
            ber_min = 1
            highest_mi = 0
            # print("-----------------------------------------")
            for clf, p in test_pred.items():
                acc = ((p == y_test) * 1).mean()
                ber = pd.DataFrame.from_dict({"err": p != y_test, "group": y_test}).groupby(by="group").mean()
                mi = self.mi_criterion(input_=p, target=y_test)
                acc_max = max(acc_max, acc)
                ber_min = min(ber_min, ber.mean().item())
                highest_mi = max(highest_mi, mi)

            reported_metrics.update({"protection: {}".format(type_): ber_min})
            reported_metrics.update({"Accuracy-S: {}".format(type_): acc_max})
            reported_metrics.update({"MI: {}".format(type_): highest_mi})

            # ---------------------------------------------------------------------- #
            # Fairness
            if self.is_testing_set:
                self.fair_mod.reset_results()

                train_orig_mapped = (self.train_prep.df[self.train_target_mask], tr_p_trans.df[~self.train_target_mask])
                train_orig_mapped = pd.concat(train_orig_mapped, axis=0).loc[self.train_prep.df.index]

                test_orig_mapped = (self.eval_set_prep.df[self.eval_set_target_mask], ts_p_trans.df[~self.eval_set_target_mask])
                test_orig_mapped = pd.concat(test_orig_mapped, axis=0).loc[self.eval_set_prep.df.index]

                self.fair_mod.compute(original_train=self.train_prep.df, rec_mapped_train=tr_p_trans.df,
                                      orig_mapped_train=train_orig_mapped, original_test=self.eval_set_prep.df,
                                      rec_mapped_test=ts_p_trans.df, orig_mapped_test=test_orig_mapped,
                                      exp_folders=self.exp_folders, mapped_decision=args.decision != "original")

        # ----------------------------------Utility------------------------------------ #
        # --> Distance
        # Distance entre données protégées originales et obtenues par changement de code
        prt_dist_mae = self.dist_module.mae(self.eval_set_prep.df[~self.eval_set_target_mask].drop(self.enc_dropped_cols,
                                                                                                   axis=1).values,
                                            ts_p_trans.df[~self.eval_set_target_mask].drop(self.enc_dropped_cols,
                                                                                           axis=1).values)
        prv_dist_mae = self.dist_module.mae(self.eval_set_prep.df[self.eval_set_target_mask].drop(self.enc_dropped_cols,
                                                                                                  axis=1).values,
                                            ts_p_trans.df[self.eval_set_target_mask].drop(self.enc_dropped_cols,
                                                                                          axis=1).values)
        dist_mae = self.dist_module.mae(self.eval_set_prep.df.drop(self.enc_dropped_cols, axis=1).values,
                                        ts_p_trans.df.drop(self.enc_dropped_cols, axis=1).values)
        data_dist_mse = self.dist_module.distance_mse(self.eval_set_prep.df.drop(self.enc_dropped_cols,
                                                                                 axis=1).values,
                                                      ts_p_trans.df.drop(self.enc_dropped_cols, axis=1).values)
        prt_dist_mse = self.dist_module.distance_mse(self.eval_set_prep.df[~self.eval_set_target_mask].drop(
            self.enc_dropped_cols, axis=1).values,
                                                     ts_p_trans.df[~self.eval_set_target_mask].drop(self.enc_dropped_cols,
                                                                                                    axis=1).values)
        prv_dist_mse = self.dist_module.distance_mse(self.eval_set_prep.df[self.eval_set_target_mask].drop(
            self.enc_dropped_cols, axis=1).values,
                                                     ts_p_trans.df[self.eval_set_target_mask].drop(self.enc_dropped_cols,
                                                                                                   axis=1).values)

        if not self.is_testing_set:
            reported_metrics.update({"fidelity: prt": 1 - prt_dist_mae, "fidelity: prv": 1 - prv_dist_mae,
                                     "fidelity: all": 1 - dist_mae, "fidelity-mse: prt": 1 - prt_dist_mse,
                                     "fidelity-mse: prv": 1 - prv_dist_mse, "fidelity-mse: all": 1 - data_dist_mse,
                                     "heuristic A": dist_mae + (1/2 - reported_metrics["protection: R-M"])**2})
        else:
            # --> Transformed
            data_div, prv_div, prt_div = (
                self.compute_diversities(ts_p_trans, self.eval_set_target_mask, self.enc_dropped_cols))

            # --> Original
            data_o_div, prv_o_div, prt_o_div = (
                self.compute_diversities(self.eval_set_prep, self.eval_set_target_mask, self.enc_dropped_cols))

            reported_metrics.update({"fidelity: prt": 1 - prt_dist_mae, "fidelity: prv": 1 - prv_dist_mae,
                                     "fidelity: all": 1 - dist_mae, "fidelity-mse: prt": 1 - prt_dist_mse,
                                     "fidelity-mse: prv": 1 - prv_dist_mse, "fidelity-mse: all": 1 - data_dist_mse,
                                     "heuristic A": dist_mae + (1 / 2 - reported_metrics["protection: R-M"]) ** 2,
                                     "diversity_prt": prt_div, "diversity_prt_org": prt_o_div,
                                     "diversity_prv": prv_div, "diversity_prv_org": prv_o_div,
                                     "diversity_rc_prv": data_div, "diversity_org": data_o_div
                                     })

            self.eval_set_prep.inverse_transform()
            ts_p_trans.inverse_transform()
            prt_cat_damage, prt_num_damage = (
                self.dam_mod(self.eval_set_prep.df[~self.eval_set_target_mask].drop(self.to_drop, axis=1),
                             ts_p_trans.df[~self.eval_set_target_mask].drop(self.to_drop, axis=1)))
            prv_cat_damage, prv_num_damage = (
                self.dam_mod(self.eval_set_prep.df[self.eval_set_target_mask].drop(self.to_drop, axis=1),
                             ts_p_trans.df[self.eval_set_target_mask].drop(self.to_drop, axis=1)))

            # Dataframes and save on disks
            reported_metrics_df = pd.DataFrame(reported_metrics, index=[0])
            reported_metrics_df["Set Index"] = self.set_index
            reported_metrics_df.to_csv(
                "{}/ProtectionClassificationFidelityDiversity.csv".format(self.exp_folders.results_dir), index=False)

            fairness_results_df = pd.DataFrame.from_dict(self.fair_mod.results)
            fairness_results_df["Set Index"] = self.set_index
            fairness_results_df.to_csv("{}/TaskAndFairness.csv".format(self.exp_folders.results_dir), index=False)

            prt_utility = pd.DataFrame.from_dict(prt_cat_damage)
            prt_utility["Set Index"] = self.set_index
            prt_utility["Group"] = "Protected"
            ###
            prv_utility = pd.DataFrame.from_dict(prv_cat_damage)
            prv_utility["Set Index"] = self.set_index
            prv_utility["Group"] = "Privileged"
            ###
            damage_utility = pd.concat([prt_utility, prv_utility], axis=0, sort=True)
            damage_utility.to_csv("{}/CatDamage.csv".format(self.exp_folders.results_dir), index=False)

        # Reset is_testing_set to previous value
        if invert_is_testing_set:
            self.is_testing_set = not self.is_testing_set

        return reported_metrics


def save_models(checkpoint_dir, epoch, name, losses, **models_to_save):
    """
    Model checkpointing for saving computing resources and ease of restart in case of problem
    Args:
        checkpoint_dir: Where to save data
        epoch: current running epoch
        name: file name
        losses: metrics to save along with the models
        **models_to_save: Models to checkpoint

    Returns:

    """
    # Model checkpointing
    # Then create a checkpoint file in this directory.
    path = os.path.join(checkpoint_dir, "{}.pt".format(name))
    # Save state to checkpoint file.
    to_save = {}
    for name, model in models_to_save.items():
        to_save.update({name: model.state_dict()})
    to_save.update(dict(epoch=epoch))
    torch.save(to_save, path)
    l_path = os.path.join(checkpoint_dir, "{}-losses.csv".format(name))
    pd.DataFrame.from_dict(losses).to_csv(l_path, index=False)


def load_models(checkpoint_dir, name, **models_to_load):
    """
    Load models from checkpoint directory
    Args:
        checkpoint_dir: location of checkpoint
        name: checkpoint name / model name
        **models_to_load: Models to load from checkpoint

    Returns:

    """
    # Load models 
    path = os.path.join(checkpoint_dir, "{}.pt".format(name))
    if not os.path.exists(path):
        return 0, dict()  # {"epoch": [], "losses": [], "type": []}
    # Load state from checkpoint file.
    checkpoint = torch.load(path)
    for name, model in models_to_load.items():
        model.load_state_dict(checkpoint[name])
    l_path = os.path.join(checkpoint_dir, "{}-losses.csv".format(name))
    losses = pd.read_csv(l_path)
    return checkpoint["epoch"], losses


def get_noise(dim, num_samples, device):
    """
    return a matrix of random values of length corresponding to the given number of sample
    Args:
        dim: dimension of the matrix (number of columns)
        num_samples: Length of matrix to generate
        device: computing device (cpu, gpu)

    Returns:

    """
    if dim > 0 and num_samples > 0:
        return torch.rand((num_samples, dim), device=device)


def train_gw(gw, discriminator, gw_optim, gw_scheduler, d_optim, groups_criterion,
             rec_criterion, train_loader, training_epochs, n_critic, errors_dict, noise_dim=1, rec_coef=0, grp_g_coef=1,
             restart_training=False, bar_info="", checkpoint_dir=None, nb_groups=2, logger=None,
             metrics_computation=None):
    """
    Training of the generator and the discriminator
    Args:
        noise_dim: number of noise nodes
        gw: generator model
        discriminator: discriminator model
        gw_optim: Optimiser of the generator model
        gw_scheduler: Scheduler for the optimiser of the generator model
        d_optim: Optimiser of the discriminator model
        groups_criterion: Criterion for the groups prediction
        rec_criterion: Criterion for the data reconstruction
        train_loader: Loader containing the training dataset
        training_epochs: Number of epochs for the training of both the generator and the discriminator
        n_critic: ratio with which the generator model is trained
        errors_dict: dictionary where to save losses during the training. Mostly useful when restarting a run and
        we need to know previous values of losses.
        rec_coef: weight of the reconstruction
        grp_g_coef: weight of the group
        restart_training: Should we restart the training from scratch ?
        bar_info: Additional infos to display with the progress bar
        checkpoint_dir: location to save checkpoints
        nb_groups: total number of groups in the dataset. Necessary to compute the target value of the BER.
        logger: Class that handle the logging of metrics with wandb and mlflow.
        metrics_computation: Module to compute metrics with.

    Returns:

    """

    start = 0
    if not restart_training and checkpoint_dir:
        start, loaded_errors_dict = load_models(checkpoint_dir, "Gw",
                                         Gw=gw, gw_optim=gw_optim.optimizer, gw_sch=gw_scheduler, D=discriminator,
                                         d_optim=d_optim.optimizer)
        if len(loaded_errors_dict):
            errors_dict = {"epoch": loaded_errors_dict["epoch"].values.tolist(),
                           "losses": loaded_errors_dict["losses"].values.tolist(),
                           "type": loaded_errors_dict["type"].values.tolist()}

    # Set the loss value for the sensitive groups when training the generator and the discriminator
    g_loss_target_value = (nb_groups - 1) / nb_groups
    d_loss_target_value = 0
    if groups_criterion.__class__.__name__ == "AccuracyLoss":
        g_loss_target_value = 0
        d_loss_target_value = 1

    for epoch in (pbar := tqdm.tqdm(range(start + 1, training_epochs + 1))):
        gw.train()
        pbar.set_description(bar_info)
        losses = {"g_grp_loss": [], "disc_grp_loss": [], "g_rec_loss": []}
        sch_step = False
        start_time = time.time()
        for i, batch_d in enumerate(train_loader):
            x = batch_d["x"]
            group_code = batch_d["y_float"].to(discriminator.device, non_blocking=True)
            # x.requires_grad = True  # Necessary to compute the gradient penalty
            z = x

            # Generate a batch of images
            z = z.to(gw.device, non_blocking=True)
            noise = get_noise(noise_dim, x.size(0), device=gw.device)
            fake_x = gw(z, noise=noise)

            # Fake images
            fake_x_grp = discriminator(fake_x)

            grp_loss = groups_criterion(fake_x_grp, group_code, group=group_code, loss_target_value=d_loss_target_value)

            d_optim.zero_grad()
            d_optim.backward([grp_loss], inputs=list(discriminator.parameters()))
            d_optim.optim_step()

            losses["disc_grp_loss"].append(grp_loss.item())

            if (i + 1) % n_critic == 0:
                sch_step = True

                # Generate a batch of images
                noise = get_noise(noise_dim, x.size(0), device=gw.device)
                fake_x = gw(z, noise=noise)
                # Fake images
                fake_x_grp = discriminator(fake_x)

                g_grp_loss = groups_criterion(fake_x_grp, group_code, group=group_code,
                                              loss_target_value=g_loss_target_value)
                losses["g_grp_loss"].append(g_grp_loss.item())

                losses_2_backward = [g_grp_loss * grp_g_coef]

                # x_wo_s : x without s, the sensitive groups
                # x_twin_drp : x values but with additional dropped columns.
                x_wo_s = batch_d["x_twin_drp"]
                x_wo_s = x_wo_s.to(gw.device, non_blocking=True)
                rec_prv_loss = rec_criterion(x_wo_s, fake_x)
                losses_2_backward.append(rec_coef * rec_prv_loss)
                losses["g_rec_loss"].append(rec_prv_loss.mean().item())

                gw_optim.zero_grad()
                gw_optim.backward(losses_2_backward, inputs=list(gw.parameters()))
                gw_optim.optim_step()

                del g_grp_loss, rec_prv_loss
            del x, fake_x, z, group_code, fake_x_grp, grp_loss, noise

        epoch_time = time.time()
        if sch_step:
            gw_scheduler.step()
        losses_to_log = {}
        for type_, vals in losses.items():
            losses_to_log.update({type_: np.array(vals).mean()})
            errors_dict["losses"].append(np.array(vals).mean())
            # if "d_penalty" == type_:
            #     errors_dict["losses"][-1] = np.log(np.array(vals).mean())
            errors_dict["type"].append(type_)
            errors_dict["epoch"].append(epoch)

        elapsed_time = epoch_time - start_time
        losses_to_log.update({"Elapsed_time": elapsed_time})
        errors_dict["losses"].append(elapsed_time)
        errors_dict["type"].append("Elapsed_time")
        losses_to_log.update({"step": epoch})
        errors_dict["epoch"].append(epoch)
        # if epoch == ae_epochs:
        # logger.log(losses_to_log)

        if metrics_computation.is_testing_set:
            # Ignore the computation of metrics if we are still training because we lost the checkpoint or whatever.
            # Only save the final model so that we do not need to redo the training again.
            save_models(checkpoint_dir, epoch=epoch, name="Gw", losses=errors_dict, Gw=gw,
                        gw_optim=gw_optim.optimizer, gw_sch=gw_scheduler, D=discriminator,
                        d_optim=d_optim.optimizer)
        else:
            # We are training with the validation set. Compute the metrics so we can build the pareto front.
            gw.eval()
            results = metrics_computation.compute_test_metrics_ext_val(g_map=gw)
            losses_to_log.update(results)
            if (epoch % args.model_save_rate == 0) or (epoch == training_epochs):
                logger.log_metrics(losses_to_log, step=epoch, name="Gw", losses=errors_dict, Gw=gw,
                            gw_optim=gw_optim.optimizer, gw_sch=gw_scheduler, D=discriminator,
                            d_optim=d_optim.optimizer)
            else:
                # Log only without saving the models.
                logger.log_metrics(losses_to_log, step=epoch)

    # Set gw into evaluation mode : for batchnorm and dropout.
    gw.eval()

    print("Finished Training Gw")


def process(config, data_info, exp_folders, validation_index, test_index, decision_type, train_with_s,
            checkpoint_dir=None, is_testing_set=False, exp_name="GANSan", transform_whole_set=False, trial_id=None):
    """

    Args:
        config:
        data_info:
        exp_folders:
        validation_index:
        test_index:
        decision_type:
        train_with_s:
        group_loss:
        checkpoint_dir:
        is_testing_set:
        exp_name:
        transform_whole_set
        trial_id
    Returns:

    """
    #
    bar_info = "{}: {}-{} -- {{}}".format(socket.gethostname(), args.dataset, data_info["SetName"])

    if checkpoint_dir is not None:
        exp_folders = copy.deepcopy(exp_folders)
        exp_folders.models_dir = checkpoint_dir

    # --------------------------------------------------------------------------------------------------------------- #
    # DATA INITIALISATION
    to_drop = data_info["OriginalGroupAttributes"][:]
    if decision_type == "original":
        to_drop += [data_info["LabelName"]]  # Add this to remove label from modified data
    dropped_cols = to_drop[:] + data_info["GroupAttributes"][:]

    train_prep, valid_prep, test_prep = get_datasets(val_index=validation_index, test_index=test_index,
                                                     exp_folders=exp_folders, data_info=data_info,
                                                     ignore_validation=is_testing_set, ignore_test=not is_testing_set)

    enc_dropped_cols = train_prep.__filter_features__(dropped_cols)
    enc_orig_group_names_list = [train_prep.__filter_features__(v) for v in data_info["OriginalGroupAttributes"]]
    enc_group_names = train_prep.__filter_features__(data_info["GroupAttributes"])
    enc_decision_names = train_prep.__filter_features__(data_info["LabelName"])

    torch_dataset = d.TorchGeneralDataset(train_prep.df, target_features=data_info["GroupAttributes"],
                                          drop_target=not train_with_s, drop_from_x_twin=data_info["GroupAttributes"],
                                          to_drop=to_drop)
    # If train_with_s is False (drop target is True), we drop the sensitive attribute from the model input variable x.
    # So x and x_twin will be the same.
    # If train_with_s is True (drop target is False), we do not drop the sensitive attribute from the model input
    # variable x. But since drop_from_x_twin hold the sensitive attribute, x_twin will be identical to x, but without
    # the sensitive columns. Thus, we can train the reconstruction objective using x_twin.

    groups_dim = torch_dataset.yf.size(1)  # Group is the variable to predict
    input_dim = torch_dataset.x.size(1)
    reconstructed_dim = torch_dataset.x_twin_drp.size(1)

    num_workers = args.num_workers * (not args.debug)
    train_loader = torchd.DataLoader(torch_dataset, batch_size=2 ** config["batch_size_exp"], shuffle=True,
                                     num_workers=num_workers, pin_memory=CUDA)
    train_loader_in_a_single_batch = torchd.DataLoader(torch_dataset, batch_size=train_prep.df.shape[0], shuffle=False,
                                                       num_workers=num_workers, pin_memory=CUDA)

    if is_testing_set:
        eval_data_prep = test_prep
        set_index = test_index

        task_path = "{}/taskResults/".format(checkpoint_dir)
        t_fig_path = "{}/taskResults/Figs/".format(checkpoint_dir)
        if not os.path.exists(task_path):
            os.makedirs(task_path)
        if not os.path.exists(t_fig_path):
            os.makedirs(t_fig_path)
        exp_folders.fig_dir = t_fig_path
        exp_folders.results_dir = task_path
    else:
        eval_data_prep = valid_prep
        set_index = validation_index

    loader = torchd.DataLoader(d.TorchGeneralDataset(eval_data_prep.df, target_features=data_info["GroupAttributes"],
                                                     drop_target=not train_with_s,
                                                     drop_from_x_twin=data_info["GroupAttributes"],
                                                     to_drop=to_drop),
                               batch_size=eval_data_prep.df.shape[0], shuffle=False,
                               num_workers=args.num_workers * CUDA,
                               pin_memory=CUDA)
    # -------------------------------------------------------------------------------------------------------------- #
    # SETTING UP DATA MASKS
    train_target_mask = get_target_mask(train_prep.df, data_info)
    eval_data_target_mask = get_target_mask(eval_data_prep.df, data_info)

    # ---------------------------------------------------------------------------------------------------------------- #
    # SETTING UP MODULE METRICS
    div_module = qm.Diversity()
    dist_module = qm.Distance()
    dam_mod = qm.Damage(categorical_in_list=True)
    train_prep.transform()
    fair_mod = std_m.FairnessSimplified(data_info["LabelName"], data_info["GroupAttributes"][0],
                                        orig_group_names=data_info["OriginalGroupAttributes"][:],
                                        enc_decision_names=enc_decision_names, enc_group_names=enc_group_names,
                                        enc_orig_group_names_list=enc_orig_group_names_list,
                                        enc_to_drop=list(set(enc_dropped_cols) - set(enc_group_names) -
                                                         set(enc_decision_names)), seed=RandomState)

    # ---------------------------------------------------------------------------------------------------------------- #

    if "gpu" in RAY_RESSOURCES_PER_TRIALS.keys() and RAY_RESSOURCES_PER_TRIALS["gpu"] > 1:
        tune.utils.wait_for_gpu()

    # Models optimisers
    rec_criterion = nn.L1Loss(reduction="none")
    # ---------------------------------------------------------------------------------------------------------------- #
    mi_criterion = MILoss()

    if CUDA:
        torch.cuda.empty_cache()

    generator_hidden_layers = [2 ** config["generator_hidden_layer_{}".format(i)] for i in range(1, 2 + 1)]
    discriminator_hidden_layers = [ 2 ** config["discriminator_hidden_layer_{}".format(i)] for i in range(1, 3 + 1)]

    # gw = models.Generator(input_dim + config["noise_dim"], generator_hidden_layers, reconstructed_dim,
    #                       checkpointing=CUDA and args.checkpointing)
    gw = models.Generator(input_dim + config["noise_dim"], generator_hidden_layers, reconstructed_dim,
                          checkpointing=True)
    gw.to(DEVICE)
    gw_optimizer = o.DualVectorOptim(gw.parameters(), lr=config["generator_learning_rate"], weight_decay=1e-8,
                                     betas=(0.5, 0.999))
    gw_scheduler = optim.lr_scheduler.StepLR(gw_optimizer.optimizer, step_size=250, gamma=config["gw_gamma"])

    if config["group_loss"] == "Ber":
        groups_criterion = BalancedErrorRateLoss(device=DEVICE)
    else:
        groups_criterion = AccuracyLoss(device=DEVICE)

    # Models optimisers
    discriminator = models.Predictor(reconstructed_dim, discriminator_hidden_layers, output_dim=groups_dim,
                                     output_probs="softmax")
    discriminator.to(DEVICE)
    d_optimizer = o.DualVectorOptim(discriminator.parameters(), lr=config["discriminator_learning_rate"],
                                    weight_decay=1e-8, betas=(0.5, 0.999))

    # Training parameters
    n_critic = max(min(15, train_prep.df.shape[0] // 2 ** config["batch_size_exp"] - 1), 1)

    gw_optim = gw_optimizer
    d_optim = d_optimizer

    errors_wgan_dict = {"epoch": [], "losses": [], "type": []}
    # Set wandb login
    # logger = setup_wandb(config=config, project="GANSan")
    logger = Logs(wandb_init=dict(config=config, project=exp_name, api_key=os.environ["WANDB_API_KEY"],
                                  rank_zero_only=False, name=trial_id),
                  # need to ensure that all workers will initialise wandb. Otherwise, they will return a RunDisable
                  # object, that can not be used.
                  mlflow_init=dict(config=config, experiment_name=exp_name, run_name=trial_id,
                                   tracking_uri=os.environ['MLFLOW_TRACKING_URI']))
    metrics_computation = MetricComputation(exp_folders, set_index, config["noise_dim"], train_prep, eval_data_prep,
                                            train_loader_in_a_single_batch, loader, train_target_mask,
                                            eval_data_target_mask, enc_dropped_cols, enc_group_names, to_drop,
                                            dist_module, div_module, dam_mod, fair_mod, mi_criterion,
                                            is_testing_set=is_testing_set)

    train_gw(gw, discriminator, gw_optim, gw_scheduler, d_optim, groups_criterion, rec_criterion,
             train_loader, config["epochs"], n_critic, errors_wgan_dict, noise_dim=config["noise_dim"],
             rec_coef=config["reconstruction"], grp_g_coef=config["group_g"], restart_training=False,
             bar_info=bar_info.format("Gw"), checkpoint_dir=checkpoint_dir, nb_groups=groups_dim,
             logger=logger, metrics_computation=metrics_computation)

    # Compute final results
    results = None
    if is_testing_set:
        results = metrics_computation.compute_test_metrics_ext_val(g_map=gw)
        logger.log_metrics(results)

    if transform_whole_set:
        whole_set_prep, _, _ = get_datasets(val_index=0, test_index=0, exp_folders=exp_folders, data_info=data_info,
                                       ignore_validation=True, ignore_test=True)
        whole_set_loader = torchd.DataLoader(
            d.TorchGeneralDataset(whole_set_prep.df, target_features=data_info["GroupAttributes"],
                                  drop_target=not train_with_s, drop_from_x_twin=data_info["GroupAttributes"],
                                  to_drop=to_drop), batch_size=whole_set_prep.df.shape[0], shuffle=False,
            num_workers=args.num_workers * CUDA, pin_memory=CUDA)
        metrics_computation.transform_data(g_map=gw, data_loader=whole_set_loader, data_prep=whole_set_prep,
                            transform=False)
        generated_data_path = f"{checkpoint_dir}/generated_data/"
        if not os.path.exists(generated_data_path):
            os.makedirs(generated_data_path)
        whole_set_prep.df.to_csv("{}/{}-whole-set-1-{}.csv".format(generated_data_path, data_info["SetName"],
                                                                   KFolds), index=False)
        print("Generated Dataset Located at: {}".format(generated_data_path))

    if CUDA:
        torch.cuda.empty_cache()
    logger.finish()
    return results


# -------------------------------------------------------------------------------------------------------------------- #
# Ray functions

def sample_parameters(func, do_sample=args.debug):
    """
    Sample the parameter and return it or return the function itself.
    Useful if debugging.
    Like a wrapper of the parameter func.
    Args:
        func: function to sample parameters from
        do_sample: whether we should sample or not.

    Returns:

    """
    if do_sample:
        return func.sample()
    else:
        return func


def search_space():
    """
    Get the parameters search space for hyperparameters search.
    Returns:

    """
    param_space = {
        "gw_gamma": sample_parameters(tune.uniform(1e-1, 0.8)),
        "batch_size_exp": sample_parameters(tune.qrandint(4, 10, 1)),
        # "batch_size": sample_parameters(tune.qrandint(128, 1024, 128)),
        "epochs": sample_parameters(tune.qrandint(300, args.max_epochs, 100)),
        "reconstruction": sample_parameters(tune.uniform(1e-1, 1)),
        "group_g": sample_parameters(tune.uniform(1e-1, 1)),
        "noise_dim": sample_parameters(tune.qrandint(1, 2, 1)),
        # "noise_dim": sample_parameters(tune.qrandint(1, 4, 1)),
        # "generator_hidden_layer_1": sample_parameters(tune.qrandint(5, 13, 1)),
        # "generator_hidden_layer_2": sample_parameters(tune.qrandint(10, 13, 1)),
        # # "generator_hidden_layer_2": sample_parameters(tune.qrandint(5, 13, 1)),

        "generator_hidden_layer_1": sample_parameters(tune.qrandint(7, 8, 1)),
        "generator_hidden_layer_2": sample_parameters(tune.qrandint(7, 8, 1)),
        "discriminator_hidden_layer_1": sample_parameters(tune.qrandint(9, 10, 1)),
        "discriminator_hidden_layer_2": sample_parameters(tune.qrandint(7, 8, 1)),
        "discriminator_hidden_layer_3": sample_parameters(tune.qrandint(6, 7, 1)),


        # "discriminator_hidden_layer_1": sample_parameters(tune.qrandint(5, 13, 1)),
        # "discriminator_hidden_layer_2": sample_parameters(tune.qrandint(5, 13, 1)),
        # "discriminator_hidden_layer_3": sample_parameters(tune.qrandint(5, 13, 1)),
        "generator_learning_rate": sample_parameters(tune.loguniform(2e-4, 2e-1)),
        "discriminator_learning_rate": sample_parameters(tune.loguniform(2e-4, 2e-1)),
        "group_loss": sample_parameters(tune.choice(["Ber", "Acc"]))
    }
    concurrency_limiter = False
    if args.debug:
        return param_space, None, None, concurrency_limiter

    search_algo = OptunaSearch(metric=list(METRICS_AND_MODES.keys()),
                               mode=list(METRICS_AND_MODES.values()), seed=RandomState)
    if args.mo == "A-SC":
        # search_algo = BasicVariantGenerator(max_concurrent=args.concurrent_trial, random_state=RandomState)
        scheduler = ASHAScheduler(time_attr='training_iteration', metric=list(METRICS_AND_MODES.keys())[0],
                                  mode=list(METRICS_AND_MODES.values())[0], max_t=args.max_epochs, grace_period=45)

    else:
        # search_algo = OptunaSearch(metric=list(METRICS_AND_MODES.keys()),
        #                            mode=list(METRICS_AND_MODES.values()), seed=RandomState)
        scheduler = None
        # No need for explicitly use concurrency limiter here, we can use the max_concurrent_trials of tune.config.
        # Actually, we do, as we can provide more options to the concurrency limiter.
        concurrency_limiter = True
        if args.concurrent_trial:
            search_algo = ConcurrencyLimiter(search_algo, max_concurrent=args.concurrent_trial,
                                             batch=args.ccrency_batches)
    return param_space, search_algo, scheduler, concurrency_limiter


def launch(data_name, validation_index, test_index):
    """
    Procedure to start the experiments.
    Args:
        data_name: data to start experimenting with
        validation_index : index of the set that will be used for the validation set
        test_index: Index of the set that will be used for the test set

    Returns:

    """
    param_space, search_algo, scheduler, concurrency_limiter = search_space()

    if args.debug:
        exp_name = "-".join(["GANSan-Debug", args.suffix, args.mo])
        data_info, exp_folders = init(data_name, approach_name=exp_name)
        # exp_folders.update_fold(fold_id=None)
        exp_folders.update_fold(fold_id=exp_name)

        process(param_space, data_info=data_info, exp_folders=exp_folders, validation_index=validation_index,
                test_index=test_index, decision_type=args.decision, train_with_s=True,
                checkpoint_dir=exp_folders.models_dir, is_testing_set=False, exp_name=exp_name)
    else:
        exp_name = "-".join(["GANSan", args.suffix, args.mo])
        data_info, exp_folders = init(data_name, approach_name=exp_name)
        # exp_folders.update_fold(fold_id=None)
        exp_folders.update_fold(fold_id=exp_name)

        resume_path = os.path.join(exp_folders.models_dir, exp_name)

        trainable = partial(process, data_info=data_info, exp_folders=exp_folders, validation_index=validation_index,
                            test_index=test_index, is_testing_set=False, decision_type=args.decision, train_with_s=True,
                            exp_name=exp_name
                            )
        trainable_with_resources = tune.with_resources(trainable, RAY_RESSOURCES_PER_TRIALS)

        if args.resume and tune.Tuner.can_restore(resume_path):
            tuner = tune.Tuner.restore(resume_path, trainable=trainable, resume_errored=True,
                                       restart_errored=False, resume_unfinished=True, )
        else:
            tuner = tune.Tuner(
                trainable_with_resources,
                tune_config=tune.TuneConfig(search_alg=search_algo, num_samples=args.ntrials, scheduler=scheduler,
                                            max_concurrent_trials=args.concurrent_trial if not concurrency_limiter else
                                            None),
                param_space=param_space,
                run_config=train.RunConfig(
                    name=exp_name,
                    storage_path=exp_folders.models_dir,
                    # callbacks=[WandbLoggerCallback(project=exp_name)]
                )
            )

        result_grid = tuner.fit()
        for metric_mode, hp_tuning_name in zip([HEURISTIC_A, MO_FP], ["heuristicA", "FP"]):
            # Get checkpoint with heuristicA
            pareto_front = utils.pareto_front_from_dataframe(result_grid.get_dataframe(), metric_mode)
            paths_configs = utils.ray_post_processing(pareto_front,
                                                      transfer_path=f"{exp_folders.results_dir}/{hp_tuning_name}",
                                                      experiment_path=f"{exp_folders.models_dir}/{exp_name}",
                                                      max_epoch_keyword="epochs")
            # Save pareto front
            pareto_front.to_csv(f"{exp_folders.results_dir}/{hp_tuning_name}/pareto_front.csv", index=False)
            # Plot pareto front
            # Compute test results
            if len(paths_configs):
                # We have at least one configuration where the checkpoint exist.
                index = -1
                for path, config in paths_configs.items():
                    # Compute test results. Will be saved directly on disk.
                    # Model will be automatically load, No training will be conducted, since max epoch has been
                    # adjusted to value of the selected checkpoint.
                    index += 1
                    # Get the trial Id,
                    trial_id = pareto_front.iloc[index]["trial_id"]
                    test_result = process(config, data_info=data_info, exp_folders=exp_folders,
                                          validation_index=validation_index, test_index=test_index,
                                          decision_type=args.decision, train_with_s=True,
                                          checkpoint_dir=path, is_testing_set=True, exp_name=f"{exp_name}-Pareto",
                                          # transform_whole_set=hp_tuning_name == "heuristicA",
                                          transform_whole_set=True,
                                          trial_id=f"{data_name}-{hp_tuning_name}-{trial_id}")
                    # Automatically Generate the dataset when using heuristicA


HEURISTIC_A = {"heuristic A": "min"}
MO_FP = {"fidelity: all": "max", "protection: R-M": "max"}
MO_FPL = {"fidelity: all": "max", "protection: R-M": "max", "g_rec_loss": "min", "g_grp_loss": "min",
          "disc_grp_loss": "min"}
##################################################################################
if args.mo == "FP":
    METRICS_AND_MODES = MO_FP
elif "A-" in args.mo:
    METRICS_AND_MODES = HEURISTIC_A
elif args.mo == "FPL":
    METRICS_AND_MODES = MO_FPL
launch(args.dataset, validation_index=3, test_index=5)

print("Completed")
### Plot pareto front in wandb and mlflow
### Corriger l'erreur sur Exp. En fait, mode SC avait metrics fid all et prot, et non heuristicA, change on disc folders
### Reduire noise node et augmenter disc min size ?
### Finir debug
### Envoyer données
### Ajouter quick mlflow
### Relancer avec SC mode.


### Comit
### Push
### Send results
### Do your readings
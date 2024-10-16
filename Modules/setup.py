import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Modules import utils as u


class ExperimentsFolders:
    """
    Class to hold directories for experiments
    """

    def __init__(self, base_dir, set_name, data_key_name, approach_name, prm_dir="/Prm/", prepared_data_dir="/Csv/",
                 is_k_val=False, exp_number=None, root_base=None):
        """
        Create the folder in which we will run the experiments

        :param base_dir: Base directory where all experiments will be saved.
        :param set_name: The dataset currently using
        :param data_key_name: The type of investigation : { Sampling, Outliers, Fairtest}
        :param approach_name: The name of the approach currently under investigation
        :param prm_dir : name of directory where to save the dataset extracted parameters
        :param prepared_data_dir: The dataset where to save the prepared data.
        :param is_k_val: prepare folders for k validation
        :param exp_number: cross validation experiment number
        :return: False if k_val is False, else will check if the csv folder (not the content) exists. If so, return True
        """
        exp_number = "TrainTest" if not (is_k_val and exp_number is not None) else "Exp={}/".format(exp_number)
        self.prm_dir = "{}/{}/{}/{}/{}".format(base_dir, data_key_name, set_name, exp_number, prm_dir)
        self.root_base = root_base
        self.exp_sets_dir = prepared_data_dir
        if is_k_val:
            self.exp_sets_dir = "{}/{}/{}/{}/Csv/".format(base_dir, data_key_name, set_name, exp_number)
        for t in [self.prm_dir, self.exp_sets_dir]:
            self.make_directory(t)

        self.set_name = set_name
        self.sampling_type = data_key_name

        # Fold id is updated elsewhere. To a
        self.exp_dir = lambda fold_id: "{}/{}/{}/{}/{}/{}".format(base_dir, data_key_name, set_name, exp_number,
                                                                  approach_name, fold_id)
        self.gen_dir = None
        self.models_dir = None
        self.results_dir = None
        self.fig_dir = None
        self.current_model_dir = None

    def __str__(self):
        """ Format String """
        return " -- {}\n -- {}\n -- {}\n -- {}\n".format(self.gen_dir, self.models_dir, self.results_dir, self.fig_dir)

    def k_validation_folders(self, nb_splits, split_name="{}.csv"):
        """ Check if the folders for k_validation are set up
        :param nb_splits: number of splits to consider
        :param split_name: name of each split (without indexes) """
        # Thoroughly check the csv folder content
        exists = True
        for i in range(1, nb_splits + 1):
            exists = exists and u.try_reading_dataframe("{}/{}".format(self.exp_sets_dir, split_name.format(i)))
        return exists

    @staticmethod
    def make_directory(path):
        """ Create directory """
        if path is not None:
            try:
                if not os.path.exists(path):
                    # If path does not exists create it
                    os.makedirs(path)
            except FileExistsError as e:
                # When checked, file did not exits. However, a concurrent process might have create the path before
                # this one was able to. Hence it throws a FileExistsError
                pass

    def update_fold(self, fold_id="", resultsDir="/Results/", genDir="/SetsModified/", modelDir="/Models/",
                    figuresDir="Figures", specific_id=""):
        """ Update the fold_id and the sub exp dirs
        :param resultsDir: The directory where the results will be saved
        :param GenDir: location where to save modified datasets.
        :param modelDir: directory where to save models if a training is involved
        """
        fold_id = "" if fold_id is None else "CrossVal={}".format(fold_id)
        self.gen_dir = "{}/{}/{}/".format(self.exp_dir(fold_id), genDir, specific_id)
        self.models_dir = "{}/{}/{}/".format(self.exp_dir(fold_id), modelDir, specific_id)
        self.results_dir = "{}/{}/{}/".format(self.exp_dir(fold_id), resultsDir, specific_id)
        self.fig_dir = "{}/{}/{}/".format(self.exp_dir(fold_id), figuresDir, specific_id)
        if modelDir is not None:
            self.current_model_dir = "{}/{}/{}/".format(self.exp_dir(fold_id), modelDir, specific_id)

        for t in [self.exp_dir(fold_id), self.results_dir, self.models_dir, self.fig_dir, self.gen_dir,
                  self.models_dir]:
            self.make_directory(t)

    def _similar_as_copy_(self, similar_as):
        """ Just copy files and make experiments similar """
        c = self.exp_sets_dir.split("_")
        p = self.prm_dir.split("_")
        for v in reversed(c[-1].split("/")):
            if len(v) != 0:
                break
        for t in reversed(p[-1].split("/")):
            if len(t) != 0:
                break
        if similar_as is not None:
            if isinstance(similar_as, int):
                similar_as_csv = "_".join(["_".join(c[:-1]), "/".join([str(similar_as), v, "/"])])
                similar_as_prm = "_".join(["_".join(p[:-1]), "/".join([str(similar_as), t, "/"])])
            if os.path.isfile("{}/{}.csv".format(similar_as_csv, 1)):
                os.system("cp -rv {}/* {}".format(similar_as_csv, self.exp_sets_dir))
                os.system("cp -rv {}/* {}".format(similar_as_prm, self.prm_dir))
                return len([name for name in os.listdir(self.exp_sets_dir) if os.path.isfile(name)])

    def k_splits(self, data, nb_splits, random_state, stratify_feature, similar_as=None):
        """ Just make the k fold split """
        sim = self._similar_as_copy_(similar_as)
        if sim is not None:
            return sim
        test = data
        nbS = int((1 / nb_splits) * test.shape[0])
        for i in range(1, nb_splits):
            st_f = stratify_feature
            good = False
            while not good:
                try:
                    train, test = train_test_split(test, train_size=nbS, random_state=random_state,
                                                   stratify=test[st_f].values)
                    good = True
                except ValueError as e:
                    if "The least populated class in y has only 1 member," in str(e):
                        print(
                            "Cannot perform stratification. Reason: {}\n Proceeding without the last attribute".format(
                                e))
                        st_f = st_f[:-1]
                    else:
                        raise ValueError(e)
            # Save set
            train.to_csv("{}/{}.csv".format(self.exp_sets_dir, i), index=False)
        test.to_csv("{}/{}.csv".format(self.exp_sets_dir, nb_splits), index=False)

        return i + 1

    def size_split(self, data, train_size, random_state, stratify_feature, similar_as=None, get_data=False):
        """ Split in training and test sets, according to the given training and testing sizes. """
        sim = self._similar_as_copy_(similar_as)
        if sim is not None:
            return sim
        test = data
        st_f = stratify_feature
        good = False
        while not good:
            try:
                train, test = train_test_split(test, train_size=train_size, random_state=random_state,
                                               stratify=test[st_f].values)
                good = True
            except ValueError as e:
                if "The least populated class in y has only 1 member," in str(e):
                    print(
                        "Cannot perform stratification. Reason: {}\n Proceeding without the last attribute".format(
                            e))
                    st_f = st_f[:-1]
                else:
                    raise ValueError(e)
        # Save set
        test.to_csv("{}/{}.csv".format(self.exp_sets_dir, 1), index=False)
        train.to_csv("{}/{}.csv".format(self.exp_sets_dir, 2), index=False)
        if get_data:
            return test, train
        return 2

    def group_extract(self, data, train_size, group_to_extract, group_attributes, maj_ids, min_ids, random_state,
                      stratify_feature, similar_as=None):
        """ Extract the group as asked and add it to the test set. If the extracted group correspond to the group to
         keep, then move one step above or below the chosen group """
        sim = self._similar_as_copy_(similar_as)
        if sim is not None:
            return sim
        groups = data.groupby(by=group_attributes).ngroup()
        g_sizes = groups.value_counts()

        def check(sample, g_at):
            """ check if can extract the group """
            for kp in [maj_ids, min_ids]:
                msk = []
                for at, vls in zip(g_at, kp):
                    msk.append(sample[at].isin(vls))
                if np.logical_and.reduce(msk):
                    return False
            return True

        def get_correct_group(dt, g_indexes, g_ordered, g_at):
            """ get the group possible to extract """
            for g in g_ordered.index:
                sample = dt[g_indexes == g].iloc[:1]
                if check(sample, g_at):
                    return g

        def extract(dt, g_indexes, g_ordered, g_at):
            """ Extract the correct group """
            sel_g = get_correct_group(dt, g_indexes, g_ordered, g_at)
            sel_mask = g_indexes == sel_g
            g_extracted = dt[sel_mask]
            other_data = dt[~sel_mask]
            test, train = self.size_split(other_data, train_size, random_state, stratify_feature, similar_as=None,
                                          get_data=True)
            test = pd.concat([test, g_extracted], axis=0, sort=True)
            test.to_csv("{}/{}.csv".format(self.exp_sets_dir, 1), index=False)
            g_extracted.to_csv("{}/{}.csv".format(self.exp_sets_dir, "group_extracted"), index=False)

        if group_to_extract == "least":
            # Extract the least present group, as long as it does not correspond to the group to keep
            g_orders = g_sizes.sort_values(ascending=True)
        elif group_to_extract == "most":
            # Extract the most present group, as long as it does not correspond to the group to keep
            g_orders = g_sizes.sort_values(ascending=False)
        elif group_to_extract == "rand":
            # Extract a random group, as long as it does not correspond to the group to keep
            g_orders = g_sizes.sample(frac=1)
        extract(dt=data, g_indexes=groups, g_ordered=g_orders, g_at=group_attributes)

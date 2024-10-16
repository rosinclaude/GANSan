import numpy as np
import pandas as pd
from Modules import classifiers as clfrs
from Modules import datasets as dts
from Modules import Visualisation


class Fairness:
    """ Compute utility metrics
     - Compute Demographic parity
     - Compute Equalized odds
     """

    def __init__(self, decision_name, enc_decision_names, enc_group_names, to_drop=[], seed=42):
        self.classifiers = clfrs.Classifiers(seed=seed)
        self.decision_name = decision_name
        self.enc_decision_names = enc_decision_names
        self.enc_group_names = enc_group_names
        self.to_drop = to_drop
        self.visualise = Visualisation.Visualize(seed=seed)
        self.results = None

    def reset_results(self):
        self.results = {
            "Attribute": [],
            "Classifier": [],
            "Metric": [],
            "Type": [],
            "Result": [],
            "GroupValue": [],
            "Scenario": [],
            "GroupName": [],
            "ResultType": [],
        }

    def add_results(self, **kwargs):
        for k, v in kwargs.items():
            self.results[k].append(v)
        for k in list(set(self.results.keys()) - set(kwargs.keys())):
            self.results[k].append("-")

    def train_models(self, x, y):
        """ Just train the models. Helps specifying the data input."""
        if len(y.shape) > 1:
            y = y.values.argmax(1)
        self.classifiers.fit(x, y)

    def split(self, data):
        to_drop = self.to_drop[:]
        to_drop.extend(self.enc_group_names)
        to_drop.extend(self.enc_decision_names)
        x = data.drop(to_drop, axis=1)
        y = data[self.enc_decision_names]
        s = data[self.enc_group_names]
        return x, y, s

    def positive_rates(self, pred_y, groups, g_name, scn, type_, y_name, clf_name, result_type, pos_y_value=1,
                       metric=None):
        """ Compute the positive rate for all given subgroups. Can also be used to compute the accuracy per
        group or the base rates
        """

        if metric is None:
            metric = "Pos. Rate"
        if pred_y is np.NaN:
            for g_index in groups.value_counts().index:
                self.add_results(Attribute=y_name, Metric=metric, Classifier=clf_name, Type=type_, Scenario=scn,
                                 Result=np.NaN, GroupValue=g_index, GroupName=g_name, ResultType=result_type)
        else:
            positive_predictions = (pred_y == pos_y_value) * 1
            d = pd.DataFrame.from_dict({"pos_pred": positive_predictions.reshape(-1),
                                        "groups": groups.values.reshape(-1)})
            d = d.groupby(by="groups").mean()
            d.reset_index(inplace=True)
            d.set_index("groups", inplace=True)
            pos = d["pos_pred"].to_dict()
            for group, result in pos.items():
                self.add_results(Attribute=y_name, Metric=metric, Classifier=clf_name, Type=type_, Scenario=scn,
                                 Result=result, GroupValue=group, GroupName=g_name, ResultType=result_type)

    def tp_and_fp(self, pred_y, y, groups, g_name, scn, type_, y_name, clf_name, result_type, pos_y_value=1):
        """ Compute the true positive and false positive rates. """

        if pred_y is np.NaN:
            for g_index in groups.value_counts().index:
                self.add_results(Attribute=y_name, Metric="Tp. Rate", Classifier=clf_name, Type=type_, Scenario=scn,
                                 Result=np.NaN, GroupValue=g_index, GroupName=g_name, ResultType=result_type)
                self.add_results(Attribute=y_name, Metric="Fp. Rate", Classifier=clf_name, Type=type_, Scenario=scn,
                                 Result=np.NaN, GroupValue=g_index, GroupName=g_name, ResultType=result_type)
        else:
            def add_missing_group(miss_v, df, c_name, g_name="groups"):
                if len(miss_v):
                    empty = pd.DataFrame.from_dict({g_name: miss_v, c_name: [np.NaN for i in miss_v]})
                    empty.set_index(g_name, inplace=True)
                    return pd.concat([df, empty], axis=0)
                return df

            def compute(result_vector, group, metric_name, tmp_m_name):
                df = pd.DataFrame.from_dict({tmp_m_name: result_vector, "groups": group})
                df = df.groupby(by="groups").mean()
                missing_groups = np.setdiff1d(np.unique(groups), np.unique(g))
                df = add_missing_group(missing_groups, df, tmp_m_name, "groups")
                df.reset_index(inplace=True)
                df.set_index("groups", inplace=True)
                for grp, result in df[tmp_m_name].items():
                    self.add_results(Attribute=y_name, Metric=metric_name, Classifier=clf_name, Type=type_,
                                     Scenario=scn, Result=result, GroupValue=grp, GroupName=g_name,
                                     ResultType=result_type)

            positive_mask = y == pos_y_value
            pred_y = pred_y.reshape(y.shape)
            tp = (pred_y[positive_mask] == pos_y_value) * 1
            fp = (pred_y[~positive_mask] == pos_y_value) * 1
            groups = groups.values.reshape(-1)

            # True positives
            g = groups[positive_mask]
            compute(result_vector=tp, group=g, metric_name="Tp. Rate", tmp_m_name="tp")

            # False positives
            g = groups[~positive_mask]
            compute(result_vector=fp, group=g, metric_name="Fp. Rate", tmp_m_name="fp")

    def fairness_results(self, x, y, rs, s, g_index, rg_index, y_name, type_, scn, result_type):
        """ Compute the fairness results using input x, target y, sensitive attributes s and group index g and
        the relabelled group index rg """

        if len(y.shape) > 1:
            if not isinstance(y, np.ndarray):
                y = y.values
            if y.shape[1] > 1:
                y = y.argmax(1)
            y.reshape(-1)
        # 1- Predict
        preds_y = self.classifiers.predict(x)
        for clf_name, pred_y in preds_y.items():
            # 2- Compute accuracy
            if pred_y is not np.NaN:
                pred_y = pred_y.reshape(-1)
            correct_pred = (pred_y == y) * 1
            # General accuracy
            self.add_results(Attribute=y_name, Metric="Accuracy", Classifier=clf_name, Type=type_, Scenario=scn,
                             Result=correct_pred.mean(), GroupValue="-", GroupName="AllGroups", ResultType=result_type)
            # Group by sensitive attribute
            columns = rs.columns.tolist()
            columns += s.columns.tolist()
            columns += ["group_index", "rel_g_index"]
            for s_name, g in zip(columns,
                                 [rs[c] for c in rs.columns] + [s[c] for c in s.columns] +
                                 [g_index["group_index"], rg_index["group_index"]]):
                # Compute accuracy per group
                self.positive_rates(correct_pred, g, s_name, scn, type_, y_name, clf_name, result_type, pos_y_value=1,
                                    metric="g. Accuracy")
                # Compute base rate
                # Ignore the classifier name
                self.positive_rates(y, g, s_name, scn, type_, y_name, "BaseRate", result_type, pos_y_value=1,
                                    metric="Base Rate")
                # Compute positive rate
                self.positive_rates(pred_y, g, s_name, scn, type_, y_name, clf_name, result_type, pos_y_value=1,
                                    metric="Pos. Rate")
                # Compute tp_fp
                self.tp_and_fp(pred_y, y, g, s_name, scn, type_, y_name, clf_name, result_type, pos_y_value=1)

    def compute(self, original_train, rec_mapped_train, orig_mapped_train, original_test, rec_mapped_test,
                orig_mapped_test, train_rel_index, train_org_index, test_rel_index, test_org_index,
                rel_sens_columns_test, orig_sens_columns_test, exp_folders, mapped_decision=False):
        """ Compute fairness metrics for each scenario """
        rel_sens_columns_test.columns = "rel:" + rel_sens_columns_test.columns
        orig_sens_columns_test.columns = "org:" + orig_sens_columns_test.columns
        o_tr_x, o_tr_y, o_tr_s = self.split(original_train)
        o_ts_x, o_ts_y, o_ts_s = self.split(original_test)
        viz_data = o_ts_x
        if mapped_decision:
            viz_data = pd.concat([o_ts_x, o_ts_y], axis=1)
        self.visualise.clusters_plots(dFrame=viz_data, labels=test_org_index,
                                      savefig="{}/Original-Cluster.png".format(exp_folders.fig_dir))
        # Normally o_tr_s == m_tr_s and o_ts_s == m_ts_s == encoded(sens_columns)
        for dt, m_train, m_test in zip(["Org-Mapped", "Rec-Mapped"], [orig_mapped_train, rec_mapped_train],
                                       [orig_mapped_test, rec_mapped_test]):
            # Extract group information and y.
            m_tr_x, m_tr_y, m_tr_s = self.split(m_train)
            m_ts_x, m_ts_y, m_ts_s = self.split(m_test)
            viz_data = m_ts_x
            if mapped_decision:
                viz_data = pd.concat([m_ts_x, m_ts_y], axis=1)
            # self.visualise.clusters_plots(dFrame=viz_data, labels=test_rel_index,
            #                               savefig="{}/{}-Rel-Cluster.png".format(exp_folders.fig_dir, dt))
            self.visualise.clusters_plots(dFrame=viz_data, labels=test_org_index,
                                          savefig="{}/{}-O.Idx-Cluster.png".format(exp_folders.fig_dir, dt))
            # Data publishing: Train on modified, test on modified
            self.train_models(m_tr_x, m_tr_y)
            # Predict and compute metrics
            self.fairness_results(x=m_ts_x, y=m_ts_y, rs=rel_sens_columns_test, s=orig_sens_columns_test, g_index=test_org_index,
                                  rg_index=test_rel_index, y_name=self.decision_name, type_="-",
                                  scn="Data Pub.", result_type=dt)

            # Fair classifier: train on sanitized data and predict using original data
            # self.train_models(m_tr_x, m_tr_y)  # Models are already trained. So no need to do it again.
            # Predict and compute metrics
            self.fairness_results(x=o_ts_x, y=o_ts_y, rs=rel_sens_columns_test, s=orig_sens_columns_test, g_index=test_org_index,
                                  rg_index=test_rel_index, y_name=self.decision_name, type_="-",
                                  scn="Fair Clfs", result_type=dt)

            # State of the art: Same as data publishing the only thing is that we use the original decision
            # Train on modified with original decision. Predict using modified data with orig dec
            self.train_models(m_tr_x, o_tr_y)
            # Predict and compute metrics
            self.fairness_results(x=m_ts_x, y=o_ts_y, rs=rel_sens_columns_test, s=orig_sens_columns_test, g_index=test_org_index,
                                  rg_index=test_rel_index, y_name=self.decision_name, type_="-",
                                  scn="SOA", result_type=dt)

            # Local sanitization: train on original data and predict using sanitized data
            self.train_models(o_tr_x, o_tr_y)
            # Predict and compute metrics
            self.fairness_results(x=m_ts_x, y=o_ts_y, rs=rel_sens_columns_test, s=orig_sens_columns_test, g_index=test_org_index,
                                  rg_index=test_rel_index, y_name=self.decision_name, type_="-",
                                  scn="Local San.", result_type=dt)
            self.fairness_results(x=m_ts_x, y=m_ts_y, rs=rel_sens_columns_test, s=orig_sens_columns_test,
                                  g_index=test_org_index,
                                  rg_index=test_rel_index, y_name=self.decision_name, type_="-",
                                  scn="Local San. M.Y", result_type=dt)

        # Baseline: Train on original, test on original data
        # self.train_models(o_tr_x, o_tr_y)  # Already trained in the local san scenario.
        self.fairness_results(x=o_ts_x, y=o_ts_y, rs=rel_sens_columns_test, s=orig_sens_columns_test, g_index=test_org_index,
                              rg_index=test_rel_index, y_name=self.decision_name, type_="-", scn="Baseline",
                              result_type="Baseline")

import numpy as np
import pandas as pd
from Modules import metrics_ as m_


class AifTransMetrics(m_.TransformationMetrics):

    def soa(self, train_set, test_set, data_prep, stop_restart):
        """ Compute metrics only for the state of the art. """
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

            trans_train = self.transformation(train_set)
            trans_test = self.transformation(test_set)
            trans_all_data = pd.concat([trans_train, trans_test], axis=0, sort=True)
            trans_all_data.reset_index(drop=True, inplace=True)
            trans_majority_data = self.get_group_data(trans_all_data, self.majority_group)
            trans_minority_data = self.get_group_data(trans_all_data, self.minority_group)
            trans_not_maj_data = trans_all_data.drop(trans_majority_data.index, axis=0)
            trans_not_min_data = trans_all_data.drop(trans_minority_data.index, axis=0)
            tr_ts_majority_data = self.get_group_data(trans_test, self.majority_group)
            tr_ts_minority_data = self.get_group_data(trans_test, self.minority_group)

            self.distinguish_soa(majority_data, not_maj_data, minority_data, not_min_data,
                                 trans_majority_data, trans_not_maj_data, trans_minority_data, trans_not_min_data,
                                 train_size=self.train_size)

            self.distortions_and_control_soa(ts_majority_data, test_set.drop(ts_majority_data.index, axis=0),
                                             ts_minority_data, test_set.drop(ts_minority_data.index, axis=0),
                                             tr_ts_majority_data, trans_test.drop(tr_ts_majority_data.index, axis=0),
                                             tr_ts_minority_data, trans_test.drop(tr_ts_minority_data.index, axis=0),
                                             prefix="Test-")
            self.distortions_and_control_soa(majority_data, not_maj_data, minority_data, not_min_data,
                                             trans_majority_data, trans_not_maj_data, trans_minority_data,
                                             trans_not_min_data, prefix="All-")

            self.fairness_and_y_accuracies_soa(train_set, test_set, trans_train, trans_test, data_prep, original_y=True)

            self.dissimilarities(test_set)

            self.has_computed = True

    def map_sensitive(self):
        """ """
        mag = []
        mig = []
        for i, n in enumerate(self.control_attribute_names):
            l = {}
            for v in self.majority_group[i]:
                l.update({n: v})
            mag.append(l)
            l = {}
            for v in self.minority_group[i]:
                l.update({n: v})
            mig.append(l)

        self.majority_group = mag
        self.minority_group = mig

    def get_group_data(self, data, group):
        """ Get the data for a given group, for instance, return only the majority group or the minority one."""
        mask = []
        for gr in group:
            m = []
            for key, select_value in gr.items():
                m.append(data[key] == select_value)
            mask.append(np.logical_or.reduce(m))
        return data[np.logical_and.reduce(mask)]

    def split_x_control(self, *args):
        split = []
        for data in args:
            split.append(
                (super(AifTransMetrics, self).split_x_control(data))
            )
        return split

    def split_decision(self, *args):
        split = []
        for data in args:
            split.append(
                (super(AifTransMetrics, self).split_decision(data))
            )
        return split

    def distinguish_soa(self, majority_data, not_majority_data, minority_data, not_minority_data,
                        trans_majority_data, trans_not_majority_data, trans_minority_data, trans_not_minority_data,
                        train_size=70):
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

        trans_majority_data = trans_majority_data.reset_index(drop=True)
        trans_not_majority_data = trans_not_majority_data.reset_index(drop=True)
        trans_minority_data = trans_minority_data.reset_index(drop=True)
        trans_not_minority_data = trans_not_minority_data.reset_index(drop=True)

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

        (maj_x, maj_c), (not_maj_x, _), (min_x, min_c), (not_min_x, _), (maj_tr, _), (not_maj_tr, _), (min_tr, _), \
        (not_min_tr, _) = self.split_x_control(majority_data, not_majority_data, minority_data, not_minority_data,
                                               trans_majority_data, trans_not_majority_data, trans_minority_data,
                                               trans_not_minority_data)

        # Not min (Maj + other combinations) to minority
        _compute2_(min_x, not_min_tr, "Min-vs-N.Min Tr.", "Mixed")
        _compute2_(min_tr, not_min_tr, "Min-vs-N.Min Tr.", "Recon-All-Trans")
        _compute2_(min_x, not_min_x, "Min-vs-N.Min", "Baseline")

        # mapped to Maj
        _compute2_(maj_x, not_maj_tr, "Maj-vs-N.Maj Tr.", "Mixed")
        _compute2_(maj_tr, not_maj_tr, "Maj-vs-N.Maj Tr.", "Recon-All-Trans")
        _compute2_(maj_x, not_maj_x, "Maj-vs-N.Maj", "Baseline")

    def distortions_and_control_soa(self, majority_data, not_majority_data, minority_data, not_minority_data,
                                    trans_majority_data, trans_not_majority_data, trans_minority_data,
                                    trans_not_minority_data, prefix="Test-"):
        d_category = "{}-Data Dist.".format(prefix)
        c_category = "{}-Control P.".format(prefix)
        # Majority same code
        (maj_x, maj_c), (not_maj_x, not_maj_c), (min_x, min_c), (not_min_x, not_min_c), (maj_tr, _), (not_maj_tr, _), \
        (min_tr, _), (not_min_tr, _) = \
            self.split_x_control(majority_data, not_majority_data, minority_data, not_minority_data,
                                 trans_majority_data, trans_not_majority_data, trans_minority_data,
                                 trans_not_minority_data)

        d_maj = self.distance_mse(maj_x, maj_tr)
        self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="Maj 2 Maj Tr.", Result=d_maj,
                         Category=d_category, ResultType="Recon-All-Trans")
        # Minority same code
        d_min = self.distance_mse(min_x, min_tr)
        self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="Min 2 Min Tr.", Result=d_min,
                         Category=d_category, ResultType="Recon-All-Trans")
        # Not Minority to minority
        # Must have the least amount of modif to make a maj predicted as min
        d_maj_to_min = self.distance_mse(not_min_x, not_min_tr)
        self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="N.Min Tr.:N.Min",
                         Result=d_maj_to_min, Category=d_category, ResultType="Recon-All-Trans")
        # not Majority to Majority
        # Must have the least amount of modif to make a min predicted as maj
        d_min_to_maj = self.distance_mse(not_maj_x, not_maj_tr)
        self.add_results(Attribute="-", Metric="Distance", Classifier="-", Type="N.Maj Tr.:N.Maj",
                         Result=d_min_to_maj, Category=d_category, ResultType="Recon")
        # All originals
        data = pd.concat([maj_x, not_maj_x], axis=0, sort=True)
        data.reset_index(drop=True, inplace=True)
        data = data.sample(frac=1)
        targets_c = pd.concat([maj_c, not_maj_c], axis=0, sort=True)
        targets_c.reset_index(drop=True, inplace=True)
        targets_c = targets_c.loc[data.index]

        # All to min
        all2min = pd.concat((min_x, not_min_tr), axis=0, sort=True).reset_index(drop=True)
        a2mn_codes = pd.concat((min_c, not_min_c), axis=0).reset_index(drop=True)
        train_a2mn_x = all2min.sample(frac=1).sample(frac=self.train_size / 100)
        train_a2mn_y = a2mn_codes.loc[train_a2mn_x.index]
        test_a2mn_x = all2min.drop(train_a2mn_x.index, axis=0)
        test_a2mn_y = a2mn_codes.loc[test_a2mn_x.index]

        # All to maj
        all2maj = pd.concat((maj_x, not_maj_tr), axis=0, sort=True).reset_index(drop=True)
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
        self.control_predict_accuracy(maj_tr, maj_c_gmx, type_="Rec. Maj", category=c_category,
                                      result_type="Recon-All-Trans.")
        # original minority accuracy
        self.control_predict_accuracy(min_x, min_c_gmx, type_="Org. Min", category=c_category, result_type="Baseline")
        # Predict minority reconstructed. Expect perfect acc or same as original. Must tell that all data is min
        self.control_predict_accuracy(min_tr, min_c_gmx, type_="Rec. Min", category=c_category,
                                      result_type="Recon-All-Trans.")
        # Predict not minority transformed to min. Must tell that all data is min
        self.control_predict_accuracy(not_min_tr, np.repeat(most_disc_c_gmx, not_min_tr.shape[0], axis=0),
                                      type_="N.Min Tr.:Min", category=c_category, result_type="Recons-All-Trans")
        self.mcontrol_fit_predict_accuracy(train_a2mn_x, train_a2mn_y, test_x=test_a2mn_x, test_y=test_a2mn_y,
                                           type_="N.Min Tr.:N.Min", category=c_category, result_type="Mixed")
        # Predict not majority transformed to maj. Must tell that all data is maj
        self.control_predict_accuracy(not_maj_tr, np.repeat(sel_prv_c_gmx, not_maj_tr.shape[0], axis=0),
                                      type_="N.Maj Tr.:Maj", category=c_category, result_type="Recons-All-Trans.")
        self.mcontrol_fit_predict_accuracy(train_a2mj_x, train_a2mj_y, test_x=test_a2mj_x, test_y=test_a2mj_y,
                                           type_="N.Maj Tr.:N.Maj", category=c_category, result_type="Mixed")

    def fairness_and_y_accuracies_soa(self, train_set, test_set, trans_train, trans_test, data_prep, original_y=False):
        """ Compute fairness metrics """

        # Baseline: Fit, predict, save.
        # Split x and control attributes
        (train_set_xy, train_set_c), (test_set_xy, test_set_c), (trans_train_xy, trans_train_c), \
        (trans_test_xy, trans_test_c) = self.split_x_control(train_set, test_set, trans_train, trans_test)

        # Split x and y, fit, predict and compute metrics
        (train_set_x, train_set_y), (test_set_x, test_set_y), (trans_train_x, trans_train_y), \
        (trans_test_x, trans_test_y) = self.split_decision(train_set_xy, test_set_xy, trans_train_xy, trans_test_xy)

        # fit, predict and metrics
        self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y, x_test=test_set_x,
                                          y_test=test_set_y, control_features=test_set_c, scn="Baseline",
                                          type_="Baseline", result_type="Baseline", arg_max=False)

        # Other scn Transform: min or maj, fit, predict
        # Get Majority/Minority control code
        train_set_c_maj = self.get_group_data(train_set_c, self.majority_group)
        test_set_c_maj = self.get_group_data(test_set_c, self.majority_group)
        # Get minority and majority data: data = data_min + data_maj
        train_set_xy_maj = train_set_xy.loc[train_set_c_maj.index]
        train_set_xy_not_maj = train_set_xy.drop(train_set_c_maj.index, axis=0)
        trans_train_xy_maj = trans_train_xy.loc[train_set_c_maj.index]
        trans_train_xy_not_maj = trans_train_xy.drop(train_set_c_maj.index, axis=0)

        test_set_xy_maj = test_set_xy.loc[test_set_c_maj.index]
        test_set_xy_not_maj = test_set_xy.drop(test_set_c_maj.index, axis=0)
        trans_test_xy_maj = trans_test_xy.loc[test_set_c_maj.index]
        trans_test_xy_not_maj = trans_test_xy.drop(test_set_c_maj.index, axis=0)

        # Minority
        train_set_c_min = self.get_group_data(train_set_c, self.minority_group)
        test_set_c_min = self.get_group_data(test_set_c, self.minority_group)
        train_set_xy_min = train_set_xy.loc[train_set_c_min.index]
        train_set_xy_not_min = train_set_xy.drop(train_set_c_min.index, axis=0)
        trans_train_xy_min = trans_train_xy.loc[train_set_c_min.index]
        trans_train_xy_not_min = trans_train_xy.drop(train_set_c_min.index, axis=0)

        test_set_xy_min = test_set_xy.loc[test_set_c_min.index]
        test_set_xy_not_min = test_set_xy.drop(test_set_c_min.index, axis=0)
        trans_test_xy_min = trans_test_xy.loc[test_set_c_min.index]
        trans_test_xy_not_min = trans_test_xy.drop(test_set_c_min.index, axis=0)

        # Order matter especially for scenario involving the original decision.

        # Re-form whole dataset
        for rt, mj, mn in zip(("Mixed", "Recon-All-Trans."),
                              ({"tr": train_set_xy_maj, "ts": test_set_xy_maj},
                               {"tr": trans_train_xy_maj, "ts": trans_test_xy_maj}),
                              ({"tr": train_set_xy_min, "ts": test_set_xy_min},
                               {"tr": trans_train_xy_min, "ts": trans_test_xy_min})):
            # Majority
            train_set_xy_all_maj = pd.concat([mj["tr"], trans_train_xy_not_maj], axis=0, sort=True)
            # Set the same order as the original data
            train_set_xy_all_maj = train_set_xy_all_maj.loc[train_set_xy.index]
            # Test
            test_set_xy_all_maj = pd.concat([mj["ts"], trans_test_xy_not_maj], axis=0, sort=True)
            # Set the same order as the original data
            test_set_xy_all_maj = test_set_xy_all_maj.loc[test_set_xy.index]

            # Minority
            # Train
            train_set_xy_all_min = pd.concat([mn["tr"], trans_train_xy_not_min], axis=0, sort=True)
            # Set the same order as the original data
            train_set_xy_all_min = train_set_xy_all_min.loc[train_set_xy.index]
            # Test
            test_set_xy_all_min = pd.concat([mn["ts"], trans_test_xy_not_min], axis=0, sort=True)
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
            (train_set_x_all_maj, train_set_y_all_maj), (test_set_x_all_maj, test_set_y_all_maj) = \
                self.split_decision(train_set_xy_all_maj, test_set_xy_all_maj)
            # Scenario
            # S1: Complete modifs: train on all data modif with modif decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_maj, y_train=train_set_y_all_maj,
                                              x_test=test_set_x_all_maj, y_test=test_set_y_all_maj,
                                              control_features=test_set_c, scn="S1", type_="All Maj.", result_type=rt,
                                              arg_max=False)
            # S2: Partial modifs: train on data modif but with orig decisions.
            # If gan model is trained without decision, then we keep the decision as original. S2 == S1
            if not original_y:
                self.fairness_fit_predict_metrics(x_train=train_set_x_all_maj, y_train=train_set_y,
                                                  x_test=test_set_x_all_maj, y_test=test_set_y,
                                                  control_features=test_set_c, scn="S2", type_="All Maj.",
                                                  result_type=rt, arg_max=False)
            # S3-y: Fair Classifier: train on modif data with orig dec and test on orig data
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_maj, y_train=train_set_y,
                                              x_test=test_set_x, y_test=test_set_y,
                                              control_features=test_set_c, scn="S3-y", type_="All Maj.", result_type=rt,
                                              train_clfs=original_y, arg_max=False)
            # S3-y_hat: Fair Classifier: train modif data with modif dec and test on orig data
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_maj, y_train=train_set_y_all_maj,
                                              x_test=test_set_x, y_test=test_set_y,
                                              control_features=test_set_c, scn="S3-y_hat", type_="All Maj.",
                                              result_type=rt, arg_max=False)
            # S4-y: Local San: train on orig data and test on modified data but with orig decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y,
                                              x_test=test_set_x_all_maj, y_test=test_set_y,
                                              control_features=test_set_c, scn="S4-y", type_="All Maj.", result_type=rt,
                                              arg_max=False)
            # S4-y_hat: Local San: train on orig data and test on modified data but with modified decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y,
                                              x_test=test_set_x_all_maj, y_test=test_set_y_all_maj,
                                              control_features=test_set_c, scn="S4-y_hat", type_="All Maj.",
                                              result_type=rt, train_clfs=False, arg_max=False)

            # Mapping all to minority
            # Split decisions
            (train_set_x_all_min, train_set_y_all_min), (test_set_x_all_min, test_set_y_all_min) =\
                self.split_decision(train_set_xy_all_min, test_set_xy_all_min)
            # Scenario
            # S1: Complete modifs: train on all data modif with modif decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_min, y_train=train_set_y_all_min,
                                              x_test=test_set_x_all_min, y_test=test_set_y_all_min,
                                              control_features=test_set_c, scn="S1", type_="All Min.", result_type=rt,
                                              arg_max=False)
            # S2: Partial modifs: train on data modif but with orig decisions
            # If gan model is trained without decision, then we keep the decision as original. S2 == S1
            if not original_y:
                self.fairness_fit_predict_metrics(x_train=train_set_x_all_min, y_train=train_set_y,
                                                  x_test=test_set_x_all_min, y_test=test_set_y,
                                                  control_features=test_set_c, scn="S2", type_="All Min.",
                                                  result_type=rt, arg_max=False)
            # S3-y: Fair Classifier: train on modif data with orig dec and test on orig data
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_min, y_train=train_set_y,
                                              x_test=test_set_x, y_test=test_set_y,
                                              control_features=test_set_c, scn="S3-y", type_="All Min.", result_type=rt,
                                              train_clfs=original_y, arg_max=False)
            # S3-y_hat: Fair Classifier: train modif data with modif dec and test on orig data
            self.fairness_fit_predict_metrics(x_train=train_set_x_all_min, y_train=train_set_y_all_min,
                                              x_test=test_set_x, y_test=test_set_y,
                                              control_features=test_set_c, scn="S3-y_hat", type_="All Min.",
                                              result_type=rt, arg_max=False)
            # S4-y: Local San: train on orig data and test on modified data but with orig decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y,
                                              x_test=test_set_x_all_min, y_test=test_set_y,
                                              control_features=test_set_c, scn="S4-y", type_="All Min.", result_type=rt,
                                              arg_max=False)
            # S4-y_hat: Local San: train on orig data and test on modified data but with modified decisions
            self.fairness_fit_predict_metrics(x_train=train_set_x, y_train=train_set_y,
                                              x_test=test_set_x_all_min, y_test=test_set_y_all_min,
                                              control_features=test_set_c, scn="S4-y_hat", type_="All Min.",
                                              result_type=rt, train_clfs=original_y, arg_max=False)

    def mapping(self, test_set, *args, **kwargs):
        """ Map data and return all """
        # 1- Remove the data code
        trans_test = self.transformation(test_set)
        (test_set_x, test_set_c), (trans_test_x, trans_test_c) = self.split_x_control(test_set, trans_test)
        # 2- Split into different groups: Majority vs not majority, minority vs not minority
        maj_c_indexes = self.get_group_data(test_set_c, self.majority_group)
        t_maj_x, t_not_maj_x = self.extract_group(test_set_x, maj_c_indexes)
        trans_t_maj_x, trans_t_not_maj_x = self.extract_group(trans_test_x, maj_c_indexes)

        min_c_indexes = self.get_group_data(test_set_c, self.minority_group)
        t_min_x, t_not_min_x = self.extract_group(test_set_x, min_c_indexes)
        trans_t_min_x, trans_t_not_min_x = self.extract_group(trans_test_x, min_c_indexes)

        return maj_c_indexes, t_maj_x, t_not_maj_x, trans_t_maj_x, trans_t_not_maj_x, \
               min_c_indexes, t_min_x, t_not_min_x, trans_t_min_x, trans_t_not_min_x

    def _prepare_targets_multi_outputs_(self, target_names, *targets):
        """ Format the control data for the multioutput prediction """
        t_columns = []
        targets_gmx = [[] for i in range(len(targets))]
        for c in target_names:
            ext_cls = self.__filter_features__(targets[0], c, self.prefix_sep)
            t_columns.append(ext_cls)
            for i in range(len(targets)):
                targets_gmx[i].append(targets[i][ext_cls].values.squeeze())

        targets_gmx = [np.array(gmx).T for gmx in targets_gmx]

        return t_columns, targets_gmx

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
                col = cols.values.squeeze()
                v_name += "{}-".format(col.tolist())
            v_name = v_name[:-1]
            mapping.update({g: v_name})
        a_name = "-".join(control_attribute_names)
        return groups, mapping, a_name

import glob
import json
import os
import shutil

import numpy as np
import pandas as pd


def try_reading_dataframe(set_path):
    """
    Read the given path, and make sure that the dataset has at least some rows in it.
    :param set_path: path to the dataset
    :return: True if can read, False if not
    """
    try:
        assert (pd.read_csv(set_path).size != 0), "DataFrame is empty"
        return True
    except (FileNotFoundError, pd.errors.EmptyDataError, AssertionError) as e:
        return False


def pareto_front_from_dataframe(df, metrics_mode):
    """
    Get the non dominated front from the given dataframe.
    Args:
        df: dataframe containing columns the result
        metrics_mode: Metrics to consider for the pareto front, with their respective mode.
         E.g: accuracy:max, runtime:min

    Returns:
        Dataframe containing the pareto front. Same format as input dataframe.
    """

    def apply_mode_to_metric(reference, row, mode):
        """
        Apply the mode of the considered metric to the given row.
        For instance, if mode is min, find all datapoints in row that are less than the given reference.
        Args:
            reference: reference point
            row: row on which to apply the mode
            mode: the mode to apply

        Returns:
            Mask containing all datapoints in row that are less or greater (depending on the mode)
             than the given reference.
        """
        if mode == "min":
            return row < reference
        if mode == "max":
            return row > reference

    def apply_reverse_mode_to_metric(reference, row, mode):
        """
        Apply the opposite of mode of the considered metric to the given row.
        For instance, if mode is min, find all datapoints in row that are greater than the given reference.
        Args:
            reference: reference point
            row: row on which to apply the mode
            mode: the mode to apply

        Returns:
            Mask containing all datapoints in row that are less or greater (depending on the mode)
             than the given reference.
        """
        if mode == "min":
            return row > reference
        if mode == "max":
            return row < reference

    metrics = list(metrics_mode.keys())
    results = df[metrics]

    stop_loop = False
    # Will contains indexes of non dominated front. Mark all as dominated to start with.
    non_dominated = results[metrics[0]] != results[metrics[0]]
    # Mask to indicate whether we have already processed the given row. So it will either be selected for
    # verification or we already know that it is dominated.
    checked_indexes = results[metrics[0]] != results[metrics[0]]

    while not stop_loop:
        # start with the first index, and continue until you find a non dominated index
        # Get all result rows we have not work with, or we do not know that they are dominated.
        current_df = results.loc[~checked_indexes]
        found_non_dominated = False
        for index in current_df.index:
            # Mark the current index as processed
            checked_indexes.loc[index] = True
            # Get the row of the current index to extract all metrics
            current_row = current_df.loc[index]
            # Excluding the current datapoint (which act as a reference), for each metric, build a mask indicating
            # whether we found a better minimum or maximum than the current reference.
            # If mode is min, for metric m1 find all rows that have values lower than reference.
            # If mode is max, for metric m2 find all rows that have values higher than reference.
            mask = [apply_mode_to_metric(current_row[m], current_df.loc[current_df.index != index][m], md)
                    for m, md in metrics_mode.items()]
            # Merge all mask using and. If there is a row that has True, it means that we found other datapoints that
            # have both a lower value for m1 and a higher value for m2 than the reference (current) row.
            # Current row is dominated.
            mask = np.logical_and.reduce(mask)
            # Check if there are any better solution.
            if not mask.any():
                # There are no better solution. Current row belong to the pareto front
                # Set the current index as not dominated.
                non_dominated.loc[index] = True
                found_non_dominated = True
                # Stop the loop for the current index so that we can remove all solutions dominated by the current one.
                break
        # Find results dominated by current solution
        if found_non_dominated:
            # The row we are analysing is an ndf solution. Mark all rows that are dominated by this solution as
            # processed
            # If mode is min, for metric m1 find all rows that have values higher than reference.
            # If mode is max, for metric m2 find all rows that have values lower than reference.
            mask = [apply_reverse_mode_to_metric(current_row[m], current_df[m], md) for m, md in metrics_mode.items()]
            mask = np.logical_and.reduce(mask)
            # Get all solutions that have both mask for m1 and m2 set to True. The solution is dominated.
            dominated_indexes = current_df.loc[mask].index
            # Mark the indexes as already checked.
            checked_indexes.loc[dominated_indexes] = True

        if checked_indexes.all():
            # We have checked all indexes, stop the loop
            stop_loop = True

    # Return data with the non dominated solutions
    return df.loc[non_dominated]


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


def ray_post_processing(pareto_front, transfer_path, experiment_path, max_epoch_keyword="epochs"):
    """
    Post process the results computed with Ray tune. Move checkpoints corresponding to the pareto front to final
     result directory.
     Update the config such that the max epoch correspond to the training iteration
    Args:
        pareto_front: data points corresponding to the pareto front
        transfer_path: location where to move the checkpoints
        experiment_path: ray tune experiment path
        max_epoch_keyword: keyword for identifying max epoch in the config file

    Returns:
        path list of the moved checkpoints to run test computations.
    """
    make_directory(transfer_path)

    paths_configs = {}
    for i in range(0, len(pareto_front)):
        checkpoint = pareto_front.iloc[i]
        # Get all directory containing the trial_id
        trial_path = glob.glob('{}/*{}*'.format(experiment_path, checkpoint["trial_id"]))[0]
        # Create the destination path
        path = "{}/{}".format(transfer_path, checkpoint["trial_id"])
        # Add the destination path to the list
        make_directory(path)

        # Changing the training epoch value
        with open(f"{trial_path}/params.json") as f:
            parameters = json.load(f)
        parameters[max_epoch_keyword] = int(checkpoint['training_iteration'])
        with open(f"{path}/params.json", 'w') as f:
            json.dump(parameters, f)

        # Move checkpoint
        checkpoint_path = glob.glob('{}/*{}*'.format(trial_path, checkpoint["checkpoint_dir_name"]))
        if len(checkpoint_path):
            checkpoint_path = checkpoint_path[0]
            file_names = os.listdir(checkpoint_path)

            for file_name in file_names:
                shutil.move(os.path.join(checkpoint_path, file_name), path)

            # Link path and respective configurations together.
            paths_configs.update({path: parameters})

    return paths_configs

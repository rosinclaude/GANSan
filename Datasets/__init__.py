from copy import deepcopy

import yaml


# Load YAML
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Merge default parameters with specific dataset config
def get_dataset_config(dataset_name, config):
    """ Read dataset config and merge it with provided config """
    default_params = config['default_params']
    dataset_config = config['datasets'][dataset_name]

    # Start with a deep copy of default_params to avoid mutation
    merged_config = deepcopy(default_params)

    # Update with dataset-specific values
    merged_config.update(dataset_config)

    return merged_config


# Function to call outside the init file
def get_config(dataset_name, config_file="Datasets/datasets.yaml"):
    """ Get the dataset config from a yaml file, merge it with defaults """
    config = load_yaml(config_file)
    return get_dataset_config(dataset_name, config)

# # Example usage
# config = load_yaml('your_file.yaml')
#
# # Access merged configuration for 'Adult'
# adult_config = get_dataset_config('Adult', config)
# print(adult_config['PosOutcomes'])  # Access the PosOutcomes
#
# # Access 'Note1' which is a copy of 'Note'
# note1_config = get_dataset_config('Note1', config)
# print(note1_config['GroupAttributes'])  # Access GroupAttributes from Note1

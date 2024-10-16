import copy

# Dataset to work on. Some of the attributes of these datasets are ignored by some approaches.
# Set the boolean corresponding to the dataset to use to True

# Variable RecommendedEpochs represent the number of epochs necessary for an AE with trois linear layers of sizes
# input, 22, 12288, 22, input, trained with 2e-4 and stepLR scheduler of epoch 150 and gamma 0.1,
# to achieve the lowest error of reconstruction on this set.

# Adult
Adult = \
    {
        "SetName": "AdultTotalCNoNan",
        "LabelName": "income",  # Decision attribute
        # pos_outcomes = [1] # Positive outcome value
        "PosOutcomes": [">50K", ">50K."],
        "GroupAttributes": ["sex"],  # Protected or sensitive attribute
        # default_group_value = [[1]] # or privileged group value or privileged_classes.
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [['Male']],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [['Female']],
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 1024,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }

Adult15K = copy.deepcopy(Adult)
Adult15K.update({"SetName": "AdultTotalCNoNan15K"})

Adult30K = copy.deepcopy(Adult)
Adult30K.update({"SetName": "AdultTotalCNoNan15K"})

# German
German = \
    {
        # ProportionsSensitive = [19, 32, 50]
        "SetName": "german_data_3_2_pr",
        "LabelName": "good_3_bad_2_customer",  # Decision attribute
        "PosOutcomes": [3],
        "GroupAttributes": ["Age_in_years"],  # Protected or sensitive attribute
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [[1]],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [[0]],
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 128,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }
# Default of credit card
Taiwan = \
    {
        "SetName": "default_of_credit_card_clients",
        "LabelName": "default_payment_next_month",  # Decision attribute
        "PosOutcomes": [0],  # Positive outcome value  ## 0 Mean no default payment
        "GroupAttributes": ["SEX"],  # Protected or sensitive attribute
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [[0]],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [[1]],  ## Female is 0, Male is 1: 60% Female, 40% Male, hence less represented is Male
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 128,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }

# Ricci
Ricci = \
    {
        "SetName": "RicciData_2",
        "LabelName": "is_promoted",  # Decision attribute
        "PosOutcomes": [1.0],
        "GroupAttributes": ["Race"],  # Protected or sensitive attribute
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [[1]],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [[0]],
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 128,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }

# lipton
Lipton = \
    {
        "SetName": "lipton",
        "LabelName": "Y",  # Decision attribute
        "PosOutcomes": [1.0],
        "GroupAttributes": ["gender"],  # Protected or sensitive attribute
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [[1]],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [[0]],
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 128,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }

# NoteData
Note = \
    {
        "SetName": "NoteData",
        "LabelName": "noteDec",  # Decision attribute
        "PosOutcomes": [1.0],
        "GroupAttributes": ["gender"],  # Protected or sensitive attribute
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [[1]],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [[0]],
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 128,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }
Note1 = Note.copy()
Note1.update({"SetName": "NoteData_1_of_3_common"})
Note2 = Note.copy()
Note2.update({"SetName": "NoteData_2_of_3_common"})
NoteNoDisc = Note.copy()
NoteNoDisc.update({"SetName": "NotesGroupTranslatedNoDisc"})
NoteNoTransNoDisc = Note.copy()
NoteNoTransNoDisc.update({"SetName": "NotesNoGroupTranslationNoDisc"})

# CategoricalOrdinal uniform
CatOrd = \
    {
        "SetName": "CategoricalOrdinalUniform",
        "LabelName": "Decision",  # Decision attribute
        "PosOutcomes": [1],
        "GroupAttributes": ["Gender"],  # Protected or sensitive attribute
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [[1]],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [[0]],
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 128,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }

# Propublica datasets
PropRecid_Sex = \
    {
        "SetName": "propublica-recidivism_categorical-binsensitive",
        "LabelName": "two_year_recid",  # Decision attribute
        "PosOutcomes": [1],
        "GroupAttributes": ["sex"],  # Protected or sensitive attribute
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [[1]],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [[0]],
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 128,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }

PropRecidNoACNoSR_Sex = PropRecid_Sex.copy()
PropRecidNoACNoSR_Sex.update({"SetName": "propublica-recidivism_categorical-binsensitive-no-age_cat_sex_race"})

PropRecid_Race = \
    {
        "SetName": "propublica-recidivism_categorical-binsensitive",
        "LabelName": "two_year_recid",  # Decision attribute
        "PosOutcomes": [1],
        "GroupAttributes": ["race"],  # Protected or sensitive attribute
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [[1]],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [[0]],
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 128,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }
PropRecidNoACNoSR_Race = PropRecid_Race.copy()
PropRecidNoACNoSR_Race.update({"SetName": "propublica-recidivism_categorical-binsensitive-no-age_cat_sex_race"})

PropRecid2 = PropRecid_Sex.copy()
PropRecid2.update({
    "GroupAttributes": ["sex", "race"],  # Protected or sensitive attribute
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [[1], [1]],
    # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
    "ProtectedGroupValue": [[0], [0]],
})

PropRecidNoACNoSR2 = PropRecid2.copy()
PropRecidNoACNoSR2.update({
    "SetName": "propublica-recidivism_categorical-binsensitive-no-age_cat_sex_race"
})

PropViolentRecid_Sex = PropRecid_Sex.copy()
PropViolentRecid_Sex.update({"SetName": "propublica-violent-recidivism_categorical-binsensitive"})

PropViolentRecidNoACNoSR_Sex = PropRecid_Sex.copy()
PropViolentRecidNoACNoSR_Sex.update({"SetName": "propublica-violent-recidivism_categorical-binsensitive-no-age_cat_sex_race"})

PropViolentRecid_Race = PropRecid_Race.copy()
PropViolentRecid_Race.update({"SetName": "propublica-violent-recidivism_categorical-binsensitive"})

PropViolentRecidNoACNoSR_Race = PropRecid_Race.copy()
PropViolentRecidNoACNoSR_Race.update({"SetName": "propublica-violent-recidivism_categorical-binsensitive-no-age_cat_sex_race"})

PropViolentRecid2 = PropRecid2.copy()
PropViolentRecid2.update({"SetName": "propublica-violent-recidivism_categorical-binsensitive"})

PropViolentRecidNoACNoSR2 = PropRecid2.copy()
PropViolentRecidNoACNoSR2.update({"SetName": "propublica-violent-recidivism_categorical-binsensitive-no-age_cat_sex_race"})


CompasGrouped = \
    {
        "SetName": "compas_grouped",
        "LabelName": "two_year_recid",  # Decision attribute
        "PosOutcomes": ["Yes"],
        "GroupAttributes": ["race"],  # Protected or sensitive attribute
        # The order must be the same as given in the sensitive attribute
        "DefaultGroupValue": [["Caucasian"]],
        # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
        "ProtectedGroupValue": [["African-American"]],
        # instance_weights_name: Column containing instances weights
        # instance_weights_name = ""
        # Threshold below which a numerical column is considered as categorical.
        "NumericAsCategoric": 5,
        # Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
        "PreprocessingExcluded": None,
        # Include overwrite excluded. Set to None or Empty list to avoid overwriting
        "PreprocessingIncluded": None,
        # Preprocessing lower bound of values
        "Scale": 0,
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,

        ## Fairmapping parameters
        "epochs": dict(encoders=900, c_models=500, gan=2000),
        "schedulers_gamma": dict(encoders=0.1, c_models=0.1, gan=0.2),
        "batch_size": 128,
        "weights": {"reconstruction": 0.95, "classification": 0.75},
        # Number of epoch necessary for a a good reconstruction quality
        "RecommendedEpochs": 450,
    }

# 2 protected attributes
# Adult 2 protected attributes.
Adult2 = copy.deepcopy(Adult)
Adult2.update({
    "GroupAttributes": ["sex", "race"],  # Protected or sensitive attribute
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [["Male"], ["White"]],
    # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
    "ProtectedGroupValue": [["Female"], ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]],
})
Adult15K2 = copy.deepcopy(Adult2)
Adult15K2.update({"SetName": "AdultTotalCNoNan15K"})
Adult30K2 = copy.deepcopy(Adult2)
Adult30K2.update({"SetName": "AdultTotalCNoNan15K"})
# German 2 protected attributes.
German2 = copy.deepcopy(German)
German2.update({
    "GroupAttributes": ["Age_in_years", "Personal_status_and_sex"],  # Protected or sensitive attribute
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [[1], ["A91", "A93", "A94"]],
    # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
    # "ProtectedGroupValue": [[0], ["A92", "A95"]],
    "ProtectedGroupValue": [[0], ["A92"]],
})

# Compas Grouped 2 protected attributes.
CompasGrouped2 = copy.deepcopy(CompasGrouped)
CompasGrouped2.update({
    "GroupAttributes": ["race", "gender"],  # Protected or sensitive attribute
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [["Caucasian"], ["Male"]],
    # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
    # "ProtectedGroupValue": [[0], ["A92", "A95"]],
    "ProtectedGroupValue": [["African-American"], ["Female"]],
})

# # 3 Protected attributes.
# Adult: Relationship added: Husband vs the Rest
Adult3 = copy.deepcopy(Adult2)
Adult3.update({
    "GroupAttributes": ["sex", "race", "relationship"],  # Protected or sensitive attribute
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [["Male"], ["White"], ["Husband"]],
    "ProtectedGroupValue": [["Female"], ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                            ["Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"]],
})

German3 = copy.deepcopy(German)
German3.update({
    "GroupAttributes": ["Age_in_years", "Personal_status_and_sex", "Telephone"],  # Protected or sensitive attribute
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [[1], ["A91", "A93", "A94"], [1]],
    "ProtectedGroupValue": [[0], ["A92"], [0]],
})

# Compas Grouped.
CompasGrouped3 = copy.deepcopy(CompasGrouped2)
CompasGrouped3.update({
    "GroupAttributes": ["race", "gender", "age"],  # Protected or sensitive attribute
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [["Caucasian"], ["Male"], [">45", "23-25", "21-22", "18-20"]],
    # Only used by sampling methods for contexts creation. But only written in same format as Default for the uniformity sake
    # "ProtectedGroupValue": [[0], ["A92", "A95"]],
    "ProtectedGroupValue": [["African-American"], ["Female"], ["26-45"]],
})

# # 4 Protected attributes.
# Adult
# Married and still in good conditions vs people living alone.
Adult4 = copy.deepcopy(Adult)
Adult4.update({
    "GroupAttributes": ["sex", "race", "marital-status", "relationship"],  # Protected or sensitive attribute
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [["Male"], ["White"], ["Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"],
                          ["Husband"], ],
    "ProtectedGroupValue": [["Female"], ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                            ["Never-Married", "Divorced", "Separated", "Widowed"],
                            ["Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"]],
})

German4 = copy.deepcopy(German)
German4.update({
    "GroupAttributes": ["Age_in_years", "Personal_status_and_sex", "Telephone", "foreign_worker"],
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [[1], ["A91", "A93", "A94"], [1], ["A201"]],
    "ProtectedGroupValue": [[0], ["A92"], [0], ["A202"]],
})

# # 5 Protected attributes.
# Adult
Adult5 = copy.deepcopy(Adult)
Adult5.update({
    "GroupAttributes": ["sex", "race", "marital-status", "relationship", "workclass"],  # Protected or sensitive attribute
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [["Male"], ["White"], ["Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"],
                          ["Husband"], ["Private"]],
    "ProtectedGroupValue": [["Female"], ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                            ["Never-Married", "Divorced", "Separated", "Widowed"],
                            ["Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"],
                            ["Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc", "Federal-gov", "Without-pay"]
                            ],
})

German5 = copy.deepcopy(German)
German5.update({
    "GroupAttributes": ["Age_in_years", "Personal_status_and_sex", "Telephone", "foreign_worker", "Housing"],
    # The order must be the same as given in the sensitive attribute
    "DefaultGroupValue": [[1], ["A91", "A93", "A94"], [1], ["A201"], ["A152"]],
    "ProtectedGroupValue": [[0], ["A92"], [0], ["A202"], ["A151", "A153"]],
})
# Attributes below maybe unused or similar to most datasets ?
# List of categorical or function to select the categorical attributes
CategoricalFeatures = lambda df: df.select_dtypes(exclude=["int", 'float', 'double']).columns.tolist()
FeaturesToKeep = []  # Leave it empty if want to keep all
# features_to_keep = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
# features_to_drop = [] # Add features to drop
FeaturesToDrop = []
# na_values = "?"# from pd.read_csv -- na_values : scalar, str, list-like, or dict, optional
Metadata = None  # metadata to add

Datasets = \
    {


        "Adult": Adult,
        "Adult15K": Adult15K,
        "Adult30K": Adult30K,
        "Adult2": Adult2,
        "Adult15K2": Adult15K2,
        "Adult30K2": Adult30K2,
        "Adult3": Adult3,
        "Adult4": Adult4,
        "Adult5": Adult5,
        "Lipton": Lipton,
        "German": German,
        "German2": German2,
        "German3": German3,
        "German4": German4,
        "German5": German5,
        "Taiwan": Taiwan,
        "Ricci": Ricci,
        "Note": Note,
        "Note1": Note1,
        "Note2": Note2,
        "NoteNoDisc": NoteNoDisc,
        "NoteNoTransNoDisc": NoteNoTransNoDisc,
        "CatOrd": CatOrd,

        "PropRecid_Sex": PropRecid_Sex,
        "PropRecidNoACNoSR_Sex": PropRecidNoACNoSR_Sex,
        "PropRecid_Race": PropRecid_Race,
        "PropRecidNoACNoSR_Race": PropRecidNoACNoSR_Race,
        "PropRecid2": PropRecid2,
        "PropRecidNoACNoSR2": PropRecidNoACNoSR2,

        "PropViolentRecid_Sex": PropViolentRecid_Sex,
        "PropViolentRecidNoACNoSR_Sex": PropViolentRecidNoACNoSR_Sex,
        "PropViolentRecid_Race": PropViolentRecid_Race,
        "PropViolentRecidNoACNoSR_Race": PropViolentRecidNoACNoSR_Race,
        "PropViolentRecid2": PropViolentRecid2,
        "PropViolentRecidNoACNoSR2": PropViolentRecidNoACNoSR2,

        "CompasGrouped": CompasGrouped,
        "CompasGrouped2": CompasGrouped2,
        "CompasGrouped3": CompasGrouped3,
    }


import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import shap
import warnings
import logging
import matplotlib.pyplot as plt
import os
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils import Utils

from logging_config import get_logger
logger = get_logger(__name__)

# Load config.yaml
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

main_config = config["main"]
explainer_config = config["explainer"]
treatments = main_config["treatments"]["names"]
dataset_path = main_config["data"]["dataset_path"]
results_dir = main_config["data"]["results_dir"]
results_file = main_config["data"]["results_file"]

colors_guidline = main_config["plots"]["colors_guidline"]
colors_fidelity = main_config["plots"]["colors_fidelity"]
colors_stability = main_config["plots"]["colors_stability"]
filenames = main_config["plots"]["filenames"]

def main():
    # initializing utils class
    ut = Utils(dataset_path)
    warnings.filterwarnings('ignore')
    
    data, X, Y, classes_names = ut.data_read_function()
    Treatments = treatments

    guidline_Plot_Data = {}
    fidelity_Plot_Data = {}
    stability_Plot_Data = {}

    for treatment_name in Treatments:
        guidline_Plot_Data = ut.Lime_Shap_guidlineComparison(X, Y, classes_names, guidline_Plot_Data, treatment_name)
        fidelity_Plot_Data = ut.Lime_Shap_fidelity(X, Y, classes_names, fidelity_Plot_Data, treatment_name)
        stability_Plot_Data = ut.Lime_Shap_stability(X, Y, classes_names, stability_Plot_Data, treatment_name)

    ut.Plot_the_data(guidline_Plot_Data, colors_guidline, filename=filenames["guidline"])
    ut.Plot_the_data(fidelity_Plot_Data, colors_fidelity, filename=filenames["fidelity"])
    ut.Plot_the_data(stability_Plot_Data, colors_stability, filename=filenames["stability"])

    # Combine into a single dictionary
    results = {
        "guidline": guidline_Plot_Data,
        "fidelity": fidelity_Plot_Data,
        "stability": stability_Plot_Data
    }

    # Convert to DataFrame
    results_df = ut.build_results_dataframe(guidline_Plot_Data, fidelity_Plot_Data, stability_Plot_Data)

    # Save the results in the result directory
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, results_file)
    results_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    
    logger.info("Main execution started.")
    main()
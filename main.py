
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils import Utils

def main():
    ut = Utils(dataset_path="./data/NKR_IKNL_breast_syntheticdata.csv")
    warnings.filterwarnings('ignore')
    data, X, Y, classes_names = ut.data_read_function( )
    Treatments=['chemo','target','hormonal','radio','surgery']

    guidline_Plot_Data={}
    fidelity_Plot_Data={}
    stability_Plot_Data = {}

    for treatment_name in Treatments :
        guidline_Plot_Data= ut.Lime_Shap_guidlineComparison(X, Y, classes_names, guidline_Plot_Data, treatment_name)
        fidelity_Plot_Data= ut.Lime_Shap_fidelity(X, Y, classes_names, fidelity_Plot_Data, treatment_name)
        stability_Plot_Data= ut.Lime_Shap_stability(X, Y,classes_names, stability_Plot_Data, treatment_name)

    colors_guidline = ['darkgoldenrod', 'brown'] #plot colors for shap and Lime
    colors_fidelity = ['gray', 'brown'] #plot colors for shap and Lime
    colors_stability = ['skyblue', 'orange'] # Bar colors in plot

    ut.Plot_the_data(guidline_Plot_Data, colors_guidline, filename = 'guidline_comparison_plot')
    ut.Plot_the_data(fidelity_Plot_Data, colors_fidelity, filename = 'fidelity_comparison_plot')
    ut.Plot_the_data(stability_Plot_Data, colors_stability, filename = 'stability_comparison_plot')

    # Combine into a single dictionary
    results = {
        "guidline": guidline_Plot_Data,
        "fidelity": fidelity_Plot_Data,
        "stability": stability_Plot_Data
    }

    # Convert to DataFrame
    results_df = pd.DataFrame({k: pd.Series(v) for k, v in results.items()})
    #save the results in the result directory
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("Result_Data.csv", index=False)




if __name__ == "__main__":
    # code to run when the file is executed directly
    main()
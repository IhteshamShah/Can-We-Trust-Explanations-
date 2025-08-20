
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

from logging_config import get_logger
logger = get_logger(__name__)


def main():
    '''
    It create the results directory and stores:
     (1) the comparison plots of fidelity, stability and comparison with the medical guidlines over LIME and SHAP
     (2) save the results in Result_Data.csv in the foloowing format

                    treatment	model	guidline	fidelity	stability
                        chemo	shap	[0.8, 0.7]	[0.75, 0.8]	[0.6, 0.65]
                        chemo	lime	[0.9, 0.85]	[0.88, 0.9]	[0.7, 0.72]


    which can be accessable late using below commands
    

               
                print(df_results[df_results["treatment"] == "chemo"])  # Access all rows for one treatment, e.g. chemo
                print(df_results[df_results["model"] == "shap"]). # Access shap rows only

                    # Access a single cell, e.g. shap-guidline values for chemo
                    value = df_results[
                        (df_results["treatment"] == "chemo") & 
                        (df_results["model"] == "shap")] ["guidline"].values[0]

    '''
   
   
    ut = Utils(dataset_path="../data/NKR_IKNL_breast_syntheticdata.csv")
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
    results_df =  ut.build_results_dataframe(guidline_Plot_Data, fidelity_Plot_Data, stability_Plot_Data)

    #save the results in the result directory
    os.makedirs("results", exist_ok=True)
    file_path = os.path.join("results", "Result_Data.csv")
    results_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    # code to run when the file is executed directly
    logger.info("Main execution started.")
    main()
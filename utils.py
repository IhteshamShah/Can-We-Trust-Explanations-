import pandas as pd
import numpy as np
import shap
import warnings
from lime.lime_tabular import LimeTabularExplainer
import logging
import matplotlib.pyplot as plt
import os

# Configure logger
logging.basicConfig(
    filename="utils.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Utils:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        logging.info(f"Utils initialized with dataset path: {dataset_path}")

    def data_read_function(self):
        try:
            all_data = pd.read_csv(self.dataset_path)
            DataSet = all_data[['leeft', 'gesl', 'tumsoort', 'diag_basis', 'topo', 'topo_sublok', 'later', 'morf',
                                'ct', 'cn', 'cm', 'cstadium', 'er_stat', 'pr_stat', 'her2_stat', 'dcis_comp', 
                                'multifoc', 'chemo', 'target', 'horm', 'rt', 'meta_chir']]
            DataSet.columns = ['Age_at_incidence_date', 'Sex', 'Tumor_type', 'Basis_for_diagnosis', 
                               'Topography_excluding', 'Topography_including', 'Lateralization', 'Morphology',
                               'cT_TNM', 'cN_TNM', 'cm_TNM', 'Stage_based_on_cTNM', 'Er_status', 'Pr_status', 
                               'HER2_state', 'DCIS_component', 'Tumor_multifocality', 'chemo', 'target', 
                               'hormonal', 'radio', 'surgery']
            DataSet.dropna(inplace=True)
            data = pd.get_dummies(DataSet, drop_first=False)
            data.reset_index(drop=True, inplace=True)
            X = data.loc[:, ~data.columns.isin(['chemo', 'target', 'hormonal', 'radio', 'surgery'])]
            Y = data[['chemo', 'target', 'hormonal', 'radio', 'surgery']]
            classes_names = {
                'chemo': ['chemo_0', 'chemo_presurgical only', 'chemo_post-surgical only', 
                          'chemo_pre and post surgical', 'chemo_Yes, no surgery'],
                'target': ['target_0', 'target_presurgical only', 'target_post-surgical only', 
                           'target_pre and post surgical', 'target_Yes, no surgery'],
                'hormonal': ['hormonal_0', 'hormonal_presurgical only', 'hormonal_post-surgical only', 
                             'hormonal_pre and post surgical', 'hormonal_Yes, no surgery'],
                'radio': ['radio_0', 'radio_presurgical only', 'radio_post-surgical only', 'radio_Yes, no surgery'],
                'surgery': ['No_surgery', 'surgery']
            }
            logging.info("Data successfully read and processed.")
            return data, X, Y, classes_names
        except Exception as e:
            logging.error(f"Error reading dataset: {e}")
            raise

    def computing_j(self, rf, x_test, y_test):
        try:
            J = []
            for j in range(500):
                data_row = x_test.iloc[[j], :]
                sample = data_row.values.reshape(1, -1)
                if rf.predict(sample)[0] == 1 and y_test[j] == 1:
                    J.append(j)
            logging.info(f"Computed indices: {J}")
            return J
        except Exception as e:
            logging.error(f"Error in computing_j: {e}")
            raise

    def Guidelines(self):
        medical_guidelines = {
            'cm_TNM': 'High', 'cN_TNM_1': 'High', 'cN_TNM_0': 'Low',
            'cT_TNM_1': 'High', 'cT_TNM_1A': 'High', 'cT_TNM_1B': 'High',
            'cT_TNM_1C': 'High', 'cT_TNM_1M': 'High', 'cT_TNM_2': 'High',
            'cT_TNM_3': 'High', 'cT_TNM_4A': 'High', 'cT_TNM_4B': 'High', 
            'cT_TNM_4C': 'High', 'cT_TNM_4D': 'High', 'cT_TNM_IS': 'Low',
            'cN_TNM_2A': 'High', 'cN_TNM_2B': 'High', 'cN_TNM_3A': 'High',
            'Er_status': 'High', 'Pr_status': 'High',
            'Stage_based_on_cTNM_3A': 'High', 'Stage_based_on_cTNM_3B': 'High', 
            'Stage_based_on_cTNM_3C': 'High', 'Stage_based_on_cTNM_4': 'High',
            'Morphology': 'High', 'DCIS_component': 'Low'
        }
        logging.info("Medical guidelines returned.")
        return medical_guidelines

    def map_importance(self, feature, importance_list):
        for item in importance_list:
            if item[0] == feature:
                return item[1]
        return 0

    def compare_with_guidelines(self, important_features, guidelines):
        comparison = [(feature, self.map_importance(feature, important_features), guidelines[feature]) 
                      for feature in guidelines]
        logging.info(f"Comparison with guidelines: {comparison}")
        return comparison

    def calculate_concordance(self, comparison):
        concordant_pairs = 0
        for feature, model_importance, guideline_importance in comparison:
            if (model_importance > 0 and guideline_importance == 'High') or \
               (model_importance == 0 and guideline_importance == 'Low'):
                concordant_pairs += 1
        concordance = concordant_pairs / len(comparison)
        logging.info(f"Concordance calculated: {concordance}")
        return concordance
    


    def Plot_the_data(self, Plot_Data, Colors, save_path=None):
        """
        Plot SHAP and LIME boxplots side-by-side for given features.
        
        Parameters:
        -----------
        Plot_Data : dict
            Keys are feature names, values are tuples/lists: (shap_values_list, lime_values_list)
        Colors : tuple/list
            Colors for SHAP and LIME plots.
        save_path : str
            File path to save the plot (optional). If None, only shows plot.
        """
        try:
                    # Data
            data = Plot_Data
            # Create the figure and axes
            fig, ax = plt.subplots(figsize=(16, 8))

            # Labels and positions
            labels = list(data.keys())
            positions = np.array(range(len(labels))) * 2.0

            # Colors for SHAP and LIME
            colors = Colors

            # Plot each boxplot
            for i, (label, values) in enumerate(data.items()):
                bp_shap = ax.boxplot(values[0], positions=[positions[i] - 0.2], widths=0.4, patch_artist=True, boxprops=dict(facecolor=colors[0]))
                bp_lime = ax.boxplot(values[1], positions=[positions[i] + 0.25], widths=0.4, patch_artist=True, boxprops=dict(facecolor=colors[1]))

            # Adding legend
            ax.legend([bp_shap["boxes"][0], bp_lime["boxes"][0]], ['SHAP', 'LIME'], loc='upper right')

            # Set the axes ticks and labels
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)

            # Adding grid
            ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.5)
            ax.set_axisbelow(True)

            # Adding X and Y axis labels
            ax.set_xlabel(' ')
            ax.set_ylabel('Values')
            # Save plot if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight")
                logging.info(f"Plot saved to {save_path}")

            plt.show()

        except Exception as e:
            logging.error(f"Error in Plot_the_data: {e}")
            raise

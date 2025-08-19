import numpy as np
import shap
import warnings
from lime.lime_tabular import LimeTabularExplainer
import logging

# Configure logger
logging.basicConfig(
    filename="explainers.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Explainers:
    def __init__(self):
        logging.info("Explainers initialized.")

    def shap_explain(self, model, x_train, y_test, data_row, Predicted_class):
        """
        Explain model predictions using SHAP KernelExplainer.
        """
        try:
            warnings.filterwarnings('ignore')

            # Background sample for SHAP
            background_sample = shap.sample(x_train, 200)

            # Create SHAP explainer
            shap_explainer = shap.KernelExplainer(model, data=background_sample)

            # Compute SHAP values
            shap_values = shap_explainer.shap_values(
                data_row, nsamples=2000, l1_reg="num_features(46)"
            )

            # Extract for predicted class
            shap_values_class = shap_values[Predicted_class][0]
            column_names = x_train.columns

            # Get positive SHAP values
            positive_indices = np.where(shap_values_class > 0)[0]
            positive_values = shap_values_class[positive_indices]
            positive_columns = np.array(column_names)[positive_indices]

            # Create sorted list
            positive_list = list(zip(positive_columns, positive_values))
            positive_list.sort(key=lambda x: x[1], reverse=False)
            top_features = positive_list[:25]

            logging.info(f"SHAP top features: {top_features}")
            return top_features

        except Exception as e:
            logging.error(f"Error in shap_explain: {e}")
            raise

    def lime_explain(self, model, x_train, y_test, sample, Predicted_class, classes_names):
        """
        Explain model predictions using LIME TabularExplainer.
        """
        try:
            warnings.filterwarnings('ignore')

            # Create LIME explainer
            lime_explainer = LimeTabularExplainer(
                x_train.values,
                feature_names=x_train.columns,
                class_names=classes_names[y_test.name],
                discretize_continuous=True,
                verbose=True
            )

            # Explain instance
            LIME_explanation = lime_explainer.explain_instance(sample[0], model, num_features=46, num_samples=2000, top_labels=20)          

            # Extract positive contributions
            lime_values = LIME_explanation.local_exp[Predicted_class]
            column_names = x_train.columns
            positive_list = [(x, y) for x, y in lime_values if y > 0]

            x_indices = [x for x, _ in positive_list]
            x_names = [column_names[idx] for idx in x_indices]
            y_values = [y for _, y in positive_list]

            # Sort and take top 25
            sorted_list = sorted(zip(x_names, y_values), key=lambda pair: pair[1], reverse=False)[:25]

            logging.info(f"LIME top features: {sorted_list}")
            return sorted_list

        except Exception as e:
            logging.error(f"Error in lime_explain: {e}")
            raise

import matplotlib.pyplot as plt
import numpy as np
import logging
import os

# Configure logging (only once in your script)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def Plot_the_data(Plot_Data, Colors, filename):

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
        bp_shap = ax.boxplot(
            values[0],
            positions=[positions[i] - 0.2],
            widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor=colors[0])
        )
        bp_lime = ax.boxplot(
            values[1],
            positions=[positions[i] + 0.25],
            widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor=colors[1])
        )

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

    # Ensure "plot" directory exists
    os.makedirs("plot", exist_ok=True)
    filepath = os.path.join("plot", f"{filename}.png")

    # Save and close
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Plot saved successfully at {filepath}")

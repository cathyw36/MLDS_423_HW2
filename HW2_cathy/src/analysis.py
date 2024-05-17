import logging
import os
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from cycler import cycler

# Configure logging
logger = logging.getLogger("clouds")

def save_figures(data: pd.DataFrame, fig_dir: Path, config: dict) -> List[Path]:
    """Generate and save exploratory data analysis figures.

    Args:
        data (pd.DataFrame): DataFrame containing features and the response variable.
        fig_dir (Path): Directory where figures will be saved.
        config (dict): Configuration settings for figure attributes.

    Returns:
        List[Path]: A list of paths where figures have been saved.
    """
    # Apply matplotlib configurations from user settings.
    mpl_settings = config.get('mpl_config', {})
    mpl.rcParams.update({
        'font.size': mpl_settings.get('font.size', 12),
        'axes.prop_cycle': cycler('color', mpl_settings.get('cycle_colors', ['b', 'g', 'r', 'c'])),
        'xtick.labelsize': mpl_settings.get('xtick.labelsize', 'medium'),
        'ytick.labelsize': mpl_settings.get('ytick.labelsize', 'medium'),
        'figure.figsize': mpl_settings.get('figure.figsize', (10, 5)),
        'axes.labelsize': mpl_settings.get('axes.labelsize', 'large'),
        'axes.labelcolor': mpl_settings.get('axes.labelcolor', 'black'),
        'axes.titlesize': mpl_settings.get('axes.titlesize', 'large'),
        'lines.color': mpl_settings.get('lines.color', 'blue'),
        'lines.linewidth': mpl_settings.get('lines.linewidth', 2),
        'text.color': mpl_settings.get('text.color', 'black'),
        'font.family': mpl_settings.get('font.family', 'sans-serif'),
        'font.sans-serif': mpl_settings.get('font.sans-serif', ['DejaVu Sans'])
    })

    # Data preparation based on configuration
    feature_col_names = config["generate_features"]["feature_col"]
    target_column = config["generate_features"]["target_col"]
    features = data[feature_col_names]
    target = data[target_column]
    figure_paths = []

    # Figure creation and saving
    for feature in features.columns:
        fig, ax = plt.subplots()
        ax.hist([features.loc[target == 0, feature], features.loc[target == 1, feature]], label=['Class 0', 'Class 1'])
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.legend()

        # Saving figure
        figure_filename = f"{feature}_eda_plot.png"
        figure_path = fig_dir / figure_filename
        try:
            fig.savefig(figure_path, bbox_inches='tight')
            logger.info(f"Figure {figure_filename} saved at {figure_path}")
        except Exception as exc:
            logger.error(f"Failed to save figure {figure_filename} at {figure_path}: {exc}")
            raise NotImplementedError from exc
        figure_paths.append(figure_path)
        plt.close(fig)  # Close the figure to release memory

    return figure_paths

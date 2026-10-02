import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_metric_vs_epsilon(summary_df: pd.DataFrame, metric_base: str, ylabel: str, title: str, output_path: str) -> None:
    """
    Plot mean metric value vs epsilon with error bars representing +/- 1 std dev.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    x = summary_df['epsilon'].values
    y_mean = summary_df[f"{metric_base}_mean"].values
    y_std = summary_df[f"{metric_base}_std"].values
    
    plt.figure()
    plt.errorbar(x, y_mean, yerr=y_std, fmt='-o', capsize=4)
    plt.xlabel('Epsilon (privacy parameter)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_metric_vs_users(summary_df: pd.DataFrame, metric_base: str, ylabel: str, title: str, output_path: str) -> None:
    """
    Plot mean metric value vs number of users with error bars representing +/- 1 std dev.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    x = summary_df['users'].values
    y_mean = summary_df[f"{metric_base}_mean"].values
    y_std = summary_df[f"{metric_base}_std"].values
    
    plt.figure()
    plt.errorbar(x, y_mean, yerr=y_std, fmt='-o', capsize=4)
    plt.xlabel('Number of Users')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_metric_vs_dictionary_size(summary_df: pd.DataFrame, metric_base: str, ylabel: str, title: str, output_path: str) -> None:
    """
    Plot mean metric value vs dictionary size with error bars representing +/- 1 std dev.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    x = summary_df['dictionary_size'].values
    y_mean = summary_df[f"{metric_base}_mean"].values
    y_std = summary_df[f"{metric_base}_std"].values
    
    plt.figure()
    plt.errorbar(x, y_mean, yerr=y_std, fmt='-o', capsize=4)
    plt.xlabel('Dictionary Size')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_metric_vs_cms_configuration(summary_df: pd.DataFrame, metric_base: str, ylabel: str, title: str, output_path: str, is_deterministic: bool = False) -> None:
    """
    Plot metric vs CMS configuration labels (m=32,k=16; etc.).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    labels = [f"m={row['m']},k={row['k']}" for _, row in summary_df.iterrows()]
    x = list(range(len(labels)))
    
    plt.figure()
    if is_deterministic:
        y_val = summary_df[metric_base].values
        plt.plot(x, y_val, '-o')
    else:
        y_mean = summary_df[f"{metric_base}_mean"].values
        y_std = summary_df[f"{metric_base}_std"].values
        plt.errorbar(x, y_mean, yerr=y_std, fmt='-o', capsize=4)
        
    plt.xticks(x, labels, rotation=15)
    plt.xlabel('CMS Configuration (m, k)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_comparison_vs_epsilon(summary_df: pd.DataFrame, metric_base: str, ylabel: str, title: str, output_path: str, ylim: tuple = None) -> None:
    """
    Plot comparative metrics vs epsilon for CMS and HCMS with error bars (+/- 1 std dev).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cms_df = summary_df[summary_df['mechanism'] == 'CMS'].sort_values('epsilon')
    hcms_df = summary_df[summary_df['mechanism'] == 'HCMS'].sort_values('epsilon')
    
    mean_col = f"{metric_base}_mean"
    std_col = f"{metric_base}_std"
    if mean_col not in summary_df.columns and f"{metric_base}_mean_seconds" in summary_df.columns:
        mean_col = f"{metric_base}_mean_seconds"
        std_col = f"{metric_base}_std_seconds"
        
    plt.figure()
    
    # CMS plot
    plt.errorbar(
        cms_df['epsilon'].values,
        cms_df[mean_col].values,
        yerr=cms_df[std_col].values,
        fmt='-o',
        capsize=4,
        label='CMS'
    )
    
    # HCMS plot
    plt.errorbar(
        hcms_df['epsilon'].values,
        hcms_df[mean_col].values,
        yerr=hcms_df[std_col].values,
        fmt='-s',
        capsize=4,
        label='HCMS'
    )

    
    plt.xlabel('Epsilon (privacy parameter)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if ylim is not None:
        plt.ylim(ylim)
        
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_categorical_comparison(categories: list, values: list, ylabel: str, title: str, output_path: str) -> None:
    """
    Plot simple categorical comparison bar chart (e.g. CMS vs HCMS communication or sketch memory).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure()
    plt.bar(categories, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


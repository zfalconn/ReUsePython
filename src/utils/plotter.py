import numpy as np
import matplotlib.pyplot as plt

def plot_model_metrics(models, metrics_dict, title="Model Comparison"):
    """
    Plots grouped bar chart for multiple metrics of models with colored bars and matching annotations.
    
    Args:
        models (list of str): Model names
        metrics_dict (dict): {metric_name: metric_values_list}, values must match len(models)
        title (str): Title of the plot
    """
    n_metrics = len(metrics_dict)
    x = np.arange(len(models))
    width = 0.8 / n_metrics  # adjust width for number of metrics

    # Color palette (auto-expand if needed)
    colors = ['skyblue', 'orange', 'lightgreen', 'pink', 'violet', 'wheat', 'salmon', 'lightgray']
    colors = colors[:n_metrics]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    axes = [ax1]

    # Create twin axes if more than 1 metric
    for i in range(1, n_metrics):
        ax_new = ax1.twinx()
        ax_new.spines["right"].set_position(("outward", 50 * i))
        axes.append(ax_new)

    # Plot each metric
    for i, ((metric_name, values), color) in enumerate(zip(metrics_dict.items(), colors)):
        bar = axes[i].bar(x + width*(i - n_metrics/2 + 0.5), values, width, label=metric_name, color=color)
        axes[i].set_ylabel(metric_name, color=color)
        axes[i].tick_params(axis="y", labelcolor=color)
        # Annotate inside bars
        for rect, val in zip(bar, values):
            height = rect.get_height()
            axes[i].text(rect.get_x() + rect.get_width()/2, height - (0.02*height),
                         f"{val:.2f}", ha='center', va='top', fontsize=9, color='black')

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.legend(loc='upper right')
    plt.show()


# ===== Example usage =====
models = ['YOLOv11n', 'YOLOv11s', 'YOLOv11m', 'YOLOv11l', 'YOLOv11x']
mAP50 = [0.992, 0.994, 0.994, 0.994, 0.9935]
Latency = [81.3, 177.0, 411.7, 667.4 , 1057.5]
FPS = [1000 / l for l in Latency]

metrics = {
    "mAP50": mAP50,
    "Latency (ms)": Latency,
    "FPS": FPS
}

plot_model_metrics(models, metrics, title="Model Comparison — mAP50, Latency, and FPS (.pt format)")

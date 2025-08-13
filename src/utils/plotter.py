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

def plot_latency_comparison(models, formats, latency_data, title="Latency Comparison Across Formats"):
    """
    Plots grouped bar chart for latency comparison across different formats and models.
    
    Args:
        models (list[str]): Model names.
        formats (list[str]): Model formats (e.g., '.pt', 'ncnn').
        latency_data (dict[str, list[float]]): Mapping format -> latency list (ms) per model.
        title (str): Chart title.
    """
    x = np.arange(len(models))  # model positions
    width = 0.8 / len(formats)  # bar width so all formats fit
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot each format
    for i, fmt in enumerate(formats):
        latencies = latency_data[fmt]
        ax.bar(x + i*width - (width*(len(formats)-1)/2), latencies, width, label=fmt)
        
        # Annotate values on top of bars
        for xi, val in zip(x, latencies):
            ax.text(xi + i*width - (width*(len(formats)-1)/2), val + max(latencies)*0.02,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=8)
    
    # Labels & formatting
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title(title)
    ax.legend(title="Format")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


# ===== Example usage =====
models = ['YOLOv11n', 'YOLOv11s', 'YOLOv11m', 'YOLOv11l', 'YOLOv11x']
mAP50_pt = [0.992, 0.994, 0.994, 0.994, 0.9935]
Latency_pt = [81.3, 177.0, 411.7, 667.4 , 1057.5]
FPS_pt = [1000 / l for l in Latency_pt]

mAP50_ncnn = [0.992, 0.994, 0.993, 0.994, 0.990]
latency_ncnn = [66.2, 139.2, 357.7, 428.9, 991.5]
FPS_ncnn = [1000 / l for l in latency_ncnn]


metrics_pt = {
    "mAP50": mAP50_pt,
    "Latency (ms)": Latency_pt,
    "FPS": FPS_pt
}

metrics_ncnn = {
    "mAP50": mAP50_ncnn,
    "Latency (ms)": latency_ncnn,
    "FPS": FPS_ncnn
}

# Calculate percentage increase
fps_increase_pct = [
    ((ncnn - pt) / pt) * 100
    for ncnn, pt in zip(FPS_ncnn, FPS_pt)
]

for model, pct in zip(['YOLOv11n', 'YOLOv11s', 'YOLOv11m', 'YOLOv11l', 'YOLOv11x'], fps_increase_pct):
    print(f"{model}: {pct:.2f}% increase in FPS")

# plot_model_metrics(models, metrics_ncnn, title="Model Comparison — mAP50, Latency, and FPS (.ncnn format)")

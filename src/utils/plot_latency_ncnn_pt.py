import matplotlib.pyplot as plt

# Example CPU latency data (replace with your actual measured averages in ms)
labels = ["PT Threaded", "PT No Thread", "NCNN Threaded", "NCNN No Thread"]
avg_cpu_latency = [150.49, 159.34, 76.53, 84.31]  # Example numbers

plt.figure(figsize=(7, 5))
plt.bar(labels, avg_cpu_latency, color='skyblue')

plt.ylabel("Average loop time (ms)")
plt.title("YOLOv11 'n' Latency Comparison - pt and ncnn format")
plt.xticks(rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with values
for i, v in enumerate(avg_cpu_latency):
    plt.text(i, v - 5, f"{v:.1f}", ha='center', fontsize=9)

plt.tight_layout()
plt.show()

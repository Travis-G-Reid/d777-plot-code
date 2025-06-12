import matplotlib.pyplot as plt
import numpy as np
import time
import random
import bisect


class BTree:
    """Simple B-tree implementation for benchmarking"""

    def __init__(self, max_children=50):
        self.root = []
        self.max_children = max_children

    def insert(self, value) -> None:
        bisect.insort(self.root, value)

    def search(self, value) -> bool:
        # Simplified search for benchmarking
        idx = bisect.bisect_left(self.root, value)
        return idx < len(self.root) and self.root[idx] == value


class SimpleGraph:
    """Simple graph using adjacency list"""

    def __init__(self):
        self.vertices = {}

    def add_vertex(self, vertex) -> None:
        if vertex not in self.vertices:
            self.vertices[vertex] = []

    def search(self, vertex):
        return vertex in self.vertices


def benchmark_seek(data_structure, values, num_seeks=1000):
    """Benchmark random seek operations"""
    search_values = random.sample(values, min(num_seeks, len(values)))

    start_time = time.perf_counter()
    for val in search_values:
        if isinstance(data_structure, list):
            _ = val in data_structure
        elif isinstance(data_structure, (dict, set)):
            _ = val in data_structure
        elif isinstance(data_structure, BTree):
            _ = data_structure.search(val)
        elif isinstance(data_structure, SimpleGraph):
            _ = data_structure.search(val)
    end_time = time.perf_counter()

    return (end_time - start_time) / len(search_values)


# Test different sizes
n_values = [10, 100, 1000, 10000, 100000, 200000]
num_trials = 3  # Number of trials to average

# Store results
results: dict[str, list[float]] = {
    "Hash Table (dict)": [],
    "Set": [],
    "List": [],
    "B-tree": [],
    "Graph": [],
}

print("Running benchmarks...")
print("-" * 60)

for n in n_values:
    print(f"Testing n = {n:,}")

    # Generate test data
    test_values = list(range(n))
    random.shuffle(test_values)

    # Dictionary (Hash Table)
    hash_times = []
    for _ in range(num_trials):
        hash_table = {val: True for val in test_values}
        hash_times.append(benchmark_seek(hash_table, test_values))
    results["Hash Table (dict)"].append(np.mean(hash_times))

    # Set
    set_times = []
    for _ in range(num_trials):
        test_set = set(test_values)
        set_times.append(benchmark_seek(test_set, test_values))
    results["Set"].append(np.mean(set_times))

    # List
    list_times = []
    for _ in range(num_trials):
        test_list = test_values.copy()
        # Limit seeks for large lists to avoid excessive runtime
        num_seeks = min(100, n) if n > 10000 else 1000
        list_times.append(benchmark_seek(test_list, test_values, num_seeks))
    results["List"].append(np.mean(list_times))

    # B-tree
    btree_times = []
    for _ in range(num_trials):
        btree = BTree()
        for val in test_values:
            btree.insert(val)
        btree_times.append(benchmark_seek(btree, test_values))
    results["B-tree"].append(np.mean(btree_times))

    # Graph
    graph_times = []
    for _ in range(num_trials):
        graph = SimpleGraph()
        for val in test_values:
            graph.add_vertex(val)
        graph_times.append(benchmark_seek(graph, test_values))
    results["Graph"].append(np.mean(graph_times))

# Normalize results to microseconds for better readability
for key in results:
    results[key] = [t * 1e6 for t in results[key]]  # Convert to microseconds

# Create projection functions based on theoretical complexities
def project_constant(n_values, times):
    """Project O(1) complexity"""
    # Use the average of the last few measurements as the constant
    avg_time = np.mean(times[-3:])
    return [avg_time for _ in n_values]

def project_logarithmic(n_values, times):
    """Project O(log n) complexity"""
    # Fit a logarithmic curve: time = a * log(n) + b
    actual_n = n_values[:len(times)]
    log_n = np.log(actual_n)
    coeffs = np.polyfit(log_n, times, 1)
    
    projected_log_n = np.log(n_values)
    return coeffs[0] * projected_log_n + coeffs[1]

def project_linear(n_values, times):
    """Project O(n) complexity"""
    # Fit a linear curve: time = a * n + b
    actual_n = n_values[:len(times)]
    coeffs = np.polyfit(actual_n, times, 1)
    
    return coeffs[0] * np.array(n_values) + coeffs[1]

# Create future n values for projection
future_n_values = [1e6, 5e6, 1e7, 5e7, 1e8]
all_n_values = n_values + future_n_values

# Create the plot
plt.figure(figsize=(14, 10))

# Define colors and markers for each data structure
styles = {
    "Hash Table (dict)": ("b-", "o", "blue", project_constant),
    "Set": ("g--", "s", "green", project_constant),
    "List": ("r-", "^", "red", project_linear),
    "B-tree": ("m-", "d", "purple", project_logarithmic),
    "Graph": ("orange", "*", "orange", project_constant),
}

# Plot each data structure
for ds_name, times in results.items():
    line_style, marker, color, project_func = styles[ds_name]
    
    # Plot actual measured data
    plt.loglog(
        n_values[: len(times)],
        times,
        line_style,
        marker=marker,
        linewidth=2,
        markersize=8,
        label=f"{ds_name} (measured)",
        color=color,
    )
    
    # Project future performance
    projected_times = project_func(all_n_values, times)
    # Only plot the projection part
    plt.loglog(
        all_n_values[len(times):],
        projected_times[len(times):],
        ':',
        linewidth=2,
        label=f"{ds_name} (projected)",
        color=color,
        alpha=0.6,
    )

# Add vertical line to separate measured from projected
plt.axvline(x=n_values[-1], color='gray', linestyle='--', alpha=0.5, label='Projection boundary')

# Customize the plot
plt.xlabel("Number of Elements (n)", fontsize=14)
plt.ylabel("Average Time per Seek (microseconds)", fontsize=14)
plt.title(
    "Seek Performance Across Data Structures with Future Projections",
    fontsize=16,
    fontweight="bold",
)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(loc="upper left", fontsize=10, frameon=True, shadow=True, ncol=2)

# Set axis limits
ax = plt.gca()
ax.set_xlim(5, 2e8)
ax.set_ylim(1e-2, 1e6)

plt.tight_layout()
plt.show()

# Print numerical comparison including projections
print("\nBenchmark Results (microseconds per seek):")
print("-" * 100)
print(f"{'Data Structure':<20} {'n=1,000':<15} {'n=100,000':<15} {'n=500,000':<15} {'n=10M (proj)':<15} {'n=100M (proj)':<15}")
print("-" * 100)

for ds_name in results:
    values = results[ds_name]
    line_style, marker, color, project_func = styles[ds_name]
    
    # Get measured values
    n1k = values[2] if len(values) > 2 else "N/A"
    n100k = values[4] if len(values) > 4 else "N/A"
    n500k = values[5] if len(values) > 5 else "N/A"
    
    # Get projected values
    projected = project_func(all_n_values, values)
    n10m_idx = n_values.index(10000) if 10000 in n_values else 0
    n10m = projected[len(n_values) + 2]  # Index for 10M
    n100m = projected[len(n_values) + 4]  # Index for 100M
    
    print(f"{ds_name:<20} {n1k:<15.3f} {n100k:<15.3f} {n500k:<15.3f} {n10m:<15.3f} {n100m:<15.3f}")

# Calculate and display performance ratios
print("\nPerformance Ratios (compared to Hash Table at n=100,000):")
print("-" * 60)
if len(results["Hash Table (dict)"]) > 4:
    hash_baseline = results["Hash Table (dict)"][4]
    for ds_name in results:
        if len(results[ds_name]) > 4:
            ratio = results[ds_name][4] / hash_baseline
            print(f"{ds_name:<20} {ratio:>10.1f}x slower")

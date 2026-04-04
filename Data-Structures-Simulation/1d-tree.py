import csv
import time
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(50000)  # Increase to handle deeper recursion

# ========== CONFIGURATION ==========

GOWALLA_FILE = "Gowalla_totalCheckins.txt"  # Update path as needed
QUERY_SIZES = [3600, 86400, 604800]  # Query widths in seconds: 1 hour, 1 day, 1 week
NUM_QUERIES = 500  # Number of queries to generate
SEED = 42

# ========== DATA PREPARATION ==========

# Pandas Loading Function
def load_gowalla_data(filename: str) -> List[Tuple[int, int]]:
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0, 1], names=['user_id', 'time_str'])
    df = df.dropna()
    df['timestamp'] = pd.to_datetime(df['time_str'], format="%Y-%m-%dT%H:%M:%SZ")
    print(df['timestamp'].head())  # Debug: Check raw datetime objects
    df['timestamp'] = df['timestamp'].astype(int) // 10**9
    min_time = df['timestamp'].min()
    print(f"Min timestamp (seconds): {min_time}")  # Debug: Check min time
    norm_user_time = [(row.user_id, row.timestamp - min_time) for _, row in df.iterrows()]
    return norm_user_time, min_time

# ========== 1D-TREE DEFINITION ==========

class TreeNode:
    def __init__(self, range_start, range_end, level, parent=None):
        self.range_start = range_start  # inclusive
        self.range_end = range_end      # exclusive
        self.level = level
        self.parent = parent
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None
        self.median: Optional[int] = None
        self.user_count: int = 0 # Count of user_ids in this node's range
        
    def __repr__(self):
        return f"TreeNode({self.range_start},{self.range_end},level={self.level},median={self.median}, user_count={self.user_count})"

def build_1dtree(data: List[Tuple[int, int]], range_start: int, range_end: int, level: int =0, parent: Optional[TreeNode] = None) -> Optional[TreeNode]:
    # data: list of (user_id, seconds)
    if not data:
        return None
    
    # Terminate if data is small or range is too narrow
    if len(data) <= 1 or range_end - range_start <= 1:
        node = TreeNode(range_start, range_end, level, parent)
        if data:
            node.median = data[0][1]
            node.user_count = len(data)  # Store count
        return node

    # Assume data is pre-sorted by time in main
    # Find leftmost median
    median_idx = (len(data) // 2) - 1 if len(data) % 2 == 0 else len(data) // 2
    median_time = data[median_idx][1]
    # Split data
    left_data = [d for d in data if d[1] < median_time]
    right_data = [d for d in data if d[1] >= median_time]
    
    # Ensure progress by checking split
    if not left_data or not right_data:
        node = TreeNode(range_start, range_end, level, parent)
        node.median = median_time
        node.user_count = len(data)  # Make leaf if split fails
        return node

    # Find the first check-in after median for right child range
    #right_start = right_data[0][1] if right_data else median_time + 1

    # Create node before recursive calls
    node = TreeNode(range_start, range_end, level, parent)
    node.median = median_time
    #node.user_count = 0  # Will be updated from children

    # Build children 
    # print(f"Level {level}, Data count: {len(data)}, Range: [{range_start}, {range_end})")
    node.left = build_1dtree(left_data, range_start, median_time, level + 1, node)
    node.right = build_1dtree(right_data, median_time, range_end, level + 1, node)

    # Set this node's range based on children (tight intervals)
    if node.left and node.right:
        node.range_start = node.left.range_start
        node.range_end = node.right.range_end
    elif node.left:
        node.range_start = node.left.range_start
        node.range_end = node.left.range_end
    elif node.right:
        node.range_start = node.right.range_start
        node.range_end = node.right.range_end
    else:
        return None

    #Update user_count from children
    node.user_count = (node.left.user_count if node.left else 0) + (node.right.user_count if node.right else 0)

    return node

def print_tree(node: TreeNode, indent=0):
    if node is None:
        return
    print('  ' * indent + f"Level {node.level}: Range [{node.range_start}, {node.range_end}), Median {node.median}, Count {node.user_count}")
    print_tree(node.left, indent + 1)
    print_tree(node.right, indent + 1)
 

# ========== QUERY GENERATION ==========

def generate_queries(timespan: int, query_size: int, num_queries: int):
    random.seed(SEED)
    queries = []
    for _ in range(num_queries):
        start = random.randint(0, max(0, timespan - query_size))
        end = start + query_size
        queries.append((start, end))
    return queries

# ========== SRC SEARCH ==========

def src_search(node: TreeNode, q_start: int, q_end: int, parent: Optional[TreeNode]=None) -> Optional[TreeNode]:
    # Returns (result_node)
    # 1. If current node is None, return parent (or None if no parent)
    if node is None:
        return parent
    
    # 2. Check if node intersects query
    node_intersects = not (node.range_end <= q_start or node.range_start >= q_end)

    # 3. If node does NOT intersect query:
    if not node_intersects:
        # 3.1 Check sister node (the other child of the parent)
        if parent is None:
            return None  # or handle root case accordingly
        sister = parent.right if node == parent.left else parent.left
        if sister is None:
            # 3.1 No sister → return parent
            return parent
        sister_intersects = not (sister.range_end <= q_start or sister.range_start >= q_end)
        if not sister_intersects:
            # 3.2 Sister has no intersection → return parent
            return parent
        sister_fully_covers = sister.range_start <= q_start and sister.range_end >= q_end
        
        # 3.3 Sister has partial intersection (not full cover) → return parent
        if not sister_fully_covers:
            return parent
        # 3.4 Sister fully covers query → recurse into sister
        return src_search(sister, q_start, q_end, parent)
    

    # 4. If node fully covers query
    node_fully_covers = node.range_start <= q_start and node.range_end >= q_end
    if node_fully_covers:
        # Check left child
        if node.left and node.left.range_start <= q_start and node.left.range_end >= q_end:
            return src_search(node.left, q_start, q_end, node)
        # Check right child
        if node.right and node.right.range_start <= q_start and node.right.range_end >= q_end:
            return src_search(node.right, q_start, q_end, node)
        # Current node is the smallest covering node
        return node
    # 5 Node intersects but does not fully cover → return parent
    return parent

# ========= UNIFORM-TIME BASELINE (THEORETICAL) =========

def count_distinct_times_from_data(gowalla: List[Tuple[int, int]]) -> int:
    """Count distinct (normalized) timestamps from loaded gowalla data."""
    times = [t for _, t in gowalla]
    return len(set(times))

def make_uniform_times(distinct_count: int, max_time: int) -> List[int]:
    """
    Create 'distinct_count' strictly increasing integer timestamps, evenly spaced over [0, max_time].
    Assumes distinct_count <= max_time + 1 (true for second-resolution data).
    """
    if distinct_count <= 0:
        return []
    if distinct_count == 1:
        return [0]
    # Evenly spaced floats, then convert to strictly increasing ints
    grid = np.linspace(0, max_time , num=distinct_count)
    vals = np.rint(grid).astype(np.int64)

    # Enforce strict increase and bounds [0, timespan]
    vals[0] = max(0, min(max_time, vals[0]))
    for i in range(1, distinct_count):
        # ensure strictly increasing by at least 1
        vals[i] = max(vals[i], vals[i-1] + 1)
        if vals[i] > max_time:
            vals[i] = max_time  # clamp; should not happen if distinct_count <= timespan+1
    return vals.tolist()

def run_time_experiment(data: List[Tuple[int, int]],
                        range_start: int,
                        range_end: int,
                        query_sizes: List[int],
                        num_queries: int,
                        output_prefix: str):
    """
    Generic runner: builds 1D-tree on 'data' (already normalized), runs queries,
    writes CSV and histograms with the given prefix (e.g., 'time_' or 'uniform_time_').
    """
    # Sort by key (time)
    data_sorted = sorted(data, key=lambda x: x[1])

    # Build tree
    print(f"Building 1D-tree for prefix '{output_prefix}' ...")
    root = build_1dtree(data_sorted, range_start, range_end, level=0)
    if root is None:
        print("Failed to build tree. Skipping.")
        return
    print("Tree built.")

    # Query domain
    timespan = range_end - range_start

    for query_size in query_sizes:
        print(f"[{output_prefix}] Generating {num_queries} queries size={query_size} ...")
        queries = generate_queries(timespan, query_size, num_queries)
        print(f"[{output_prefix}] Generated {len(queries)} queries.")

        out_rows = []
        for (q_start, q_end) in queries:
            t1 = time.perf_counter()
            res = src_search(root, q_start, q_end)
            t2 = time.perf_counter()
            proc_time = t2 - t1

            if res is None:
                out_rows.append([q_start, q_end, "invalid", "invalid", -1, -1, -1, proc_time])
                continue

            node_range_size = res.range_end - res.range_start
            fp_ratio = node_range_size / (q_end - q_start)
            out_rows.append([q_start, q_end, res.range_start, res.range_end, res.level, fp_ratio, res.user_count, proc_time])

        # Write CSV
        output_csv = f"{output_prefix}src_experiment_output_size_{query_size}.csv"
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_start", "query_end",
                "result_node.range_start", "result_node.range_end",
                "result_node.level", "false_positive_ratio",
                "node_count", "processing_time"
            ])
            writer.writerows(out_rows)
        print(f"[{output_prefix}] Results saved to {output_csv}")

        # Histograms (no seaborn; one figure per chart; no explicit colors)
        df = pd.DataFrame(out_rows, columns=[
            "query_start", "query_end",
            "result_node.range_start", "result_node.range_end",
            "result_node.level", "false_positive_ratio",
            "node_count", "processing_time"
        ])
        valid = df[df["result_node.level"] != -1]

        if not valid.empty:
            plt.figure()
            valid["result_node.level"].hist(bins=range(int(valid["result_node.level"].min()),
                                                       int(valid["result_node.level"].max()) + 2))
            plt.xlabel("Returned Node Level")
            plt.ylabel("Frequency")
            plt.title(f"{output_prefix}Histogram of Returned Node Levels (Query Size: {query_size}s)")
            plt.savefig(f"{output_prefix}node_level_histogram_size_{query_size}.png")
            plt.close()

            plt.figure()
            valid["false_positive_ratio"].hist(bins=50)
            plt.xlabel("False Positive Ratio")
            plt.ylabel("Frequency")
            plt.title(f"{output_prefix}Histogram of False Positive Ratios (Query Size: {query_size}s)")
            plt.savefig(f"{output_prefix}fp_ratio_histogram_size_{query_size}.png")
            plt.close()

            plt.figure()
            valid["processing_time"].hist(bins=50)
            plt.xlabel("Processing Time (sec)")
            plt.ylabel("Frequency")
            plt.title(f"{output_prefix}Histogram of Query Processing Times (Query Size: {query_size}s)")
            plt.savefig(f"{output_prefix}processing_time_histogram_size_{query_size}.png")
            plt.close()

            print(f"[{output_prefix}] Histograms saved for query size {query_size}.")
        else:
            print(f"[{output_prefix}] No valid results for query size {query_size}.")


def main_time_and_uniform():
    """
    Runs TWO experiments:
      1) Real time-based (outputs with 'time_' prefix)
      2) Theoretical uniform time-based (prefix 'uniform_time_')
    """
    print("Loading time data...")
    gowalla, min_time_global = load_gowalla_data(GOWALLA_FILE)
    if not gowalla:
        print("No valid data loaded. Exiting.")
        return

    # Real dataset domain (normalized seconds)
    times = [t for _, t in gowalla]
    max_time = max(times)
    range_start = min(times)
    range_end = max(times) + 1  # right-open
    timespan = range_end - range_start
    
    print(f"Data loaded. Points: {len(gowalla)}")
    print(f"Time domain: [{range_start}, {range_end}) span={range_end - range_start}")

    # ---- 1) Real TIME experiment with 'time_' prefix ----
    run_time_experiment(
        data=gowalla,
        range_start=range_start,
        range_end=range_end,
        query_sizes=QUERY_SIZES,
        num_queries=NUM_QUERIES,
        output_prefix="time_"
    )

    # ---- 2) UNIFORM TIME experiment with 'uniform_time_' prefix ----
    # Count distinct times
    num_distinct = count_distinct_times_from_data(gowalla)

    print(f"Distinct check-in times: {num_distinct}")
    print(f"Timespan (seconds): {timespan}")

    # Build uniform timestamps over [0, timespan], length = num_distinct
    uniform_ts = make_uniform_times(num_distinct, max_time)

    # Create uniform dataset (dummy IDs 0..K-1)
    uniform_data = list(zip(range(len(uniform_ts)), uniform_ts))

    # Use exact same domain [0, timespan] -> range_end = timespan + 1
    run_time_experiment(
        data=uniform_data,
        range_start=0,
        range_end=timespan,
        query_sizes=QUERY_SIZES,
        num_queries=NUM_QUERIES,
        output_prefix="uniform_time_"
    )


if __name__ == "__main__":
    main_time_and_uniform()



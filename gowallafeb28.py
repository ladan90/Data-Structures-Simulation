
import csv
import time
import random
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import bisect
import matplotlib.pyplot as plt
import sys
import math
from collections import deque, defaultdict
import os
import shutil
import glob
import openpyxl

sys.setrecursionlimit(50000) 
l2_results = []  # Global list to collect L2 data
final_table_rows = []   # <--- ADD THIS LINE

# ========== CONFIGURATION ==========

GOWALLA_FILE = "Gowalla_totalCheckins.txt"
QUERY_RANGES = [60, 3600, 86400, 604800]  # 1 min , 1 hour, 1day , 1 week
SEED = 66
INITIAL_QUERIES_PER_SIZE = 5000
ADD_QUERIES = 500
MAX_QUERIES_PER_SIZE = 200000# Total queries per query range; continue until this is reached
TARGET_QUERIES = 120000  
EPSILON = 0.001
N_FIXED = 4194304  # Fixed N=2^22
n_FIXED = 21  # Since 2^{n+1}=2^22
TIMESPAN = [0, 2678399]
#S_STAR = [4, 5, 6, 7, 8, 9, 10,11,12,13,14, 15, 16, 17, 18, 19, 20]  # List of S_STAR values 
HORIZONTAL_BAR = False  # Set True for x=P(k), y=k (horizontal)
BASE_OUTPUT_DIR = "DenseDAG9"

# ========== DATA PREPARATION ==========

def load_gowalla_data(filename: str) -> List[int]:
    """
    Load times only from Gowalla, normalize to [0, max-min], return sorted unique times list.
    """
    try:
        df = pd.read_csv(filename, sep='\t', header=None, usecols=[1], names=['time_str'])
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return []
    df = df.dropna()
    try:
        df['timestamp'] = pd.to_datetime(df['time_str'], format="%Y-%m-%dT%H:%M:%SZ")
    except ValueError as e:
        print(f"Error: Invalid timestamp format in data: {e}")
        return []
    print(df['timestamp'].head())  # Print datetimes
    df['timestamp'] = df['timestamp'].astype(int) // 10**9
    min_time = df['timestamp'].min()
    print(f"Min timestamp (seconds): {min_time}")
    times = sorted(set(int(t - min_time) for t in df['timestamp']))
    return times[:N_FIXED] # Limit to first 4194304 distinct times



# ========== 1D-TREE DEFINITION ==========

class TreeNode:
    def __init__(self, range_start: int, range_end: int, level: int, parent=None):
        self.range_start = range_start  # inclusive
        self.range_end = range_end      # exclusive
        self.level = level
        self.parent: Optional["TreeNode"] = parent
        self.children: List["TreeNode"] = []
        self.median: Optional[int] = None
        self.user_count: int = 0  # number of points in this node's range

    def __repr__(self):
        return f"TreeNode([{self.range_start},{self.range_end}), level={self.level}, median={self.median}, count={self.user_count}]"

    def print_tree(self, indent=0, file=None):
        output = ' ' * indent + str(self) + f", children={len(self.children)}\n"
        if file:
            file.write(output)
        else:
            print(output)
        for child in self.children:
            child.print_tree(indent + 2, file)

def build_1dtree(times: List[int], range_start: int, range_end: int) -> Optional[TreeNode]:
    n = len(times)
    if n == 0:
        return None

    root = TreeNode(range_start, range_end, 0)
    queue = deque([(root, 0, n)])  # (node, idx_start, idx_end)

    while queue:
        node, idx_start, idx_end = queue.popleft()
        count = idx_end - idx_start

        if count <= 1 or (node.range_end - node.range_start) <= 1:
            node.user_count = count
            if count >= 1:
                node.median = times[idx_start]
            continue

        if count <= 4:
            uniq = sorted(set(times[idx_start:idx_end]))
            if len(uniq) == 1:
                t = uniq[0]
                node.range_start = t
                node.median = t
                node.user_count = count
                continue

            leaves = []
            for i, t in enumerate(uniq):
                seg_start = t
                seg_end = uniq[i+1] if i+1 < len(uniq) else node.range_end
                leaf = TreeNode(seg_start, seg_end, node.level + 1, node)
                leaf.median = t
                leaf.user_count = 1  # Unique
                leaves.append(leaf)
            node.children = leaves
            node.median = uniq[len(uniq)//2]
            node.user_count = len(leaves)
            if leaves:
                node.range_start = leaves[0].range_start
                node.range_end = leaves[-1].range_end
            continue

        mid_idx = idx_start + count // 2
        median = times[mid_idx]
        node.median = median

        split_idx = bisect.bisect_left(times[idx_start:idx_end], median) + idx_start

        if split_idx == idx_start or split_idx == idx_end:
            node.user_count = count
            continue

        left = TreeNode(node.range_start, median, node.level + 1, node)
        right = TreeNode(median, node.range_end, node.level + 1, node)
        node.children = [left, right]
        queue.append((left, idx_start, split_idx))
        queue.append((right, split_idx, idx_end))

    return root

# ========== SRC SEARCH FOR TREE ==========

def src_search_tree(node: Optional[TreeNode], q_start: int, q_end: int) -> Optional[TreeNode]:
    if q_start > q_end:
        return None
    
    if node is None:
        return None


    if node.range_end <= q_start or node.range_start >= q_end:
        return None

    if not (node.range_start <= q_start and node.range_end >= q_end):
        return None

    for child in node.children:
        res = src_search_tree(child, q_start, q_end)
        if res is not None:
            return res

    return node

# ========== 1D-DAG DEFINITION ==========

class DagNode3:
    def __init__(self, range_start: int, range_end: int, level: int):
        self.range_start = range_start  # inclusive
        self.range_end = range_end      # exclusive
        self.level = level
        self.parents: List["DagNode3"] = []
        self.children: List["DagNode3"] = []
        self.median: Optional[int] = None
        self.user_count: int = 0
        self.is_middle_child: bool = False

    def __repr__(self):
        return f"DagNode3([{self.range_start},{self.range_end}), level={self.level}, median={self.median}, count={self.user_count}, is_middle={self.is_middle_child}]"

    def print_dag(self, indent=0, file=None):
        output = ' ' * indent + str(self) + f", parents={len(self.parents)}, children={len(self.children)}\n"
        if file:
            file.write(output)
        else:
            print(output)
        for child in self.children:
            child.print_dag(indent + 2, file)

def build_1ddag3(times: List[int], range_start: int, range_end: int) -> Optional[DagNode3]:
    n = len(times)
    if n == 0:
        return None

    node_map = {}  # (start, end) -> DagNode

    def create_node(start, end, lev, is_middle=False) -> DagNode3:
        key = (start, end)
        if key in node_map:
            node = node_map[key]
            node.is_middle_child = node.is_middle_child or is_middle
            return node
        node = DagNode3(start, end, lev)
        node.is_middle_child = is_middle
        node_map[key] = node
        return node

    root = create_node(range_start, range_end, 0)
    queue = deque([(root, 0, n)])  # (node, idx_start, idx_end)

    while queue:
        curr_node, c_idx_start, c_idx_end = queue.popleft()
        if curr_node.user_count > 0:
            continue  # Processed

        c_count = c_idx_end - c_idx_start

        if c_count <= 1 or curr_node.range_end - curr_node.range_start <= 1:
            curr_node.user_count = c_count
            continue

        ready = True
        left_sib = None
        right_sib = None
        shared_left = None
        shared_right = None
        if curr_node.is_middle_child and curr_node.parents:
            parent = curr_node.parents[0]
            if len(parent.children) >= 3:
                left_sib = parent.children[0]
                right_sib = parent.children[-1]
                if left_sib.user_count == 0 or right_sib.user_count == 0 or not left_sib.children or not right_sib.children:
                    ready = False
                else:
                    shared_left = left_sib.children[-1]
                    shared_right = right_sib.children[0]
                    if shared_left.user_count == 0 or shared_right.user_count == 0:
                        ready = False

        if not ready:
            queue.append((curr_node, c_idx_start, c_idx_end))
            continue

        curr_node.user_count = c_count

        if c_count < 3:
            mid_idx = c_idx_start + c_count // 2
            median = times[mid_idx]
            curr_node.median = median

            left_split = bisect.bisect_left(times, median, c_idx_start, c_idx_end)

            left_start = curr_node.range_start
            left_end = median
            if left_split > c_idx_start:
                left_node = create_node(left_start, left_end, curr_node.level + 1)
                left_node.parents.append(curr_node)
                curr_node.children.append(left_node)
                queue.append((left_node, c_idx_start, left_split))

            right_start = median
            right_end = curr_node.range_end
            if c_idx_end > left_split:
                right_node = create_node(right_start, right_end, curr_node.level + 1)
                right_node.parents.append(curr_node)
                curr_node.children.append(right_node)
                queue.append((right_node, left_split, c_idx_end))

            continue

        mid_idx = c_idx_start + c_count // 2
        median = times[mid_idx]
        curr_node.median = median

        left_split = bisect.bisect_left(times, median, c_idx_start, c_idx_end)

        if curr_node.is_middle_child:
            if c_count <= 4:
                collected_children = []
                for sib in [left_sib, right_sib]:
                    if sib:
                        for child in sib.children:
                            if child.range_end > curr_node.range_start and child.range_start < curr_node.range_end:
                                collected_children.append(child)
                                if curr_node not in child.parents:
                                    child.parents.append(curr_node)
                collected_children = list(set(collected_children))
                collected_children.sort(key=lambda x: x.range_start)
                curr_node.children = collected_children
            else:
                if shared_left and curr_node not in shared_left.parents:
                    shared_left.parents.append(curr_node)
                if shared_right and curr_node not in shared_right.parents:
                    shared_right.parents.append(curr_node)

                children = [shared_left] if shared_left else []
                if shared_left and shared_right:
                    m1 = shared_left.median if shared_left.median is not None else shared_left.range_start
                    m2 = shared_right.median if shared_right.median is not None else shared_right.range_start
                    if m1 < m2:
                        middle_start = m1
                        middle_end = m2
                        middle_idx_start = bisect.bisect_left(times, middle_start, c_idx_start, c_idx_end)
                        middle_idx_end = bisect.bisect_left(times, middle_end, c_idx_start, c_idx_end)
                        if middle_idx_end > middle_idx_start:
                            middle_node = create_node(middle_start, middle_end, curr_node.level + 1, is_middle=True)
                            middle_node.parents.append(curr_node)
                            queue.append((middle_node, middle_idx_start, middle_idx_end))
                            children.append(middle_node)
                if shared_right:
                    children.append(shared_right)
                curr_node.children = children
        else:
            if c_count <= 4:
                uniq = sorted(set(times[c_idx_start:c_idx_end]))
                if len(uniq) == 1:
                    t = uniq[0]
                    curr_node.range_start = t
                    curr_node.median = t
                    curr_node.user_count = 1
                    continue

                children = []
                for i, t in enumerate(uniq):
                    seg_start = t
                    seg_end = uniq[i+1] if i+1 < len(uniq) else curr_node.range_end
                    child = create_node(seg_start, seg_end, curr_node.level + 1)
                    child.parents.append(curr_node)
                    child.median = t
                    child.user_count = 1
                    children.append(child)
                curr_node.children = children
                curr_node.median = uniq[len(uniq)//2]
                curr_node.user_count = len(children)
                if children:
                    curr_node.range_start = children[0].range_start
                    curr_node.range_end = children[-1].range_end
            else:
                # Compute left_median, right_median for middle
                left_idx_start = c_idx_start
                left_idx_end = left_split
                left_count = left_idx_end - left_idx_start
                left_mid_idx = left_idx_start + left_count // 2
                left_median = times[left_mid_idx] if left_count > 0 else None

                right_idx_start = left_split
                right_idx_end = c_idx_end
                right_count = right_idx_end - right_idx_start
                right_mid_idx = right_idx_start + right_count // 2
                right_median = times[right_mid_idx] if right_count > 0 else None

                left_start = curr_node.range_start
                left_end = median
                left = create_node(left_start, left_end, curr_node.level + 1) if left_count > 0 else None
                if left:
                    left.parents.append(curr_node)

                children = [left] if left else []
                if left_median is not None and right_median is not None and left_median < right_median:
                    middle_start = left_median
                    middle_end = right_median
                    middle_idx_start = bisect.bisect_left(times, middle_start, c_idx_start, c_idx_end)
                    middle_idx_end = bisect.bisect_left(times, middle_end, c_idx_start, c_idx_end)
                    middle_count = middle_idx_end - middle_idx_start
                    if middle_count > 0:
                        middle = create_node(middle_start, middle_end, curr_node.level + 1, is_middle=True)
                        middle.parents.append(curr_node)
                        queue.append((middle, middle_idx_start, middle_idx_end))
                        children.append(middle)

                right_start = median
                right_end = curr_node.range_end
                right = create_node(right_start, right_end, curr_node.level + 1) if right_count > 0 else None
                if right:
                    right.parents.append(curr_node)
                if right:
                    children.append(right)

                curr_node.children = children

                if left:
                    queue.append((left, left_idx_start, left_idx_end))
                if right:
                    queue.append((right, right_idx_start, right_idx_end))

    return root

# ========== SRC SEARCH FOR DAG ==========

def src_search_dag3(node: Optional[DagNode3], q_start: int, q_end: int) -> Optional[DagNode3]:
    if node is None:
        return None

    if node.range_end <= q_start or node.range_start >= q_end:
        return None

    if not (node.range_start <= q_start and node.range_end >= q_end):
        return None

    for child in node.children:
        res = src_search_dag3(child, q_start, q_end)
        if res is not None:
            return res

    return node

#===========5DAG definition========
class DagNode5:
    def __init__(self, range_start: int, range_end: int, level: int):
        self.range_start = range_start
        self.range_end = range_end
        self.level = level
        self.parents: List["DagNode5"] = []
        self.children: List["DagNode5"] = []
        self.median: Optional[int] = None
        self.left_median: Optional[int] = None
        self.right_median: Optional[int] = None
        self.user_count: int = 0
        self.is_middle_child: bool = False
        self.is_left_middle_child: bool = False
        self.is_right_middle_child: bool = False
        self.requeue_count: int = 0

    def __repr__(self):
        return (f"DagNode5([{self.range_start},{self.range_end}), level={self.level}, "
                f"median={self.median}, count={self.user_count}, is_middle={self.is_middle_child}, "
                f"is_lm={self.is_left_middle_child}, is_rm={self.is_right_middle_child}]")

    def print_dag(self, indent=0):
        print(' ' * indent + str(self) + f", parents={len(self.parents)}, children={len(self.children)}")
        for child in self.children:
            child.print_dag(indent + 2)

def compute_median(times: List[int], start: int, end: int) -> Optional[int]:
    count = end - start
    return times[start + count // 2] if count > 0 else None

def build_1ddag5(times: List[int], range_start: int, range_end: int) -> Optional[DagNode5]:
    if not times:
        return None

    node_map = {}

    def create_node(start, end, level, is_middle=False, is_left_middle=False, is_right_middle=False) -> DagNode5:
        key = (start, end)
        if key in node_map:
            node = node_map[key]
            node.is_middle_child |= is_middle
            node.is_left_middle_child |= is_left_middle
            node.is_right_middle_child |= is_right_middle
            return node
        node = DagNode5(start, end, level)
        node.is_middle_child = is_middle
        node.is_left_middle_child = is_left_middle
        node.is_right_middle_child = is_right_middle
        node_map[key] = node
        return node

    root = create_node(range_start, range_end, 0)
    queue = deque([(root, 0, len(times))])

    while queue:
        curr_node, idx_start, idx_end = queue.popleft()
        if curr_node.user_count > 0:
            continue

        count = idx_end - idx_start
        if count <= 1 or curr_node.range_end - curr_node.range_start <= 1:
            curr_node.user_count = count
            continue

        ready = True
        if curr_node.is_middle_child and curr_node.parents:
            parent = curr_node.parents[0]
            if len(parent.children) >= 3:
                left_sib, right_sib = parent.children[0], parent.children[-1]
                if (left_sib.user_count == 0 or right_sib.user_count == 0 or
                        not left_sib.children or not right_sib.children or
                        left_sib.children[-1].user_count == 0 or right_sib.children[0].user_count == 0):
                    ready = False

        if not ready:
            curr_node.requeue_count += 1
            if curr_node.requeue_count <= 3:
                queue.append((curr_node, idx_start, idx_end))
                continue

        curr_node.user_count = count
        median = compute_median(times, idx_start, idx_end)
        curr_node.median = median
        left_split = bisect.bisect_left(times, median, idx_start, idx_end)

        if count <= 4:
            uniq = sorted(set(times[idx_start:idx_end]))
            if len(uniq) == 1:
                t = uniq[0]
                curr_node.range_start = t
                curr_node.median = t
                curr_node.user_count = 1
                continue
            for i, t in enumerate(uniq):
                seg_start = t
                seg_end = uniq[i + 1] if i + 1 < len(uniq) else curr_node.range_end
                child = create_node(seg_start, seg_end, curr_node.level + 1)
                child.parents.append(curr_node)
                child.median = t
                child.user_count = 1
                curr_node.children.append(child)
            continue

        # Split into left, middle, right and possibly left-middle and right-middle
        left_start = curr_node.range_start
        left_end = median
        right_start = median
        right_end = curr_node.range_end

        left = create_node(left_start, left_end, curr_node.level + 1)
        right = create_node(right_start, right_end, curr_node.level + 1)
        left.parents.append(curr_node)
        right.parents.append(curr_node)

        left_idx_start = idx_start
        left_idx_end = left_split
        right_idx_start = left_split
        right_idx_end = idx_end

        left_count = left_idx_end - left_idx_start
        right_count = right_idx_end - right_idx_start

        if left_count > 0:
            left.median = compute_median(times, left_idx_start, left_idx_end)
            queue.append((left, left_idx_start, left_idx_end))
            curr_node.children.append(left)

        if right_count > 0:
            right.median = compute_median(times, right_idx_start, right_idx_end)
            queue.append((right, right_idx_start, right_idx_end))
            curr_node.children.append(right)

        # Middle child
        if left_count > 0 and right_count > 0:
            left_mid = compute_median(times, left_idx_start, left_idx_end)
            right_mid = compute_median(times, right_idx_start, right_idx_end)
            if left_mid is not None and right_mid is not None and left_mid < right_mid:
                middle_start = left_mid
                middle_end = right_mid
                mid_idx_start = bisect.bisect_left(times, middle_start, idx_start, idx_end)
                mid_idx_end = bisect.bisect_left(times, middle_end, idx_start, idx_end)
                if mid_idx_end > mid_idx_start:
                    middle = create_node(middle_start, middle_end, curr_node.level + 1, is_middle=True)
                    middle.parents.append(curr_node)
                    middle.median = compute_median(times, mid_idx_start, mid_idx_end)
                    queue.append((middle, mid_idx_start, mid_idx_end))
                    curr_node.children.insert(1, middle)

                    # Left-middle
                    if left.left_median is None and left_count > 1:
                        left_left_split = bisect.bisect_left(times, left_mid, left_idx_start, left_idx_end)
                        left_left_mid = compute_median(times, left_idx_start, left_left_split)
                        left.left_median = left_left_mid

                    if middle.right_median is None:
                        mid_right_split = bisect.bisect_left(times, middle.median, mid_idx_start, mid_idx_end)
                        middle_right_mid = compute_median(times, mid_right_split, mid_idx_end)
                        middle.right_median = middle_right_mid

                    if left.left_median and middle.right_median and left.left_median < middle.right_median:
                        lm_start = left.left_median
                        lm_end = middle.right_median
                        lm_idx_start = bisect.bisect_left(times, lm_start, idx_start, idx_end)
                        lm_idx_end = bisect.bisect_left(times, lm_end, idx_start, idx_end)
                        if lm_idx_end > lm_idx_start:
                            left_middle = create_node(lm_start, lm_end, curr_node.level + 1, is_left_middle=True)
                            left_middle.parents.append(curr_node)
                            queue.append((left_middle, lm_idx_start, lm_idx_end))
                            curr_node.children.insert(1, left_middle)

                    # Right-middle
                    if middle.left_median is None:
                        mid_left_split = bisect.bisect_left(times, middle.median, mid_idx_start, mid_idx_end)
                        middle_left_mid = compute_median(times, mid_idx_start, mid_left_split)
                        middle.left_median = middle_left_mid

                    if right.right_median is None and right_count > 1:
                        right_right_split = bisect.bisect_left(times, right_mid, right_idx_start, right_idx_end)
                        right_right_mid = compute_median(times, right_right_split, right_idx_end)
                        right.right_median = right_right_mid

                    if middle.left_median and right.right_median and middle.left_median < right.right_median:
                        rm_start = middle.left_median
                        rm_end = right.right_median
                        rm_idx_start = bisect.bisect_left(times, rm_start, idx_start, idx_end)
                        rm_idx_end = bisect.bisect_left(times, rm_end, idx_start, idx_end)
                        if rm_idx_end > rm_idx_start:
                            right_middle = create_node(rm_start, rm_end, curr_node.level + 1, is_right_middle=True)
                            right_middle.parents.append(curr_node)
                            queue.append((right_middle, rm_idx_start, rm_idx_end))
                            curr_node.children.insert(-1, right_middle)

        curr_node.children.sort(key=lambda x: x.range_start)

    return root


#=============SRC search dag5==========
def src_search_dag5(node: Optional[DagNode5], q_start: int, q_end: int) -> Optional[DagNode5]:
    if node is None:
        return None

    if node.range_end <= q_start or node.range_start >= q_end:
        return None

    if not (node.range_start <= q_start and node.range_end >= q_end):
        return None

    for child in node.children:
        res = src_search_dag5(child, q_start, q_end)
        if res is not None:
            return res

    return node




# ========== UNIFORM-TIME BASELINE ==========

def make_uniform_times(distinct_count: int, timespan: int) -> List[int]:
    """
    Generate distinct_count uniformly spaced integers in [0, timespan].
    """
    if distinct_count <= 0:
        return []
    if distinct_count > timespan + 1:
        raise ValueError(f"Cannot create {distinct_count} unique integers in [0, {timespan}]")
    if distinct_count == 1:
        return [0]
    step = timespan / (distinct_count - 1)
    vals = [round(i * step) for i in range(distinct_count)]
    vals[0] = 0
    for i in range(1, distinct_count):
        vals[i] = max(vals[i], vals[i - 1] + 1)
    vals[-1] = timespan
    return vals



# ========== THEORETICAL P(k) FROM the Theorem 1 ==========
def theoretical_p(k: int, n: int, s_star: int, c: int) -> float:
    if k < 0:
        return 0.0
    N = 2 ** (n + 1)
    # Raise error for invalid s_star
    if s_star <= 0 or s_star >= N:
        raise ValueError(f"Invalid s_star: {s_star}. Must be in (0, {N}).")
    if s_star == 1:
        return 1.0 if k == 0 else 0.0

    den = N - s_star                        
    if den <= 0:
        return 0.0

    m = math.floor(math.log2(s_star - 1))
    j = n - m

    # Threshold for small vs large (exact integer comparison to avoid float error)
    two_nj1 = 2 ** (n - j + 1)              # 2^{n-j+1}
    is_small = (s_star * (c - 1) <= (c - 2) * two_nj1)

    if is_small:
        # Small-s case (Lemma ldd-small-s)
        if k == 0:
            return (N - (2 ** j) * s_star) / den
        elif 1 <= k <= j:
            return ((2 ** (j - k)) * s_star) / den
    else:
        # Large-s case (Lemma ldd-large-s)
        if k == j:
            return ((c - 2) * two_nj1 - (c - 2) * s_star) / den
        elif 1 <= k <= j - 1:
            return ((c - 2) * (2 ** (n - k)) -
                    (c - 3) * (2 ** (j - k - 1)) * s_star) / den
        elif k == 0:
            return (-(c - 4) * (2 ** n) +
                    (c - 3) * (2 ** (j - 1)) * s_star) / den

    return 0.0                              # For k > j or invalid


# ========== GENERATE BATCH QUERIES ==========

def generate_batch_queries(timespan: int, query_range: int, batch_size: int, seed_offset: int = 0):
    random.seed(SEED + seed_offset)
    queries = []
    for _ in range(batch_size):
        start = random.randint(0, max(0, timespan - query_range))
        end = start + query_range
        queries.append((start, end))
    return queries

# ========== PROCESS RESULTS ==========
# didnt use!

def process_results(records: dict, s: int, queries_per_size: dict, output_dir: str, prefix: str, color: str):
    valid = [r for r in records[s] if r[2] is not None]
    valid_levels = [r[2] for r in valid]

    if not valid:
        print(f"[{prefix}] No valid levels for query_range {s} at {queries_per_size[s]} queries.")
        return

    # Bar plot of returned level distribution
    plt.figure()
    unique_levels, counts = np.unique(valid_levels, return_counts=True)
    total_valid = len(valid_levels)
    emp_probs = counts / total_valid
    plt.bar(unique_levels , emp_probs, width=0.4, label='Empirical', alpha=0.7, color=color)
    plt.xlabel("Returned Level")
    plt.ylabel("Probability")
    plt.title(f"{prefix} Returned Level Distribution (range={s}, {queries_per_size[s]} queries)")
    plt.legend()
    plt.xticks(unique_levels)
    plt.yticks(emp_probs, [f"{p:.3f}" for p in emp_probs], fontsize=6)
    full_output_dir = os.path.join(BASE_OUTPUT_DIR, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    #try:
    plt.savefig(os.path.join(full_output_dir, f"{prefix}_level_dist_s{s}_q{queries_per_size[s]}.png"))
    #except Exception as e:
    #    print(f"Error saving plot: {e}")
    #finally:
    plt.close()

    # Excel file of returned levels
    df = pd.DataFrame(valid, columns=['query start', 'query end', 'returned level'])
    try:
        df.to_csv(os.path.join(full_output_dir, f"{prefix}_levels_s{s}_q{queries_per_size[s]}.csv"), index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Results
    avg_level = np.mean(valid_levels) if valid_levels else 0
    level_counts = dict(zip(unique_levels, counts))
    prob_table = dict(zip(unique_levels, emp_probs))
    print(f"[{prefix}] Results for query range {s}, {queries_per_size[s]} queries:")
    print(f"Average Returned Level: {avg_level:.2f}")

    # Build distribution table
    if valid_levels:
        unique_levels, counts = np.unique(valid_levels, return_counts=True)
        total_valid = len(valid_levels)
        p_emp = counts / total_valid
        avg_level = np.mean(valid_levels) if valid_levels else 0
        dist = pd.DataFrame({
            'level': unique_levels,
            'count': counts,
            'probability': p_emp
        })
        # Add average as a new row
        avg_row = pd.DataFrame({
            'level': ['AVG'],
            'count': [len(valid_levels)],
            'probability': [avg_level]
        })
        dist = pd.concat([dist, avg_row], ignore_index=True)
        dist.to_csv(os.path.join(full_output_dir, f"{prefix}_level_dist_table_s{s}_q{queries_per_size[s]}.csv"), index=False)



# ========== RUN EXPERIMENT ==========
def run_1d_experiment(times: List[int], uniform_times: List[int], query_range: List[int]):
    gowalla_min = min(times)
    gowalla_max = max(times)
    range_end = gowalla_max + 1 # Exclusive upper bound
    timespan = gowalla_max - gowalla_min
    normalized_timespan = timespan # Already normalized in load_gowalla_data
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Compute step based on distinct_count (len(times) == N_FIXED)
    distinct_count = len(times)  # This is N_FIXED=4194304
    step = timespan / (distinct_count - 1) if distinct_count > 1 else 1  # Avoid div-by-zero, though unlikely
    
    # Build 1D-Trees
    print("Building 1D-Tree for Gowalla times...")
    gowalla_tree = build_1dtree(times, gowalla_min, range_end)
    if gowalla_tree is None:
        print("Failed to build Gowalla tree. Skipping.")
        return
    print("Gowalla 1D-Tree built.")
    print("Building 1D-Tree for Uniform times...")
    # Offset uniform times to match Gowalla range
    uniform_times_offset = [gowalla_min + t for t in uniform_times]
    uniform_tree = build_1dtree(uniform_times_offset, gowalla_min, range_end)
    if uniform_tree is None:
        print("Failed to build Uniform tree. Skipping.")
        return
    print("Uniform 1D-Tree built.")

    # Build 1D-DAG3
    print("Building 1D-DAG3 for Gowalla times...")
    gowalla_dag3 = build_1ddag3(times, gowalla_min, range_end)
    if gowalla_dag3 is None:
        print("Failed to build Gowalla dag. Skipping.")
        return
    print("Gowalla 1D-DAG3 built.")
    print("Building 1D-DAG3 for Uniform times...")
    #uniform_times_offset = [gowalla_min + t for t in uniform_times]
    uniform_dag3 = build_1ddag3(uniform_times_offset, gowalla_min, range_end)
    if uniform_dag3 is None:
        print("Failed to build Uniform dag. Skipping.")
        return
    print("Uniform 1D-DAG3 built.")

    # Build 1D-DAG5
    print("Building 1D-DAG5 for Gowalla times...")
    gowalla_dag5 = build_1ddag5(times, gowalla_min, range_end)
    if gowalla_dag5 is None:
        print("Failed to build Gowalla dag5. Skipping.")
        return
    print("Gowalla 1D-DAG5 built.")
    print("Building 1D-DAG5 for Uniform times...")
    #uniform_times_offset = [gowalla_min + t for t in uniform_times]
    uniform_dag5 = build_1ddag5(uniform_times_offset, gowalla_min, range_end)
    if uniform_dag5 is None:
        print("Failed to build Uniform dag5. Skipping.")
        return
    print("Uniform 1D-DAG5 built.")

    # Experiment loop - per query_range independent
    records_gowalla_tree = {s: [] for s in query_range}
    records_uniform_tree = {s: [] for s in query_range}
    records_gowalla_dag3 = {s: [] for s in query_range}
    records_uniform_dag3 = {s: [] for s in query_range}
    records_gowalla_dag5 = {s: [] for s in query_range}
    records_uniform_dag5 = {s: [] for s in query_range}
    for s in query_range:
        print(f"Processing query range {s}...")

        # Compute s_star for the query range
        s_star = round(s / step)  # e.g., for s=60, ~5
        if s_star < 1:
            print(f"Warning: Computed s_star={s_star} for s={s} is invalid; skipping or defaulting.")
            continue  # Or set s_star=1
        print(f"Computed s_star for query_range={s}: {s_star}")

        queries_count = 0
        seed_offset = 0
        # Stabilization state
        g_dag3_stable = False
        g_dag5_stable = False
        u_dag3_stable = False
        u_dag5_stable = False
        g_dag3_stable_count = None
        g_dag5_stable_count = None
        u_dag3_stable_count = None
        u_dag5_stable_count = None
        # Empirical distribution trackers
        gowalla_prev_dist_dag3 = None # previous snapshot
        uniform_prev_dist_dag3 = None
        gowalla_prev_dist_dag5 = None  # for 5-DAG snapshot
        uniform_prev_dist_dag5 = None  
        gowalla_penult_dist_dag3 = None # penultimate snapshot
        uniform_penult_dist_dag3 = None
        gowalla_penult_dist_dag5 = None # penultimate snapshot
        uniform_penult_dist_dag5 = None

        while queries_count < MAX_QUERIES_PER_SIZE and not (g_dag3_stable and g_dag5_stable and u_dag3_stable and u_dag5_stable):
            batch_size = INITIAL_QUERIES_PER_SIZE if queries_count == 0 else ADD_QUERIES
            qs = generate_batch_queries(normalized_timespan, s, batch_size, seed_offset)
            for q_start, q_end in qs:
                # TREE
                res_g_tree = src_search_tree(gowalla_tree, q_start, q_end)
                res_u_tree = src_search_tree(uniform_tree, q_start, q_end)
                records_gowalla_tree[s].append((q_start, q_end, res_g_tree.level if res_g_tree else None))
                records_uniform_tree[s].append((q_start, q_end, res_u_tree.level if res_u_tree else None))
                # DAG3
                res_g_dag3 = src_search_dag3(gowalla_dag3, q_start, q_end)
                res_u_dag3 = src_search_dag3(uniform_dag3, q_start, q_end)
                records_gowalla_dag3[s].append((q_start, q_end, res_g_dag3.level if res_g_dag3 else None))
                records_uniform_dag3[s].append((q_start, q_end, res_u_dag3.level if res_u_dag3 else None))
                # DAG5
                res_g_dag5 = src_search_dag5(gowalla_dag5, q_start, q_end)
                res_u_dag5 = src_search_dag5(uniform_dag5, q_start, q_end)
                records_gowalla_dag5[s].append((q_start, q_end, res_g_dag5.level if res_g_dag5 else None))
                records_uniform_dag5[s].append((q_start, q_end, res_u_dag5.level if res_u_dag5 else None))
            queries_count += batch_size
            seed_offset += batch_size
            # Optional per-batch outputs
            tree_output_dir = "1d-tree results"
            dag3_output_dir = "1d-dag3 results"
            dag5_output_dir = "1d-dag5 results"
            os.makedirs(tree_output_dir, exist_ok=True)
            os.makedirs(dag3_output_dir, exist_ok=True)
            os.makedirs(dag5_output_dir, exist_ok=True)

            tmp_counts = {s: queries_count}
            # Tree
            process_results(records_gowalla_tree, s, tmp_counts, tree_output_dir, "gowalla_tree", "blue")
            process_results(records_uniform_tree, s, tmp_counts, tree_output_dir, "uniform_tree", "green")
            # DAG3
            process_results(records_gowalla_dag3, s, tmp_counts, dag3_output_dir, "gowalla_dag3", "blue")
            process_results(records_uniform_dag3, s, tmp_counts, dag3_output_dir, "uniform_dag3", "green")
            # DAG5
            process_results(records_gowalla_dag5, s, tmp_counts, dag5_output_dir, "gowalla_dag5", "blue")
            process_results(records_uniform_dag5, s, tmp_counts, dag5_output_dir, "uniform_dag5", "green")

            # Compute current LDD dists
            gowalla_current_dist_dag3 = compute_ldd(records_gowalla_tree[s], records_gowalla_dag3[s])
            uniform_current_dist_dag3 = compute_ldd(records_uniform_tree[s], records_uniform_dag3[s])
            gowalla_current_dist_dag5 = compute_ldd(records_gowalla_tree[s], records_gowalla_dag5[s])
            uniform_current_dist_dag5 = compute_ldd(records_uniform_tree[s], records_uniform_dag5[s])


            # For 3-DAG (tree vs 3-DAG)
            l2_g_dag3 = float("nan")
            l2_u_dag3 = float("nan")
            if queries_count > INITIAL_QUERIES_PER_SIZE:
                if gowalla_prev_dist_dag3 is not None:
                    prev_dict_g_dag3 = dict(zip(gowalla_prev_dist_dag3[0], gowalla_prev_dist_dag3[2]))
                    curr_dict_g_dag3 = dict(zip(gowalla_current_dist_dag3[0], gowalla_current_dist_dag3[2]))
                    l2_g_dag3 = compute_l2_norm(prev_dict_g_dag3, curr_dict_g_dag3)
                if uniform_prev_dist_dag3 is not None:
                    prev_dict_u_dag3 = dict(zip(uniform_prev_dist_dag3[0], uniform_prev_dist_dag3[2]))
                    curr_dict_u_dag3 = dict(zip(uniform_current_dist_dag3[0], uniform_current_dist_dag3[2]))
                    l2_u_dag3 = compute_l2_norm(prev_dict_u_dag3, curr_dict_u_dag3)

            # For 5-DAG (tree vs 5-DAG)
            l2_g_dag5 = float("nan")
            l2_u_dag5 = float("nan")
            if queries_count > INITIAL_QUERIES_PER_SIZE:
                if gowalla_prev_dist_dag5 is not None:
                    prev_dict_g_dag5 = dict(zip(gowalla_prev_dist_dag5[0], gowalla_prev_dist_dag5[2]))
                    curr_dict_g_dag5 = dict(zip(gowalla_current_dist_dag5[0], gowalla_current_dist_dag5[2]))
                    l2_g_dag5 = compute_l2_norm(prev_dict_g_dag5, curr_dict_g_dag5)
                if uniform_prev_dist_dag5 is not None:
                    prev_dict_u_dag5 = dict(zip(uniform_prev_dist_dag5[0], uniform_prev_dist_dag5[2]))
                    curr_dict_u_dag5 = dict(zip(uniform_current_dist_dag5[0], uniform_current_dist_dag5[2]))
                    l2_u_dag5 = compute_l2_norm(prev_dict_u_dag5, curr_dict_u_dag5)


            # --- store penultimate BEFORE overwriting prev ---
            gowalla_penult_dist_dag3 = gowalla_prev_dist_dag3 if gowalla_prev_dist_dag3 else None
            uniform_penult_dist_dag3 = uniform_prev_dist_dag3 if uniform_prev_dist_dag3 else None
            gowalla_penult_dist_dag5 = gowalla_prev_dist_dag5 if gowalla_prev_dist_dag5 else None
            uniform_penult_dist_dag5 = uniform_prev_dist_dag5 if uniform_prev_dist_dag5 else None

            if queries_count > INITIAL_QUERIES_PER_SIZE:
                if not g_dag3_stable and gowalla_prev_dist_dag3 is not None and l2_g_dag3 < EPSILON:
                    g_dag3_stable = True
                    g_dag3_stable_count = queries_count
                    print(f"Gowalla DAG3 hit stable at {queries_count} for s={s}, L2: {l2_g_dag3:.6f}")
                    save_final_ldd(*gowalla_current_dist_dag3, s, queries_count, "gowalla_dag3", 
                                   l2_g_dag3, s_star=s_star, suffix="_stable")

                if not g_dag5_stable and gowalla_prev_dist_dag5 is not None and l2_g_dag5 < EPSILON:
                    g_dag5_stable = True
                    g_dag5_stable_count = queries_count
                    print(f"Gowalla DAG5 hit stable at {queries_count} for s={s}, L2: {l2_g_dag5:.6f}")
                    save_final_ldd(*gowalla_current_dist_dag5, s, queries_count, "gowalla_dag5", 
                                   l2_g_dag5, s_star=s_star, suffix="_stable")

                if not u_dag3_stable and uniform_prev_dist_dag3 is not None and l2_u_dag3 < EPSILON:
                    u_dag3_stable = True
                    u_dag3_stable_count = queries_count
                    print(f"Uniform DAG3 hit stable at {queries_count} for s={s}, L2: {l2_u_dag3:.6f}")
                    save_final_ldd(*uniform_current_dist_dag3, s, queries_count, "uniform_dag3", 
                                   l2_u_dag3, s_star=s_star, suffix="_stable")

                if not u_dag5_stable and uniform_prev_dist_dag5 is not None and l2_u_dag5 < EPSILON:
                    u_dag5_stable = True
                    u_dag5_stable_count = queries_count
                    print(f"Uniform DAG5 hit stable at {queries_count} for s={s}, L2: {l2_u_dag5:.6f}")
                    save_final_ldd(*uniform_current_dist_dag5, s, queries_count, "uniform_dag5", 
                                   l2_u_dag5, s_star=s_star, suffix="_stable")

                # One clean full-state record, no mess
                if g_dag3_stable and g_dag5_stable and u_dag3_stable and u_dag5_stable:
                    print(f" All paths locked in at {queries_count} for s={s}")
                    # keep the four saves again—now timestamped as aligned
                    save_final_ldd(*gowalla_current_dist_dag3, s, queries_count, "gowalla_dag3", 
                                   l2_g_dag3, s_star=s_star, suffix="_all_stable")
                    save_final_ldd(*gowalla_current_dist_dag5, s, queries_count, "gowalla_dag5", 
                                   l2_g_dag5, s_star=s_star, suffix="_all_stable")
                    save_final_ldd(*uniform_current_dist_dag3, s, queries_count, "uniform_dag3", 
                                   l2_u_dag3, s_star=s_star, suffix="_all_stable")
                    save_final_ldd(*uniform_current_dist_dag5, s, queries_count, "uniform_dag5", 
                                   l2_u_dag5, s_star=s_star, suffix="_all_stable")


            # --- now update prev to current (ONE place only) ---
            gowalla_prev_dist_dag3 = gowalla_current_dist_dag3
            uniform_prev_dist_dag3 = uniform_current_dist_dag3
            gowalla_prev_dist_dag5 = gowalla_current_dist_dag5
            uniform_prev_dist_dag5 = uniform_current_dist_dag5
        


        if g_dag3_stable and g_dag5_stable and u_dag3_stable and u_dag5_stable:
          print(f"All paths stabilized at {queries_count} queries; running additional 120000 queries in batches of {ADD_QUERIES}.")
          extra_queries_target = 120000
          extra_queries_count = 0
          while extra_queries_count < extra_queries_target and queries_count < MAX_QUERIES_PER_SIZE:
              batch_size = min(ADD_QUERIES, extra_queries_target - extra_queries_count)
              qs = generate_batch_queries(normalized_timespan, s, batch_size, seed_offset)
              for q_start, q_end in qs:
                  # TREE queries (append to records)
                  res_g_tree = src_search_tree(gowalla_tree, q_start, q_end)
                  res_u_tree = src_search_tree(uniform_tree, q_start, q_end)
                  records_gowalla_tree[s].append((q_start, q_end, res_g_tree.level if res_g_tree else None))
                  records_uniform_tree[s].append((q_start, q_end, res_u_tree.level if res_u_tree else None))
                  # DAG3 queries
                  res_g_dag3 = src_search_dag3(gowalla_dag3, q_start, q_end)
                  res_u_dag3 = src_search_dag3(uniform_dag3, q_start, q_end)
                  records_gowalla_dag3[s].append((q_start, q_end, res_g_dag3.level if res_g_dag3 else None))
                  records_uniform_dag3[s].append((q_start, q_end, res_u_dag3.level if res_u_dag3 else None))
                  # DAG5 queries
                  res_g_dag5 = src_search_dag5(gowalla_dag5, q_start, q_end)
                  res_u_dag5 = src_search_dag5(uniform_dag5, q_start, q_end)
                  records_gowalla_dag5[s].append((q_start, q_end, res_g_dag5.level if res_g_dag5 else None))
                  records_uniform_dag5[s].append((q_start, q_end, res_u_dag5.level if res_u_dag5 else None))
              queries_count += batch_size
              extra_queries_count += batch_size
              seed_offset += batch_size
              
              # Save level dist plots during extras (with "_extra" prefix for distinction)
              tmp_counts = {s: queries_count}
              # Tree
              process_results(records_gowalla_tree, s, tmp_counts, tree_output_dir, "gowalla_tree_extra", "blue")
              process_results(records_uniform_tree, s, tmp_counts, tree_output_dir, "uniform_tree_extra", "green")
              # DAG3
              process_results(records_gowalla_dag3, s, tmp_counts, dag3_output_dir, "gowalla_dag3_extra", "blue")
              process_results(records_uniform_dag3, s, tmp_counts, dag3_output_dir, "uniform_dag3_extra", "green")
              # DAG5
              process_results(records_gowalla_dag5, s, tmp_counts, dag5_output_dir, "gowalla_dag5_extra", "blue")
              process_results(records_uniform_dag5, s, tmp_counts, dag5_output_dir, "uniform_dag5_extra", "green")
              
              # Compute current LDD and check L2 during extras
              gowalla_current_dist_dag3 = compute_ldd(records_gowalla_tree[s], records_gowalla_dag3[s])
              uniform_current_dist_dag3 = compute_ldd(records_uniform_tree[s], records_uniform_dag3[s])
              gowalla_current_dist_dag5 = compute_ldd(records_gowalla_tree[s], records_gowalla_dag5[s])
              uniform_current_dist_dag5 = compute_ldd(records_uniform_tree[s], records_uniform_dag5[s])
              
              # For DAG3
              l2_g_dag3 = float("nan")
              l2_u_dag3 = float("nan")
              if gowalla_prev_dist_dag3 is not None:
                  prev_dict_g_dag3 = dict(zip(gowalla_prev_dist_dag3[0], gowalla_prev_dist_dag3[2]))
                  curr_dict_g_dag3 = dict(zip(gowalla_current_dist_dag3[0], gowalla_current_dist_dag3[2]))
                  l2_g_dag3 = compute_l2_norm(prev_dict_g_dag3, curr_dict_g_dag3)
              if uniform_prev_dist_dag3 is not None:
                  prev_dict_u_dag3 = dict(zip(uniform_prev_dist_dag3[0], uniform_prev_dist_dag3[2]))
                  curr_dict_u_dag3 = dict(zip(uniform_current_dist_dag3[0], uniform_current_dist_dag3[2]))
                  l2_u_dag3 = compute_l2_norm(prev_dict_u_dag3, curr_dict_u_dag3)

              # For DAG5
              l2_g_dag5 = float("nan")
              l2_u_dag5 = float("nan")
              if gowalla_prev_dist_dag5 is not None:
                  prev_dict_g_dag5 = dict(zip(gowalla_prev_dist_dag5[0], gowalla_prev_dist_dag5[2]))
                  curr_dict_g_dag5 = dict(zip(gowalla_current_dist_dag5[0], gowalla_current_dist_dag5[2]))
                  l2_g_dag5 = compute_l2_norm(prev_dict_g_dag5, curr_dict_g_dag5)
              if uniform_prev_dist_dag5 is not None:
                  prev_dict_u_dag5 = dict(zip(uniform_prev_dist_dag5[0], uniform_prev_dist_dag5[2]))
                  curr_dict_u_dag5 = dict(zip(uniform_current_dist_dag5[0], uniform_current_dist_dag5[2]))
                  l2_u_dag5 = compute_l2_norm(prev_dict_u_dag5, curr_dict_u_dag5)
              
              # Save LDD snapshots during extras with "_extra" suffix
              save_final_ldd(gowalla_current_dist_dag3[0], gowalla_current_dist_dag3[1], gowalla_current_dist_dag3[2], s, queries_count, "gowalla_dag3", l2_g_dag3, s_star=s_star, suffix="_extra")
              save_final_ldd(uniform_current_dist_dag3[0], uniform_current_dist_dag3[1], uniform_current_dist_dag3[2], s, queries_count, "uniform_dag3", l2_u_dag3, s_star=s_star, suffix="_extra")
              save_final_ldd(gowalla_current_dist_dag5[0], gowalla_current_dist_dag5[1], gowalla_current_dist_dag5[2], s, queries_count, "gowalla_dag5", l2_g_dag5, s_star=s_star, suffix="_extra")
              save_final_ldd(uniform_current_dist_dag5[0], uniform_current_dist_dag5[1], uniform_current_dist_dag5[2], s, queries_count, "uniform_dag5", l2_u_dag5, s_star=s_star, suffix="_extra")


              # Update prev for next L2 check
              gowalla_prev_dist_dag3 = gowalla_current_dist_dag3
              uniform_prev_dist_dag3 = uniform_current_dist_dag3
              gowalla_prev_dist_dag5 = gowalla_current_dist_dag5
              uniform_prev_dist_dag5 = uniform_current_dist_dag5
        

        # Final snapshot for this s (both stabilized or maxed out)
        print(f"Saving both at final count {queries_count} for s={s}")
        gowalla_final_dist_dag3 = compute_ldd(records_gowalla_tree[s], records_gowalla_dag3[s])
        uniform_final_dist_dag3 = compute_ldd(records_uniform_tree[s], records_uniform_dag3[s])
        gowalla_final_dist_dag5 = compute_ldd(records_gowalla_tree[s], records_gowalla_dag5[s])
        uniform_final_dist_dag5 = compute_ldd(records_uniform_tree[s], records_uniform_dag5[s])

        # L2 (final) vs penultimate (NOT vs prev if prev==final)
        if gowalla_penult_dist_dag3 is not None:
            penult_dict_g_dag3 = dict(zip(gowalla_penult_dist_dag3[0], gowalla_penult_dist_dag3[2]))
            final_dict_g_dag3 = dict(zip(gowalla_final_dist_dag3[0], gowalla_final_dist_dag3[2]))
            l2_g_final_dag3 = compute_l2_norm(penult_dict_g_dag3, final_dict_g_dag3)
        else:
            l2_g_final_dag3 = float("nan")

        if uniform_penult_dist_dag3 is not None:
            penult_dict_u_dag3 = dict(zip(uniform_penult_dist_dag3[0], uniform_penult_dist_dag3[2]))
            final_dict_u_dag3 = dict(zip(uniform_final_dist_dag3[0], uniform_final_dist_dag3[2]))
            l2_u_final_dag3 = compute_l2_norm(penult_dict_u_dag3, final_dict_u_dag3)
        else:
            l2_u_final_dag3 = float("nan")
        
        if gowalla_penult_dist_dag5 is not None:
            penult_dict_g_dag5 = dict(zip(gowalla_penult_dist_dag5[0], gowalla_penult_dist_dag5[2]))
            final_dict_g_dag5 = dict(zip(gowalla_final_dist_dag5[0], gowalla_final_dist_dag5[2]))
            l2_g_final_dag5 = compute_l2_norm(penult_dict_g_dag5, final_dict_g_dag5)
        else:
            l2_g_final_dag5 = float("nan")

        if uniform_penult_dist_dag5 is not None:
            penult_dict_u_dag5 = dict(zip(uniform_penult_dist_dag5[0], uniform_penult_dist_dag5[2]))
            final_dict_u_dag5 = dict(zip(uniform_final_dist_dag5[0], uniform_final_dist_dag5[2]))
            l2_u_final_dag5 = compute_l2_norm(penult_dict_u_dag5, final_dict_u_dag5)
        else:
            l2_u_final_dag5 = float("nan")
 
        save_final_ldd(gowalla_final_dist_dag3[0], gowalla_final_dist_dag3[1], gowalla_final_dist_dag3[2], s, queries_count, "gowalla_dag3", l2_g_final_dag3, s_star=s_star, suffix="_final")
        save_final_ldd(uniform_final_dist_dag3[0], uniform_final_dist_dag3[1], uniform_final_dist_dag3[2], s, queries_count, "uniform_dag3", l2_u_final_dag3, s_star=s_star, suffix="_final")
        save_final_ldd(gowalla_final_dist_dag5[0], gowalla_final_dist_dag5[1], gowalla_final_dist_dag5[2], s, queries_count, "gowalla_dag5", l2_g_final_dag5, s_star=s_star, suffix="_final")
        save_final_ldd(uniform_final_dist_dag5[0], uniform_final_dist_dag5[1], uniform_final_dist_dag5[2], s, queries_count, "uniform_dag5", l2_u_final_dag5, s_star=s_star, suffix="_final")



        # If max reached without stabilization, still persist a clear note
        if not (g_dag3_stable and g_dag5_stable):
            print(f"Gowalla did not fully stabilize (DAG3: {g_dag3_stable}, DAG5: {g_dag5_stable}) for s={s}; using {queries_count} queries")
        if not (u_dag3_stable and u_dag5_stable):
            print(f"Uniform did not fully stabilize (DAG3: {u_dag3_stable}, DAG5: {u_dag5_stable}) for s={s}; using {queries_count} queries")
    # Generate final L2 CSV
    pd.DataFrame(l2_results).to_csv(os.path.join(BASE_OUTPUT_DIR, f"seed{SEED}_all_l2_norms.csv"))

    

# Helper to compute LDD dict {k: p(k)}
def compute_ldd(
    tree_records: List[Tuple[int, int, Optional[int]]],
    dag_records: List[Tuple[int, int, Optional[int]]]
) -> tuple:
    m = min(len(tree_records), len(dag_records))
    ldds = []
    for i in range(m):
        ts, te, tl = tree_records[i]
        ds, de, dl = dag_records[i]
        if tl is None or dl is None:
            continue
        diff = dl - tl
        if diff >= 0:
            ldds.append(diff)

    if not ldds:
        return np.array([]), np.array([]), np.array([])

    unique_k, counts_k = np.unique(ldds, return_counts=True)
    emp_probs = counts_k / len(ldds)
    return unique_k, counts_k, emp_probs



 # Helper to compute L2 norm between two dist dicts
def compute_l2_norm(dist1: dict, dist2: dict) -> float:
    unique_keys = set(dist1.keys()) | set(dist2.keys())
    sum_sq = 0.0
    for k in unique_keys:
        p1 = dist1.get(k, 0.0)
        p2 = dist2.get(k, 0.0)
        sum_sq += (p1 - p2) ** 2
    return math.sqrt(sum_sq)


# Helper to get theoretical dist {k: p(k)} – NOW TAKES c
def get_theoretical_dist(s_star: int, c: int) -> dict:
    dist = {}
    expected_k = 0.0
    expected_exp_k = 0.0

    if s_star <= 1:
        dist[0] = 1.0
        return {'dist': dist, 'expected_k': 0.0, 'expected_exp_k': 1.0}

    N = 2 ** (n_FIXED + 1)
    if s_star >= N:
        return {'dist': dist, 'expected_k': 0.0, 'expected_exp_k': 0.0}

    m = math.floor(math.log2(s_star - 1))
    j = n_FIXED - m
    max_k = j  # 0 to j inclusive (higher k are 0)

    for k in range(max_k + 1):
        p = theoretical_p(k, n_FIXED, s_star, c)
        if p > 0:  # Clamp tiny negative floats due to precision
            dist[k] = max(p, 0.0)
            expected_k += k * dist[k]
            expected_exp_k += (2 ** k) * dist[k]

    return {'dist': dist, 'expected_k': expected_k, 'expected_exp_k': expected_exp_k}

# ==================== NEW: FIND BEST s* ====================
def find_best_s_star(emp_dist: dict, c: int, nominal_s_star: int, 
                     search_range: int = 50) -> dict:
    N = 2 ** (n_FIXED + 1)
    candidates = list(range(max(1, nominal_s_star - search_range),
                           min(N-1, nominal_s_star + search_range + 1)))
    
    results = []
    best = {'s_star': nominal_s_star, 'l2': float('inf'), 'dist': None,
            'E_k': 0.0, 'E_2k': 0.0}
    
    for s_star in candidates:
        theo_data = get_theoretical_dist(s_star, c)
        l2 = compute_l2_norm(emp_dist, theo_data['dist'])
        
        results.append({
            'candidate_s_star': s_star,
            'L2': l2,
            'E_k_theo': theo_data['expected_k'],
            'E_2k_theo': theo_data['expected_exp_k']
        })
        
        if l2 < best['l2']:
            best = {
                's_star': s_star,
                'l2': l2,
                'dist': theo_data['dist'],
                'E_k': theo_data['expected_k'],
                'E_2k': theo_data['expected_exp_k']
            }
    
    # Nominal stats for comparison
    nominal_data = get_theoretical_dist(nominal_s_star, c)
    nominal_L2 = compute_l2_norm(emp_dist, nominal_data['dist'])
    
    return {
        'best_s_star': best['s_star'],
        'min_L2': best['l2'],
        'nominal_s_star': nominal_s_star,
        'nominal_L2': nominal_L2,
        'nominal_E_k': nominal_data['expected_k'],
        'nominal_E_2k': nominal_data['expected_exp_k'],
        'best_E_k': best['E_k'],
        'best_E_2k': best['E_2k'],
        'improvement_L2': nominal_L2 - best['l2'],
        'improvement_E_k': nominal_data['expected_k'] - best['E_k'],
        'improvement_E_2k': nominal_data['expected_exp_k'] - best['E_2k'],
        'all_candidates': pd.DataFrame(results),
        'best_theo_dist': best['dist']
    }


# Helper to save final LDD (plot, CSV, Excel, L2 norms)
def save_final_ldd(unique_k: np.ndarray, counts_k: np.ndarray, emp_probs: np.ndarray, 
                   s: int, queries_count: int, prefix: str, l2_emp: float, 
                   s_star: int, suffix: str = ''):
    
    ldd_output_dir = f"seed{SEED}_ldd_size_{s}"
    full_ldd_dir = os.path.join(BASE_OUTPUT_DIR, ldd_output_dir)
    os.makedirs(full_ldd_dir, exist_ok=True)

    # === YOUR ORIGINAL CODE STARTS HERE (unchanged) ===
    max_k = max(unique_k) if len(unique_k) > 0 else 0
    emp_dist = dict(zip(unique_k, emp_probs))

    c = 5 if 'dag5' in prefix.lower() else 3

    # === NEW: FIND BEST s* (this is the only new part) ===
    search_result = find_best_s_star(
        emp_dist=emp_dist,
        c=c,
        nominal_s_star=s_star,
        search_range=20
    )
    best_s_star = search_result['best_s_star']
    best_theo_dist = search_result['best_theo_dist']
    best_E_k = search_result['best_E_k']
    best_E_2k = search_result['best_E_2k']

    # === CONTINUE WITH YOUR ORIGINAL CODE, but use BEST values ===
    L2_emp_theo = search_result['min_L2']          # now best, not nominal

    all_k = sorted(set(unique_k) | set(best_theo_dist.keys()))
    emp_probs_dict = dict(zip(unique_k, emp_probs))
    counts_dict = dict(zip(unique_k, counts_k))
    emp_p = [emp_probs_dict.get(k, 0.0) for k in all_k]
    counts = [counts_dict.get(k, 0) for k in all_k]
    theo_p = [best_theo_dist.get(k, 0.0) for k in all_k]

    expected_k_emp = sum(k * emp_probs_dict.get(k, 0.0) for k in all_k)
    expected_exp_k_emp = sum((2 ** k) * emp_probs_dict.get(k, 0.0) for k in all_k)

    # Plot with BEST s*
    plt.figure()
    plt.bar(all_k, emp_p, width=0.4, label='Empirical', alpha=0.7, color='purple')
    plt.xlabel("Level Difference (k = level_DAG - level_TREE)")
    plt.ylabel("Probability")
    plt.title(f"{prefix} Final Level Difference Distribution (s={s}, {queries_count} queries, best s*={best_s_star}, c={c})")
    plt.legend()

    k_theo = np.array(sorted(best_theo_dist.keys()))
    p_theo = np.array([best_theo_dist[kk] for kk in k_theo])
    plt.plot(k_theo, p_theo, color='red', label=f'Theoretical (best s*={best_s_star}, c={c})', marker='o', linestyle='-')
    plt.legend()
    plt.xticks(all_k)
    plt.savefig(os.path.join(full_ldd_dir, f"{prefix}_final_ldd_plot_s{s}_q{queries_count}{suffix}_best_sstar{best_s_star}_c{c}.png"))
    plt.close()

    # Table (uses BEST theoretical)
    dist_df = pd.DataFrame({
        'k': all_k,
        'count': counts,
        'max_k_observed': [max_k] * len(all_k),
        'empirical_p(k)': emp_p,
        'theoretical_p(k)': theo_p,
        'difference': [emp - theo for emp, theo in zip(emp_p, theo_p)],
    })
    dist_df.to_excel(os.path.join(full_ldd_dir, f"{prefix}_final_ldd_dist_table_s{s}_q{queries_count}{suffix}_best_sstar{best_s_star}_c{c}.xlsx"), index=False)

    # Save L2 norms (now with BEST values)
    l2_results.append({
        'query_range': s,
        'dataset': prefix,
        'queries_count': queries_count,
        'max_k': max_k,
        'L2_emp': l2_emp,
        'L2_emp_theo': L2_emp_theo,           # best
        'expected_exp_k_emp': expected_exp_k_emp,
        'expected_k_emp': expected_k_emp,
        'expected_k_theo': best_E_k,          # best
        'expected_exp_k_theo': best_E_2k,     # best
        'suffix': suffix,
        's_star': best_s_star                 # best
    })

    with open(os.path.join(full_ldd_dir, f"{prefix}_l2_norms_s{s}_q{queries_count}{suffix}_best_sstar{best_s_star}.txt"), 'w') as f:
        f.write(f"L2 norm between last two empirical distributions: {l2_emp:.6f}\n")
        f.write(f"L2 norm between final empirical and best theoretical: {L2_emp_theo:.6f}\n")
        f.write(f"Expected level difference (emp): {expected_k_emp:.6f}\n")
        f.write(f"Expected exponential ratio (emp): {expected_exp_k_emp:.6f}\n")
        f.write(f"Expected level difference (best theo): {best_E_k:.6f}\n")
        f.write(f"Expected exponential ratio (best theo): {best_E_2k:.6f}\n")

    # === NEW: Collect row for the big final table ===
    final_table_rows.append({
        'query_length': s,
        'nominal_s_star': search_result['nominal_s_star'],
        'nominal_L2': search_result['nominal_L2'],
        'nominal_E_k': search_result['nominal_E_k'],
        'nominal_E_2k': search_result['nominal_E_2k'],
        'best_s_star': best_s_star,
        'best_L2': L2_emp_theo,
        'best_E_k': best_E_k,
        'best_E_2k': best_E_2k,
        'delta_L2': search_result['improvement_L2'],
        'delta_E_k': search_result['improvement_E_k'],
        'delta_E_2k': search_result['improvement_E_2k'],
        'dataset': prefix
    })




if __name__ == "__main__":
    # Comment out the following lines to skip empirical experiments:
    print("Loading Gowalla data...")
    times = load_gowalla_data(GOWALLA_FILE)
    if len(times) < N_FIXED:
        print(f"Warning: Only {len(times)} distinct times available, using all.")
    times = times[:N_FIXED]
    uniform_times = make_uniform_times(N_FIXED, max(times) - min(times))
    print(f"Uniform times generated with range [{min(uniform_times)}, {max(uniform_times)}]")
    print(f"Number of distinct times: {len(times)}")
    run_1d_experiment(times, uniform_times, QUERY_RANGES)
    # Save the big comparative table
    if final_table_rows:
        final_df = pd.DataFrame(final_table_rows)
        final_df.to_excel(os.path.join(BASE_OUTPUT_DIR, f"seed{SEED}_gowalla_best_sstar_table.xlsx"), index=False)
        print(f"Saved final table: seed{SEED}_gowalla_best_sstar_table.xlsx")

    l2_df = pd.DataFrame(l2_results)

    # ===================================================================
    # COMBINED WHISKER PLOT: Expected Level Difference 
    # 3-DAG (blue) vs 5-DAG (red) side-by-side
    # ===================================================================
    plt.figure(figsize=(10, 6))

    query_ranges = [60, 3600, 86400, 604800]
    x_positions = np.arange(len(query_ranges))   # 0, 1, 2, 3
    width = 0.35

    data_3dag = []
    data_5dag = []
    theo_3dag = []
    theo_5dag = []

    for s in query_ranges:
        # 3-DAG (blue)
        ds_df3 = l2_df[l2_df['dataset'] == 'gowalla_dag3']
        qr_df3 = ds_df3[ds_df3['query_range'] == s].sort_values('queries_count')
        data_3dag.append(qr_df3[qr_df3['queries_count'] >= 5000]['expected_k_emp'].tolist())
        theo_3dag.append(qr_df3.iloc[0]['expected_k_theo'] if not qr_df3.empty else None)

        # 5-DAG (red)
        ds_df5 = l2_df[l2_df['dataset'] == 'gowalla_dag5']
        qr_df5 = ds_df5[ds_df5['query_range'] == s].sort_values('queries_count')
        data_5dag.append(qr_df5[qr_df5['queries_count'] >= 5000]['expected_k_emp'].tolist())
        theo_5dag.append(qr_df5.iloc[0]['expected_k_theo'] if not qr_df5.empty else None)

    # 3-DAG boxes (blue)
    plt.boxplot(data_3dag, positions=x_positions - width/2, widths=width,
                patch_artist=True, whis=[0, 100],
                boxprops=dict(facecolor='lightblue', color='blue', linewidth=2),
                whiskerprops=dict(color='blue', linewidth=2),
                capprops=dict(color='blue', linewidth=2),
                medianprops=dict(color='darkblue', linewidth=2.5))

    # 5-DAG boxes (red)
    plt.boxplot(data_5dag, positions=x_positions + width/2, widths=width,
                patch_artist=True, whis=[0, 100],
                boxprops=dict(facecolor='lightcoral', color='red', linewidth=2),
                whiskerprops=dict(color='red', linewidth=2),
                capprops=dict(color='red', linewidth=2),
                medianprops=dict(color='darkred', linewidth=2.5))

    # Theoretical dots
    for i, (t3, t5) in enumerate(zip(theo_3dag, theo_5dag)):
        if t3 is not None:
            plt.plot(x_positions[i] - width/2, t3, 'o', color='blue', markersize=10, 
                     label='3-DAG vs. 1D-Tree Theoretical' if i == 0 else None)
        if t5 is not None:
            plt.plot(x_positions[i] + width/2, t5, 'o', color='red', markersize=10, 
                     label='5-DAG vs. 1D-Tree Theoretical' if i == 0 else None)

    plt.xticks(x_positions, query_ranges)
    plt.xlabel("Query Lengths", fontsize=14)
    plt.ylabel("Expected Level Difference", fontsize=14)
    plt.title("Gowalla: 3-DAG vs 5-DAG (Expected Level Difference)", fontsize=15)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "gowalla_combined_level_diff_whisker.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # ===================================================================
    # COMBINED WHISKER PLOT: Gowalla 3-DAG (blue) + 5-DAG (red) side-by-side
    # ===================================================================
    plt.figure(figsize=(10, 6))

    query_ranges = [60, 3600, 86400, 604800]
    x_positions = np.arange(len(query_ranges))   # 0, 1, 2, 3
    width = 0.35

    data_3dag = []
    data_5dag = []
    theo_3dag = []
    theo_5dag = []

    for s in query_ranges:
        # 3-DAG (blue)
        ds_df3 = l2_df[l2_df['dataset'] == 'gowalla_dag3']
        qr_df3 = ds_df3[ds_df3['query_range'] == s].sort_values('queries_count')
        data_3dag.append(qr_df3[qr_df3['queries_count'] >= 5000]['expected_exp_k_emp'].tolist())
        theo_3dag.append(qr_df3.iloc[0]['expected_exp_k_theo'] if not qr_df3.empty else None)

        # 5-DAG (red)
        ds_df5 = l2_df[l2_df['dataset'] == 'gowalla_dag5']
        qr_df5 = ds_df5[ds_df5['query_range'] == s].sort_values('queries_count')
        data_5dag.append(qr_df5[qr_df5['queries_count'] >= 5000]['expected_exp_k_emp'].tolist())
        theo_5dag.append(qr_df5.iloc[0]['expected_exp_k_theo'] if not qr_df5.empty else None)

    # 3-DAG boxes (blue)
    plt.boxplot(data_3dag, positions=x_positions - width/2, widths=width,
                patch_artist=True, whis=[0, 100],
                boxprops=dict(facecolor='lightblue', color='blue', linewidth=2),
                whiskerprops=dict(color='blue', linewidth=2),
                capprops=dict(color='blue', linewidth=2),
                medianprops=dict(color='darkblue', linewidth=2.5))

    # 5-DAG boxes (red)
    plt.boxplot(data_5dag, positions=x_positions + width/2, widths=width,
                patch_artist=True, whis=[0, 100],
                boxprops=dict(facecolor='lightcoral', color='red', linewidth=2),
                whiskerprops=dict(color='red', linewidth=2),
                capprops=dict(color='red', linewidth=2),
                medianprops=dict(color='darkred', linewidth=2.5))

    # Theoretical dots
    for i, (t3, t5) in enumerate(zip(theo_3dag, theo_5dag)):
        if t3 is not None:
            plt.plot(x_positions[i] - width/2, t3, 'o', color='blue', markersize=10, 
                     label='3-DAG vs. 1D-Tree Theoretical' if i == 0 else None)
        if t5 is not None:
            plt.plot(x_positions[i] + width/2, t5, 'o', color='red', markersize=10, 
                     label='5-DAG vs. 1D-Tree Theoretical' if i == 0 else None)

    
    plt.xticks(x_positions, query_ranges)           # shows actual numbers: 60, 3600, ...
    plt.xlabel("Query Lengths", fontsize=14)        
    plt.ylabel("Expected FP-Competitive Ratio", fontsize=14)
    plt.title("Gowalla: 3-DAG vs 5-DAG (Expected FP-Competitive Ratio)", fontsize=15)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, "gowalla_combined_exp_ratio_whisker.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

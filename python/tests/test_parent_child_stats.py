
import msprime
import io
import textwrap
import tskit
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import test_tree_stats as tts
from functools import partial
from time import time
from tqdm import tqdm

def naive_node_parent_child_stat(
    ts, W, f, windows=None, polarised=False, span_normalise=True
):
    windows = ts.parse_windows(windows)
    n, K = W.shape
    M = f(W[0], W[0]).shape[0]
    total = np.sum(W, axis=0)
    sigma = np.zeros((ts.num_trees, ts.num_nodes, M))
    for tree in ts.trees():
        X = np.zeros((ts.num_nodes, K))
        X[ts.samples()] = W
        for u in tree.nodes(order="postorder"):
            for v in tree.children(u):
                X[u] += X[v]
        s = np.zeros((ts.num_nodes, M))
        for u in range(ts.num_nodes):
            for v in tree.children(u):
                s[u] = f(X[u], X[v])
                if not polarised:
                    s[u] += f(total - X[u], total - X[v])
        sigma[tree.index] = s * tree.span
    return tts.windowed_tree_stat(ts, sigma, windows, span_normalise=span_normalise)

def node_parent_child_stat_1(ts, sample_weights, summary_func, windows=None, polarised=False, span_normalise=True):
    '''
    Within a tree, for each parent node, computes statistics involving the weights of both the parent and its children
    Specifically, the sum over each child of that parent of summary_func(parent, child).
    In this implementation, the parent[] array is looped over to find the children of each parent
    '''

    n, state_dim = sample_weights.shape
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    result_dim = summary_func(sample_weights[0], sample_weights[0]).shape[0]
    result = np.zeros((num_windows, ts.num_nodes, result_dim))
    state = np.zeros((ts.num_nodes, state_dim))
    state[ts.samples()] = sample_weights
    total_weight = np.sum(sample_weights, axis=0)

    def node_summary(u, parent):
        #print(u, parent)
        s = 0
        for child in range(ts.num_nodes):
            if parent[child] == u:
                #print(f'{u}:{state[u]}, {child}:{state[child]}')
                s += summary_func(state[u], state[child])
                if not polarised:
                    s += summary_func(total_weight - state[u], total_weight - state[child])
        return s

    window_index = 0
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    # contains summary_func(state[u]) for each node
    current_values = np.zeros((ts.num_nodes, result_dim))
    # contains the location of the last time we updated the output for a node.
    last_update = np.zeros((ts.num_nodes, 1))
    for (t_left, t_right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            u = edge.child
            v = edge.parent
            parent[u] = -1
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] -= state[u]
                current_values[v] = node_summary(v, parent)
                v = parent[v]

        for edge in edges_in:
            u = edge.child
            v = edge.parent
            parent[u] = v
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] += state[u]
                current_values[v] = node_summary(v, parent)
                v = parent[v]

        # Update the windows 
        while window_index < num_windows and windows[window_index + 1] <= t_right:
            w_right = windows[window_index + 1]
            # Flush the contribution of all nodes to the current window.
            for u in range(ts.num_nodes):
                result[window_index, u] += (w_right - last_update[u]) * current_values[
                    u
                ]
                last_update[u] = w_right
            window_index += 1

    assert window_index == windows.shape[0] - 1
    if span_normalise:
        for j in range(num_windows):
            result[j] /= windows[j + 1] - windows[j]
    return result

def node_parent_child_stat_2(ts, sample_weights, summary_func, windows=None, polarised=False, span_normalise=True):
    '''
    Within a tree, for each parent node, computes statistics involving the weights of both the parent and its children
    Specifically, the sum over each child of that parent of summary_func(parent, child).
    In this implementation, children are determined by the tree.children(u) method
    '''

    n, state_dim = sample_weights.shape
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    result_dim = summary_func(sample_weights[0], sample_weights[0]).shape[0]
    result = np.zeros((num_windows, ts.num_nodes, result_dim))
    state = np.zeros((ts.num_nodes, state_dim))
    state[ts.samples()] = sample_weights
    total_weight = np.sum(sample_weights, axis=0)

    def node_summary(u, children):
        #print(u, parent)
        s = 0
        for child in children:
            s += summary_func(state[u], state[child])
            if not polarised:
                s += summary_func(total_weight - state[u], total_weight - state[child])
        return s

    window_index = 0
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    # contains summary_func(state[u]) for each node
    current_values = np.zeros((ts.num_nodes, result_dim))
    # contains the location of the last time we updated the output for a node.
    last_update = np.zeros((ts.num_nodes, 1))
    for ((t_left, t_right), edges_out, edges_in), tree in zip(ts.edge_diffs(), ts.trees()):
        for edge in edges_out:
            u = edge.child
            v = edge.parent
            parent[u] = -1
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] -= state[u]
                current_values[v] = node_summary(v, tree.children(v))
                v = parent[v]

        for edge in edges_in:
            u = edge.child
            v = edge.parent
            parent[u] = v
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] += state[u]
                current_values[v] = node_summary(v, tree.children(v))
                v = parent[v]

        # Update the windows 
        while window_index < num_windows and windows[window_index + 1] <= t_right:
            w_right = windows[window_index + 1]
            # Flush the contribution of all nodes to the current window.
            for u in range(ts.num_nodes):
                result[window_index, u] += (w_right - last_update[u]) * current_values[
                    u
                ]
                last_update[u] = w_right
            window_index += 1

    assert window_index == windows.shape[0] - 1
    if span_normalise:
        for j in range(num_windows):
            result[j] /= windows[j + 1] - windows[j]
    return result

def node_parent_child_stat_3(ts, sample_weights, summary_func, windows=None, polarised=False, span_normalise=True):
    '''
    Within a tree, for each parent node, computes statistics involving the weights of both the parent and its children
    Specifically, the sum over each child of that parent of summary_func(parent, child).
    In this implementation, the summary function is updated after all the weights are propagated up the tree.
    This version is currently not working
    '''

    n, state_dim = sample_weights.shape
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    result_dim = summary_func(sample_weights[0], sample_weights[0]).shape[0]
    result = np.zeros((num_windows, ts.num_nodes, result_dim))
    state = np.zeros((ts.num_nodes, state_dim))
    state[ts.samples()] = sample_weights
    total_weight = np.sum(sample_weights, axis=0)

    def node_summary(u, children):
        #print(u, parent)
        s = 0
        for child in children:
            s += summary_func(state[u], state[child])
            if not polarised:
                s += summary_func(total_weight - state[u], total_weight - state[child])
        return s

    window_index = 0
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    
    # contains summary_func(state[u]) for each node
    current_values = np.zeros((ts.num_nodes, result_dim))
    for u in range(ts.num_nodes):
        current_values[u] = node_summary(u, parent)
    # contains the location of the last time we updated the output for a node.
    last_update = np.zeros(ts.num_nodes)
    for ((t_left, t_right), edges_out, edges_in), tree in zip(ts.edge_diffs(), ts.trees()):
        for edge in edges_out:
            u = edge.child
            v = edge.parent
            parent[u] = -1
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] -= state[u]
                v = parent[v]

        for edge in edges_in:
            u = edge.child
            v = edge.parent
            parent[u] = v
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] += state[u]
                v = parent[v]

        for v in range(ts.num_nodes):
            if last_update[v] == t_left:
                current_values[v] = node_summary(v, tree.children(v))

        # Update the windows 
        while window_index < num_windows and windows[window_index + 1] <= t_right:
            w_right = windows[window_index + 1]
            # Flush the contribution of all nodes to the current window.
            for u in range(ts.num_nodes):
                result[window_index, u] += (w_right - last_update[u]) * current_values[
                    u
                ]
                last_update[u] = w_right
            window_index += 1

    assert window_index == windows.shape[0] - 1
    if span_normalise:
        for j in range(num_windows):
            result[j] /= windows[j + 1] - windows[j]
    return result

def node_parent_child_stat_4(ts, sample_weights, summary_func, windows=None, polarised=False, span_normalise=True):
    '''
    Within a tree, for each parent node, computes a statistic of the form:
    Sum(summary_function(weights(parent),weights(child)) for each child of that parent)
    In this implementation, 
    '''

    n, state_dim = sample_weights.shape
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    result_dim = summary_func(sample_weights[0], sample_weights[0]).shape[0]
    result = np.zeros((num_windows, ts.num_nodes, result_dim))
    state = np.zeros((ts.num_nodes, state_dim))
    state[ts.samples()] = sample_weights
    total_weight = np.sum(sample_weights, axis=0)

    def node_summary(u, children):
        #print(u, parent)
        s = 0
        for child in children:
            s += summary_func(state[u], state[child])
            if not polarised:
                s += summary_func(total_weight - state[u], total_weight - state[child])
        return s

    window_index = 0
    parent = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    children = [[] for i in range(ts.num_nodes)]
    # contains summary_func(state[u]) for each node
    current_values = np.zeros((ts.num_nodes, result_dim))
    # contains the location of the last time we updated the output for a node.
    last_update = np.zeros((ts.num_nodes, 1))
    for ((t_left, t_right), edges_out, edges_in), tree in zip(ts.edge_diffs(), ts.trees()):
        for edge in edges_out:
            u = edge.child
            v = edge.parent
            parent[u] = -1
            children[v].remove(u)
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] -= state[u]
                current_values[v] = node_summary(v, children[v])
                v = parent[v]

        for edge in edges_in:
            u = edge.child
            v = edge.parent
            parent[u] = v
            children[v].append(u)
            while v != -1:
                result[window_index, v] += (t_left - last_update[v]) * current_values[v]
                last_update[v] = t_left
                state[v] += state[u]
                current_values[v] = node_summary(v, children[v])
                v = parent[v]

        # Update the windows 
        while window_index < num_windows and windows[window_index + 1] <= t_right:
            w_right = windows[window_index + 1]
            # Flush the contribution of all nodes to the current window.
            for u in range(ts.num_nodes):
                result[window_index, u] += (w_right - last_update[u]) * current_values[
                    u
                ]
                last_update[u] = w_right
            window_index += 1

    assert window_index == windows.shape[0] - 1
    if span_normalise:
        for j in range(num_windows):
            result[j] /= windows[j + 1] - windows[j]
    return result

def node_parent_child_stat(ts, sample_weights, summary_func, windows=None, polarised=False, span_normalise=True):
    return node_parent_child_stat_3(ts, sample_weights, summary_func, windows=windows, polarised=polarised, span_normalise=span_normalise   )

def verify_pc_contributions(ts):
    '''
    Verification function to check that passing (lambda parent, child: child) to node_parent_child_stat()
    correctly calculates the ancestral contribution of each node
    '''
    weights = np.ones((ts.num_samples,1))
    f = lambda parent: parent
    pc_f = lambda parent, child: child
    contributions = tts.node_general_stat(ts, weights, f, polarised=True)
    pc_stat_values = node_parent_child_stat(ts, weights, pc_f, polarised=True)
    assert np.allclose(pc_stat_values[ts.num_samples:,:], contributions[ts.num_samples:,:])

def verify_pc_unary(ts):
    '''
    Verification function to check that removing unary nodes have zero coalescent
    And that removing unary nodes does not affect the coalescent of other nodes
    (coalescent being the sum over children of child*(parent-child), aka the number of pairs of samples that coalesce at that node)
    '''
    weights = np.ones((ts.num_samples,1))
    pc_f = lambda parent, child: child*(parent-child)
    coalescences_unary = node_parent_child_stat(ts, weights, pc_f, polarised=True)
    coalescences_basic = node_parent_child_stat(ts.simplify(), weights, pc_f, polarised=True)
    assert np.allclose(coalescences_unary, coalescences_basic)

def verify_pc_expectations(ts):
    '''
    Verification function to check that the coalescent converges to the same value
    as if it were calculated based on the average contributions.
    Currently not yet implemented
    '''
    weights = np.ones((ts.num_samples,1))
    f = lambda parent: parent
    pc_f = lambda parent, child: child
    contributions = tts.node_general_stat(ts, weights, f, polarised=True)
    coalescences = node_parent_child_stat(ts, weights, pc_f, polarised=True)
    #get expected coalescences from contributions
    expected_coalescences = ''
    assert np.allclose(coalescences, expected_coalescences)

def verify_pc_algorithms_equal(v1, f1, v2, f2):
    '''
    General verification method to check that
    two versions of node_parent_child_stat() return the same value.
    '''
    def _verify_pc_algorithms_equal(ts):
        weights = np.ones((ts.num_samples,1))
        stat1 = v1(ts, weights, f1, polarised=True)
        stat2 = v2(ts, weights, f2, polarised=True)
        assert np.allclose(stat1, stat2)
    return _verify_pc_algorithms_equal

def test_parent_child_stat(verification = verify_pc_contributions, num_samples = 2, recombination_rate=0, sequence_length=10, num_replicates=100, **kwargs):
    ts_reps = msprime.sim_ancestry(4, recombination_rate=recombination_rate, sequence_length=sequence_length, num_replicates=num_replicates, **kwargs)
    for ts in ts_reps:
        verification(ts)


def benchmark_pc_algorithms(methods):
    times = np.zeros(len(methods))
    reps = msprime.sim_ancestry(100, recombination_rate=0.01, sequence_length=10000, num_replicates=10)
    for ts in tqdm(reps):
        weights = np.ones((ts.num_samples,1))
        for method_num, method in enumerate(methods):
            start = time()
            method(ts, weights)
            times[method_num] += time()-start
    for method_num, method_time in enumerate(times):
        print(f'Method {method_num} ran in {method_time} seconds')


def run_all_pc_tests():
    num_samples=2
    sequence_length=3
    recombination_rate=1
    test_parent_child_stat(num_samples=num_samples, recombination_rate=recombination_rate, sequence_length=sequence_length)
    test_parent_child_stat(verification=verify_pc_unary, num_samples=num_samples, recombination_rate=recombination_rate, sequence_length=sequence_length, record_unary=True)
    pc_f = lambda parent, child: child*(parent-child)
    #test_parent_child_stat(verification=verify_pc_algorithms_equal(naive_node_parent_child_stat, pc_f, node_parent_child_stat, pc_f), num_samples=num_samples, recombination_rate=recombination_rate, sequence_length=sequence_length)
    test_parent_child_stat(verification=verify_pc_algorithms_equal(node_parent_child_stat_2, pc_f, node_parent_child_stat, pc_f), num_samples=num_samples, recombination_rate=recombination_rate, sequence_length=sequence_length)
    print('All tests ran successfully.')

def run_visual_test():
    ts = msprime.sim_ancestry(2, recombination_rate=0, sequence_length=1)
    weights = np.ones((ts.num_samples,1))
    pc_f = lambda parent, child: child*(parent-child)
    node_parent_child_stat(ts, weights, pc_f, polarised=True)
    print(ts.draw_text())

def methods_to_benchmark():
    pc_f = lambda parent, child: child*(parent-child)
    f = lambda x: x
    return [
        #lambda ts, weights: naive_node_parent_child_stat(ts, weights, pc_f, polarised=True),
        lambda ts, weights: tts.node_general_stat(ts, weights, f, polarised=True),
        lambda ts, weights: node_parent_child_stat_2(ts, weights, pc_f, polarised=True),
        lambda ts, weights: node_parent_child_stat_5(ts, weights, pc_f, polarised=True),
    ]

run_all_pc_tests()
#run_visual_test()
#benchmark_pc_algorithms(methods_to_benchmark())
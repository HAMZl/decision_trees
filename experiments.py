"""
Reproduce every figure and number in the README.

    python experiments.py            # writes images/*.png and results/metrics.json

All experiments are seeded, so re-running gives identical output.
"""

import copy
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import COMP2611_DT as dt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, 'images')
RES = os.path.join(HERE, 'results')
os.makedirs(IMG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# Palette (validated reference palette, light mode)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE = '#2a78d6'    # series 1 / "Yes"
ORANGE = '#eb6834'  # series 2 / "No"
NEUTRAL_FILL = '#f0efec'

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
    'font.size': 11,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK_2,
    'axes.titlecolor': INK,
    'axes.titleweight': 'bold',
    'axes.titlesize': 13,
    'axes.titlelocation': 'left',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelcolor': INK_2,
    'ytick.labelcolor': INK_2,
    'legend.frameon': False,
    'legend.labelcolor': INK_2,
    'lines.linewidth': 2,
})


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.relpath(path, HERE))


# ______________________________________________________________________________
# Tree drawing


def draw_tree(tree, title, name, subtitle=None):
    """Top-down tree diagram: grey boxes are tests, coloured pills are leaves."""
    pos, edges = {}, []
    next_x = [0]

    def layout(node, depth):
        if isinstance(node, dt.DecisionLeaf):
            pos[id(node)] = (next_x[0], -depth)
            next_x[0] += 1
        else:
            xs = []
            for val, child in node.branches.items():
                layout(child, depth + 1)
                xs.append(pos[id(child)][0])
                edges.append((node, child, val))
            pos[id(node)] = ((min(xs) + max(xs)) / 2, -depth)
        return node

    layout(tree, 0)
    nodes = {id(n): n for n in dt.iter_nodes(tree)}
    n_leaves = next_x[0]
    depth = dt.tree_depth(tree)
    fig, ax = plt.subplots(figsize=(max(6, 1.15 * n_leaves), 1.0 + 1.25 * (depth + 1)))
    ax.set_axis_off()
    ax.grid(False)

    for parent, child, val in edges:
        (x0, y0), (x1, y1) = pos[id(parent)], pos[id(child)]
        ax.plot([x0, x1], [y0 - 0.17, y1 + 0.17], color=AXIS, lw=1.4, zorder=1)
        t = 0.68  # label nearer the child so sibling labels don't collide
        ax.text(x0 + t * (x1 - x0), y0 + t * (y1 - y0), val, ha='center', va='center',
                fontsize=9, color=INK_2, zorder=3,
                bbox=dict(boxstyle='round,pad=0.2', fc=SURFACE, ec='none'))

    for key, (x, y) in pos.items():
        node = nodes[key]
        if isinstance(node, dt.DecisionLeaf):
            fc, tc, label, w = (BLUE if node.result == 'Yes' else ORANGE), '#ffffff', node.result, 0.62
        else:
            fc, tc, label = NEUTRAL_FILL, INK, node.attr_name
            w = max(0.9, 0.11 * len(label) + 0.3)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.17), w, 0.34,
                                    boxstyle='round,pad=0.02,rounding_size=0.12',
                                    fc=fc, ec=AXIS if fc == NEUTRAL_FILL else fc,
                                    lw=1, zorder=2))
        ax.text(x, y, label, ha='center', va='center', fontsize=10, color=tc,
                weight='bold', zorder=4)

    ax.set_xlim(-0.7, n_leaves - 0.3)
    ax.set_ylim(-depth - 0.45, 0.45)
    full_title = title + ('\n' if subtitle else '')
    ax.set_title(full_title, loc='left', pad=6)
    if subtitle:
        ax.text(0, 1.0, subtitle, transform=ax.transAxes, fontsize=10, color=INK_2, va='bottom')
    save(fig, name)


# ______________________________________________________________________________
# Experiments


def tennis_experiments(metrics):
    tennis, tree = dt.learn_tennis_tree('tennis')
    draw_tree(dt.order(copy.deepcopy(tree)), 'Learned tree: Play Tennis', 'tennis_tree.png',
              'ID3 on all 14 examples. Temp is never used.')

    gains = dt.information_gains(tennis)
    items = sorted(gains.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bars = ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE, height=0.55)
    bars[-1].set_color(BLUE)
    for b, (_, v) in zip(bars, items):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, '{:.3f} bits'.format(v),
                va='center', fontsize=9, color=INK_2)
    ax.set_xlim(0, max(gains.values()) * 1.25)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Information gain at the root (bits)')
    ax.set_title('Outlook is the best first split')
    save(fig, 'tennis_info_gain.png')

    _, _, _, holdout = dt.test_tennis_tree('tennis')
    metrics['tennis'] = dict(
        root_information_gain={k: round(v, 4) for k, v in gains.items()},
        tree_nodes=dt.tree_size(tree),
        holdout_error_last_3=round(holdout, 4),
        loocv_error_tree=round(dt.leave_one_out(tennis, dt.DecisionTreeLearner), 4),
        loocv_error_plurality=round(dt.leave_one_out(tennis, dt.PluralityLearner), 4),
    )


def learning_curve_experiment(metrics, seeds=range(30)):
    sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200]
    curves = {}
    for noise in (0.0, 0.1):
        errs, nodes = [], []
        for s in seeds:
            train = dt.SyntheticRestaurant(200, seed=100 + s, noise=noise)
            test = dt.SyntheticRestaurant(1000, seed=10_000 + s)
            _, e, n = dt.learning_curve(train, test, sizes)
            errs.append(e)
            nodes.append(n)
        curves[noise] = (np.array(errs), np.array(nodes))
    base = []
    for s in seeds:
        train = dt.SyntheticRestaurant(200, seed=100 + s)
        test = dt.SyntheticRestaurant(1000, seed=10_000 + s)
        base.append(dt.err_ratio(dt.PluralityLearner(train), test))
    baseline = float(np.mean(base))

    labels = {0.0: 'Clean labels', 0.1: '10% label noise'}
    colors = {0.0: BLUE, 0.1: ORANGE}

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for noise, (errs, _) in curves.items():
        m = errs.mean(0)
        lo, hi = np.percentile(errs, [10, 90], axis=0)
        ax.fill_between(sizes, lo, hi, color=colors[noise], alpha=0.15, lw=0)
        ax.plot(sizes, m, color=colors[noise], label=labels[noise])
        ax.text(sizes[-1] + 4, m[-1], '{}  {:.1%}'.format(labels[noise], m[-1]),
                color=INK_2, va='center', fontsize=9)
    ax.axhline(baseline, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(sizes[-1] + 4, baseline, 'Plurality baseline  {:.0%}'.format(baseline),
            color=MUTED, va='center', fontsize=9)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Test error (1,000 clean examples)')
    ax.set_title('Learning curve: synthetic restaurant domain', pad=22)
    ax.text(0, 1.02, 'Mean of {} seeds; band = 10th-90th percentile'.format(len(seeds)),
            transform=ax.transAxes, fontsize=9, color=INK_2, va='bottom')
    ax.legend(loc='upper right')
    save(fig, 'learning_curve.png')

    true_size = dt.tree_size(dt.waiting_decision_tree)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for noise, (_, nodes) in curves.items():
        m = nodes.mean(0)
        ax.plot(sizes, m, color=colors[noise], label=labels[noise])
        ax.text(sizes[-1] + 4, m[-1], '{:.0f} nodes'.format(m[-1]), color=INK_2,
                va='center', fontsize=9)
    ax.axhline(true_size, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(3, true_size + 2, 'True tree: {} nodes'.format(true_size), color=MUTED, fontsize=9)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, None)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Nodes in learned tree')
    ax.set_title('Noise makes ID3 grow a much bigger tree')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.92))
    save(fig, 'tree_size.png')

    idx = {n: i for i, n in enumerate(sizes)}
    metrics['learning_curve'] = dict(
        seeds=len(seeds), test_examples=1000, plurality_error=round(baseline, 4),
        true_tree_nodes=true_size,
        **{'{}_noise_{:.0%}'.format(k, noise): {
            str(n): round(float(arr.mean(0)[idx[n]]), 4) for n in (10, 50, 100, 200)}
            for noise, (errs, nodes) in curves.items()
            for k, arr in (('error', errs), ('nodes', nodes))})


def pruning_experiment(metrics, seeds=range(20)):
    """Train on noisy data, prune on a noisy validation set, test on clean data."""
    noises = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    res = {k: [] for k in ('err_full', 'err_pruned', 'n_full', 'n_pruned')}
    for noise in noises:
        row = {k: [] for k in res}
        for s in seeds:
            train = dt.SyntheticRestaurant(200, seed=200 + s, noise=noise)
            val = dt.SyntheticRestaurant(300, seed=5_000 + s, noise=noise)
            test = dt.SyntheticRestaurant(1000, seed=20_000 + s)
            tree = dt.DecisionTreeLearner(train)
            row['err_full'].append(dt.err_ratio(tree, test))
            row['n_full'].append(dt.tree_size(tree))
            _, _, _, pruned, _ = dt.prune_tree(tree, val, verbose=False)
            row['err_pruned'].append(dt.err_ratio(pruned, test))
            row['n_pruned'].append(dt.tree_size(pruned))
        for k in res:
            res[k].append(row[k])
    res = {k: np.array(v) for k, v in res.items()}

    x = [n * 100 for n in noises]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (kf, kp, ylabel, title, fmt) in zip(axes, [
        ('err_full', 'err_pruned', 'Test error (clean labels)', 'Error', '{:.1%}'),
        ('n_full', 'n_pruned', 'Nodes in tree', 'Tree size', '{:.0f}'),
    ]):
        for key, color, label in ((kf, ORANGE, 'Unpruned'), (kp, BLUE, 'Chi-squared pruned')):
            m = res[key].mean(1)
            ax.plot(x, m, color=color, marker='o', ms=6, mec=SURFACE, mew=2, label=label)
            ax.text(x[-1] + 1, m[-1], fmt.format(m[-1]), color=INK_2, va='center', fontsize=9)
        ax.set_xlabel('Label noise in training & validation data (%)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(-1, 34)
        ax.set_ylim(0, None)
        if kf == 'err_full':
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].legend(loc='upper left')
    fig.suptitle('Chi-squared pruning: smaller trees and lower error as noise rises',
                 x=0.07, ha='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.07, 0.905, '200 training / 300 validation / 1,000 clean test examples; '
             'mean of {} seeds; alpha = 0.05'.format(len(seeds)),
             fontsize=9, color=INK_2)
    fig.subplots_adjust(top=0.8, wspace=0.25)
    save(fig, 'pruning_vs_noise.png')

    metrics['pruning_vs_noise'] = {
        '{:.0%}'.format(n): dict(
            error_unpruned=round(float(res['err_full'][i].mean()), 4),
            error_pruned=round(float(res['err_pruned'][i].mean()), 4),
            nodes_unpruned=round(float(res['n_full'][i].mean()), 1),
            nodes_pruned=round(float(res['n_pruned'][i].mean()), 1))
        for i, n in enumerate(noises)}


def coursework_trees(metrics):
    """The exact Task 3/4 pipeline from the coursework, drawn before and after pruning."""
    train, test = dt.genSyntheticTrainSet(), dt.genSyntheticTestSet()
    tree, err = dt.train_tree(train, test)
    _, samples = dt.train_restaurant_tree(train, test)
    draw_tree(dt.order(copy.deepcopy(dt.waiting_decision_tree)),
              'Ground truth: AIMA restaurant tree', 'restaurant_true_tree.png',
              'The tree used to label every synthetic example.')
    draw_tree(dt.order(copy.deepcopy(tree)), 'Learned from 200 synthetic examples',
              'restaurant_tree_unpruned.png',
              '{} nodes, depth {}; test error {:.1%}'.format(
                  dt.tree_size(tree), dt.tree_depth(tree), err))

    prune_set = dt.genPruneTestSet()
    before = dt.err_ratio(tree, prune_set)
    size_before = dt.tree_size(tree)
    _, p, delta, pruned, after = dt.prune_tree(copy.deepcopy(tree), prune_set, verbose=False)
    draw_tree(dt.order(copy.deepcopy(pruned)), 'After chi-squared pruning',
              'restaurant_tree_pruned.png',
              '{} nodes, depth {}; error on 1,000 attribute-noise examples {:.1%} -> {:.1%}'.format(
                  dt.tree_size(pruned), dt.tree_depth(pruned), before, after))
    metrics['coursework'] = dict(
        train_tree_nodes=size_before, train_tree_depth=dt.tree_depth(tree),
        test_error=round(err, 4), samples_required=samples,
        pruned_tree_nodes=dt.tree_size(pruned), prune_set_error_before=round(before, 4),
        prune_set_error_after=round(after, 4))


if __name__ == '__main__':
    metrics = {}
    tennis_experiments(metrics)
    coursework_trees(metrics)
    learning_curve_experiment(metrics)
    pruning_experiment(metrics)
    with open(os.path.join(RES, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

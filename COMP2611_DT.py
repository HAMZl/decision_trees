"""
COMP2611 - Decision Trees
=========================

ID3-style decision tree learning (information gain splits) with
chi-squared (significance) pruning, built on the AIMA ``learning.py`` code.

Run ``python COMP2611_DT.py`` for a text report of every coursework task, or
``python experiments.py`` to regenerate the figures in ``images/``.
"""

import copy
import csv
import os
from collections import defaultdict

import numpy as np
import scipy.stats as stats

from utils import argmax_random_tie, mean, mode, normalize, remove_all, unique

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ______________________________________________________________________________
# Data sets


def _resolve_csv(name):
    """Find ``name + '.csv'`` relative to the cwd first, then to this file."""
    path = name if name.endswith('.csv') else name + '.csv'
    if os.path.exists(path):
        return path
    return os.path.join(DATA_DIR, os.path.basename(path))


class DataSet:
    """
    d.examples   A list of examples. Each one is a list of attribute values.
    d.attr_names List of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.name       Name of the data set.
    """

    def __init__(self, examples=None, attrs=None, attr_names=None, name='', target=-1):
        self.name = name

        if examples is None:
            with open(_resolve_csv(name), 'r', newline='') as file:
                rows = [row for row in csv.reader(file) if row]
            attr_names, self.examples = rows[0], rows[1:]
        else:
            self.examples = examples

        # attrs are the indices of examples, unless otherwise stated.
        if self.examples and attrs is None:
            attrs = list(range(len(self.examples[0])))
        self.attrs = attrs

        if isinstance(attr_names, str):
            self.attr_names = attr_names.split()
        else:
            self.attr_names = attr_names or attrs

        self.target = self.attr_num(target)
        self.inputs = [a for a in self.attrs if a != self.target]
        # possible values for each attribute, in order of first appearance
        self.values = list(map(unique, zip(*self.examples)))

    def attr_num(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attr_names.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None for i, attr_i in enumerate(example)]

    def classes_to_numbers(self, classes=None):
        """Converts class names to numbers."""
        if not classes:
            classes = sorted(self.values[self.target])
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])

    def remove_examples(self, value=''):
        """Remove examples that contain given value."""
        self.examples = [x for x in self.examples if value not in x]
        self.values = list(map(unique, zip(*self.examples)))

    def split_values_by_classes(self):
        """Split values into buckets according to their class."""
        buckets = defaultdict(lambda: [])
        target_names = self.values[self.target]
        for v in self.examples:
            item = [a for a in v if a not in target_names]
            buckets[v[self.target]].append(item)
        return buckets

    def subset(self, examples):
        """A new DataSet with the same schema but different examples."""
        return DataSet(examples=examples, attr_names=self.attr_names,
                       attrs=self.attrs, target=self.target, name=self.name)

    def __repr__(self):
        return '<DataSet({}): {:d} examples, {:d} attributes>'.format(
            self.name, len(self.examples), len(self.attrs))


def err_ratio(predict, dataset, examples=None):
    """Return the proportion of the examples that are NOT correctly predicted."""
    examples = examples if examples is not None else dataset.examples
    if len(examples) == 0:
        return 0.0
    right = sum(predict(dataset.sanitize(e)) == e[dataset.target] for e in examples)
    return 1 - (right / len(examples))


def grade_learner(predict, tests):
    """Fraction of (values, output) tests the learner gets right."""
    return mean(int(predict(X) == y) for X, y in tests)


def train_test_split(dataset, start=None, end=None, test_split=None):
    """
    Either hold out examples[start:end] as the test set, or (with
    ``test_split``) hold out the *last* ``test_split * 100%`` of examples.
    Returns (train_set, test_set).
    """
    examples = dataset.examples
    if test_split is None:
        start = start or 0
        end = len(examples) if end is None else end
        train = examples[:start] + examples[end:]
        val = examples[start:end]
    else:
        total_size = len(examples)
        # round() so e.g. 14 * 0.2 = 2.8 holds out 3 examples, not 2
        val_size = int(round(total_size * test_split))
        train_size = total_size - val_size
        train = examples[:train_size]
        val = examples[train_size:]
    return dataset.subset(train), dataset.subset(val)


def leave_one_out(dataset, learner):
    """Leave-one-out cross-validation error of ``learner`` on ``dataset``."""
    n = len(dataset.examples)
    wrong = 0
    for i in range(n):
        train, test = train_test_split(dataset, start=i, end=i + 1)
        wrong += err_ratio(learner(train), test) > 0
    return wrong / n


# ______________________________________________________________________________
# Learners


def PluralityLearner(dataset):
    """Always predict the most popular class in the training data (baseline)."""
    most_popular = mode([e[dataset.target] for e in dataset.examples])

    def predict(example):
        return most_popular

    return predict


class DecisionFork:
    """
    A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values.

    ``pos``/``neg`` count the "Yes"/"No" examples that reached this node the
    last time the tree was called with ``target`` set (used for pruning).
    """

    def __init__(self, attr, attr_name=None, default_child=None, branches=None, parent=None):
        self.attr = attr
        self.attr_name = attr_name or attr
        self.default_child = default_child
        self.branches = branches or {}
        self.pos = 0
        self.neg = 0
        self.parent_node = parent

    def __call__(self, example, target=None):
        """Classify ``example``; if ``target`` is given, also record counts."""
        if target is not None:
            if example[target] == "Yes":
                self.pos += 1
            else:
                self.neg += 1

        attr_val = example[self.attr]
        if attr_val in self.branches:
            return self.branches[attr_val](example, target)
        if self.default_child is None:
            raise KeyError('No branch for {}={!r}'.format(self.attr_name, attr_val))
        # unseen attribute value: fall back to the training plurality
        return self.default_child(example, target)

    def clear_count(self):
        self.pos = 0
        self.neg = 0
        return 0

    def add(self, val, subtree):
        """Add a branch. If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def display(self, indent=0):
        print(tree_to_text(self, indent))

    def __repr__(self):
        return 'DecisionFork({0!r}, {1!r}, {2!r})'.format(self.attr, self.attr_name, self.branches)


class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""

    def __init__(self, result, parent=None):
        self.pos = 0
        self.neg = 0
        self.result = result
        self.parent_node = parent

    def __call__(self, example, target=None):
        if target is not None:
            if example[target] == "Yes":
                self.pos += 1
            else:
                self.neg += 1
        return self.result

    def clear_count(self):
        self.pos = 0
        self.neg = 0
        return 0

    def display(self, indent=0):
        print('RESULT =', self.result)

    def __repr__(self):
        return repr(self.result)


def DecisionTreeLearner(dataset):
    """ID3: greedily split on the attribute with the highest information gain."""

    target, values = dataset.target, dataset.values

    def decision_tree_learning(examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return plurality_value(parent_examples)
        if all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        if len(attrs) == 0:
            return plurality_value(examples)
        A = choose_attribute(attrs, examples)
        tree = DecisionFork(A, dataset.attr_names[A], plurality_value(examples))
        for (v_k, exs) in split_by(A, examples):
            subtree = decision_tree_learning(exs, remove_all(A, attrs), examples)
            tree.add(v_k, subtree)
        return tree

    def plurality_value(examples):
        """Leaf predicting the most common target value (ties -> first value)."""
        popular = argmax_random_tie(values[target], key=lambda v: count(target, v, examples))
        return DecisionLeaf(popular)

    def count(attr, val, examples):
        return sum(e[attr] == val for e in examples)

    def all_same_class(examples):
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs, key=lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        def I(exs):
            return information_content([count(target, v, exs) for v in values[target]])

        n = len(examples)
        remainder = sum((len(e_i) / n) * I(e_i) for (_, e_i) in split_by(attr, examples))
        return I(examples) - remainder

    def split_by(attr, examples):
        return [(v, [e for e in examples if e[attr] == v]) for v in values[attr]]

    return decision_tree_learning(dataset.examples, dataset.inputs)


def information_content(values):
    """Entropy (bits) of the distribution given by the counts in ``values``."""
    probabilities = normalize(remove_all(0, values))
    return sum(-p * np.log2(p) for p in probabilities)


def information_gains(dataset, examples=None):
    """{attribute name: information gain} for a split at the root."""
    examples = examples if examples is not None else dataset.examples
    t = dataset.target

    def I(exs):
        return information_content([sum(e[t] == v for e in exs) for v in dataset.values[t]])

    base = I(examples)
    gains = {}
    for a in dataset.inputs:
        rem = 0.0
        for v in dataset.values[a]:
            sub = [e for e in examples if e[a] == v]
            if sub:
                rem += len(sub) / len(examples) * I(sub)
        gains[dataset.attr_names[a]] = float(base - rem)
    return gains


# ______________________________________________________________________________
# Tree utilities


def iter_nodes(tree):
    """Yield every node (forks and leaves) in the tree, pre-order."""
    yield tree
    if isinstance(tree, DecisionFork):
        for child in tree.branches.values():
            yield from iter_nodes(child)


def tree_size(tree):
    """Number of nodes (forks + leaves)."""
    return sum(1 for _ in iter_nodes(tree))


def tree_depth(tree):
    """Number of tests on the longest root-to-leaf path."""
    if isinstance(tree, DecisionLeaf):
        return 0
    return 1 + max(tree_depth(c) for c in tree.branches.values())


def tree_to_text(tree, indent=0):
    """Readable, indented rendering of a tree."""
    if isinstance(tree, DecisionLeaf):
        return '-> {}'.format(tree.result)
    lines = []
    for val, sub in tree.branches.items():
        prefix = '    ' * indent + '{} = {} '.format(tree.attr_name, val)
        if isinstance(sub, DecisionLeaf):
            lines.append(prefix + tree_to_text(sub))
        else:
            lines.append(prefix)
            lines.append(tree_to_text(sub, indent + 1))
    return '\n'.join(lines)


def order(tree):
    """Sort every fork's branches by value (for deterministic display)."""
    if isinstance(tree, DecisionFork):
        tree.branches = dict(sorted(tree.branches.items(), key=lambda kv: kv[0]))
        for child in tree.branches.values():
            order(child)
    return tree


def clear_counts(tree):
    """Reset the pos/neg counters on every node, *including* the root."""
    for node in iter_nodes(tree):
        node.clear_count()


# ______________________________________________________________________________
# Chi-squared pruning

# Task 4c
def deviation(value, parent_pos, parent_neg):
    """
    Pearson chi-squared contribution of one child: the squared difference
    between the actual (value.pos, value.neg) and expected counts, where the
    expected counts assume the child has the same class ratio as its parent.
    """
    n_child = value.pos + value.neg
    n_parent = parent_pos + parent_neg
    if n_child == 0 or n_parent == 0:
        return 0.0
    exp_pos = parent_pos * n_child / n_parent
    exp_neg = parent_neg * n_child / n_parent
    dev = 0.0
    if exp_pos > 0:
        dev += (value.pos - exp_pos) ** 2 / exp_pos
    if exp_neg > 0:
        dev += (value.neg - exp_neg) ** 2 / exp_neg
    return dev


def chi_square_test(fork):
    """
    Test whether ``fork``'s split is significant on the counted examples.
    Returns (delta, degrees_of_freedom, p_value). Degrees of freedom is
    (number of non-empty branches - 1), as in AIMA section 19.3.4.
    """
    children = [c for c in fork.branches.values() if c.pos + c.neg > 0]
    delta = sum(deviation(c, fork.pos, fork.neg) for c in children)
    dof = max(len(children) - 1, 1)
    return delta, dof, float(stats.chi2.sf(delta, dof))


def replaceFork(parent, leaf):
    """Replace fork ``parent`` with ``leaf`` in the tree (in place)."""
    if parent.parent_node is None:
        # The root object cannot be swapped out, so collapse all its branches.
        for key in list(parent.branches):
            parent.branches[key] = leaf
        parent.default_child = leaf
        return 1
    for key, value in list(parent.parent_node.branches.items()):
        if value is parent:
            parent.parent_node.branches[key] = leaf
            return 1
    return 0


def _link_parents(tree, parent=None):
    tree.parent_node = parent
    if isinstance(tree, DecisionFork):
        for child in tree.branches.values():
            _link_parents(child, tree)


def prune_pass(tree, dataset, examples=None, alpha=0.05, verbose=False):
    """
    One bottom-up chi-squared pruning pass.

    Routes ``examples`` through the tree to collect counts, then visits forks
    post-order. A fork whose children are all leaves is replaced by a
    majority leaf unless its split is significant at level ``alpha``.
    A fork that no example reaches is left alone (there is no evidence).
    Returns a list of dicts describing each test performed.
    """
    examples = examples if examples is not None else dataset.examples
    clear_counts(tree)
    _link_parents(tree)
    for e in examples:
        tree(e, dataset.target)

    log = []

    def visit(node):
        if isinstance(node, DecisionLeaf):
            return node
        for key, child in list(node.branches.items()):
            node.branches[key] = visit(child)
        if not all(isinstance(c, DecisionLeaf) for c in node.branches.values()):
            return node
        if node.pos + node.neg == 0:
            return node
        delta, dof, p = chi_square_test(node)
        pruned = p > alpha
        log.append(dict(attr=node.attr_name, delta=delta, dof=dof, p_value=p,
                        pos=node.pos, neg=node.neg, pruned=pruned))
        if verbose:
            print('  test {:<12} chi2={:7.3f} dof={} p={:.4f} -> {}'.format(
                node.attr_name, delta, dof, p, 'PRUNE' if pruned else 'keep'))
        if not pruned:
            return node
        if node.pos != node.neg:
            result = 'Yes' if node.pos > node.neg else 'No'
        else:
            result = node.default_child.result if node.default_child else 'No'
        leaf = DecisionLeaf(result)
        # carry the counts so the parent can be tested in this same pass
        leaf.pos, leaf.neg = node.pos, node.neg
        return leaf

    new_root = visit(tree)
    if new_root is not tree:  # the whole tree collapsed to one leaf
        replaceFork(tree, new_root)
    _link_parents(tree)
    return log


# Task 4d
def evaluate(predict, dataset, examples=None, alpha=0.05, verbose=True):
    """
    Run one pruning pass of tree ``predict`` on ``dataset`` and return
    (p_value, delta, error_rate) where p_value/delta come from the last test
    performed and error_rate is measured *after* pruning.
    """
    examples = examples if examples is not None else dataset.examples
    if len(examples) == 0:
        return 0.0, 0.0, 0.0
    log = prune_pass(predict, dataset, examples, alpha=alpha, verbose=verbose)
    p_value = log[-1]['p_value'] if log else 0.0
    delta = log[-1]['delta'] if log else 0.0
    return p_value, delta, err_ratio(predict, dataset, examples)


# ______________________________________________________________________________
# Restaurant domain [AIMA Figure 19.3]


def RestaurantDataSet(examples=None):
    """Build a DataSet of Restaurant waiting examples."""
    return DataSet(name='restaurant1', target='Wait', examples=examples,
                   attr_names='Alternate Bar Fri/Sat Hungry Patrons Raining Reservation WaitEstimate Wait')


restaurant = RestaurantDataSet()


def T(attr_name, branches):
    branches = {value: (child if isinstance(child, DecisionFork) else DecisionLeaf(child))
                for value, child in branches.items()}
    return DecisionFork(attr=restaurant.attr_num(attr_name), attr_name=attr_name, branches=branches)


# The "true" tree from AIMA used to label synthetic restaurant data.
waiting_decision_tree = T('Patrons',
                          {'None': 'No', 'Some': 'Yes',
                           'Full': T('WaitEstimate',
                                     {'>60': 'No', '0-10': 'Yes',
                                      '30-60': T('Alternate',
                                                 {'No': T('Reservation',
                                                          {'Yes': 'Yes',
                                                           'No': T('Bar', {'No': 'No',
                                                                           'Yes': 'Yes'})}),
                                                  'Yes': T('Fri/Sat', {'No': 'No', 'Yes': 'Yes'})}),
                                      '10-30': T('Hungry',
                                                 {'No': 'Yes',
                                                  'Yes': T('Alternate',
                                                           {'No': 'Yes',
                                                            'Yes': T('Raining',
                                                                     {'No': 'No',
                                                                      'Yes': 'Yes'})})})})})


def _flip(label):
    return 'No' if label == 'Yes' else 'Yes'


def SyntheticRestaurant(n=100, seed=2000, noise=0.0):
    """n examples labelled by the true tree; ``noise`` = P(label flipped)."""
    rng = np.random.RandomState(seed)

    def gen():
        example = [rng.choice(v) for v in restaurant.values]
        example[restaurant.target] = waiting_decision_tree(example)
        if noise and rng.random_sample() < noise:
            example[restaurant.target] = _flip(example[restaurant.target])
        return example

    return RestaurantDataSet([gen() for _ in range(n)])


def SyntheticRestaurantTest(n=100, seed=4000, noise=0.0):
    """
    Test examples labelled by the true tree (optionally with label noise).

    NOTE: the original coursework version never applied the true tree, so
    its labels were drawn uniformly at random, and its noise branch
    (``rand >= 1.1``, plus ``==`` instead of ``=``) could never run.
    """
    return SyntheticRestaurant(n, seed=seed, noise=noise)


def SyntheticRestaurantPruneTest(n=100, seed=4000):
    """Examples labelled by the true tree, then with *attribute* noise:
    for ~40% of examples Raining (or, failing that, Bar) is forced to 'No'."""
    rng = np.random.RandomState(seed)

    def gen():
        example = [rng.choice(v) for v in restaurant.values]
        example[restaurant.target] = waiting_decision_tree(example)
        if rng.random_sample() >= 0.2 and rng.random_sample() >= 0.5:
            if example[5] == "Yes":
                example[5] = "No"
            elif example[1] == "Yes":
                example[1] = "No"
        return example

    return RestaurantDataSet([gen() for _ in range(n)])


# ______________________________________________________________________________
# Coursework tasks

# TASK 1
def learn_tennis_tree(filename):
    """Learn a tree from the whole of ``filename``. Returns (dataset, tree)."""
    dataSet = DataSet(name=filename, target='Play')
    tree = DecisionTreeLearner(dataSet)
    return dataSet, tree


# TASK 2
def test_tennis_tree(filename):
    """
    Hold out the last 20% of ``filename``, learn on the rest, and test.
    Returns (trainSet, testSet, tree, error).

    (The original version learnt the tree on *all* the data before
    splitting, so the test examples had been seen in training.)
    """
    dataSet = DataSet(name=filename, target='Play')
    trainSet, testSet = train_test_split(dataSet, test_split=0.2)
    tree = DecisionTreeLearner(trainSet)
    error = err_ratio(tree, testSet)
    return trainSet, testSet, tree, error


# TASK 3a
def genSyntheticTrainSet():
    return SyntheticRestaurant(200)


# TASK 3b
def genSyntheticTestSet():
    return SyntheticRestaurantTest()


def learning_curve(trainSet, testSet, sizes=None, learner=DecisionTreeLearner):
    """Test error (and tree size) when learning from the first n training examples."""
    sizes = sizes or range(1, len(trainSet.examples) + 1)
    errors, sizes_out, nodes = [], [], []
    for n in sizes:
        model = learner(trainSet.subset(trainSet.examples[:n]))
        errors.append(err_ratio(model, testSet))
        sizes_out.append(n)
        nodes.append(tree_size(model) if isinstance(model, (DecisionFork, DecisionLeaf)) else None)
    return sizes_out, errors, nodes


# TASK 3c
def train_restaurant_tree(trainSet, testSet, N=200):
    """
    Learn trees from the first 1..N training examples; return the tree learnt
    from all N and the smallest n that reaches the same test error.
    """
    N = min(N, len(trainSet.examples))
    tree = DecisionTreeLearner(trainSet.subset(trainSet.examples[:N]))
    final_error = err_ratio(tree, testSet)
    _, errors, _ = learning_curve(trainSet, testSet, range(1, N + 1))
    samples_required = next(n for n, e in enumerate(errors, 1) if e <= final_error)
    return tree, samples_required


# TASK 3d
def train_tree(trainSet, testSet):
    tree = DecisionTreeLearner(trainSet)
    return tree, err_ratio(tree, testSet)


# TASK 4a
def genPruneTestSet():
    return SyntheticRestaurantPruneTest(1000)


# TASK 4b
def prune_tree(tree, testSet, alpha=0.05, verbose=True):
    """
    Repeatedly prune ``tree`` with chi-squared tests on ``testSet`` until a
    pass makes no change. Returns (testSet, p_value, delta, tree, error_rate).
    """
    p_value, delta = 0.0, 0.0
    while True:
        log = prune_pass(tree, testSet, alpha=alpha, verbose=verbose)
        if log:
            p_value, delta = log[-1]['p_value'], log[-1]['delta']
        if not any(t['pruned'] for t in log):
            break
    clear_counts(tree)
    error_rate = err_ratio(tree, testSet)
    return testSet, p_value, delta, tree, error_rate


if __name__ == "__main__":
    print('=' * 60, '\nTask 1 - tennis tree (all 14 examples)\n' + '=' * 60)
    tennis, tennis_tree = learn_tennis_tree('tennis')
    print(tree_to_text(tennis_tree))
    print('Root information gain:',
          {k: round(v, 3) for k, v in information_gains(tennis).items()})

    print('\n' + '=' * 60, '\nTask 2 - tennis hold-out test\n' + '=' * 60)
    train, test, t2_tree, err = test_tennis_tree('tennis')
    print('train={} test={} hold-out error={:.3f}'.format(
        len(train.examples), len(test.examples), err))
    print('leave-one-out error: tree={:.3f} plurality={:.3f}'.format(
        leave_one_out(tennis, DecisionTreeLearner), leave_one_out(tennis, PluralityLearner)))

    print('\n' + '=' * 60, '\nTask 3 - synthetic restaurant\n' + '=' * 60)
    train_set = genSyntheticTrainSet()
    test_set = genSyntheticTestSet()
    restaurant_tree, samples = train_restaurant_tree(train_set, test_set)
    tree, error_rate = train_tree(train_set, test_set)
    print('tree: {} nodes, depth {}; test error={:.3f}; samples needed={}'.format(
        tree_size(tree), tree_depth(tree), error_rate, samples))

    print('\n' + '=' * 60, '\nTask 4 - chi-squared pruning\n' + '=' * 60)
    prune_data = genPruneTestSet()
    before = err_ratio(tree, prune_data)
    size_before = tree_size(tree)
    _, p_value, delta, pruned_tree, error = prune_tree(copy.deepcopy(tree), prune_data)
    print('nodes {} -> {}; error on prune set {:.3f} -> {:.3f}'.format(
        size_before, tree_size(pruned_tree), before, error))

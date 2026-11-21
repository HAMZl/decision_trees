import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import COMP2611_DT as dt  # noqa: E402


@pytest.fixture(scope='module')
def tennis():
    return dt.learn_tennis_tree('tennis')


def test_entropy():
    assert dt.information_content([5, 5]) == pytest.approx(1.0)
    assert dt.information_content([10, 0]) == pytest.approx(0.0)
    assert dt.information_content([9, 5]) == pytest.approx(0.940, abs=1e-3)


def test_tennis_root_is_outlook_with_textbook_gain(tennis):
    data, tree = tennis
    assert tree.attr_name == 'Outlook'
    gains = dt.information_gains(data)
    assert gains['Outlook'] == pytest.approx(0.247, abs=1e-3)
    assert max(gains, key=gains.get) == 'Outlook'


def test_tennis_tree_fits_training_data(tennis):
    data, tree = tennis
    assert dt.err_ratio(tree, data) == 0
    assert dt.tree_size(tree) == 8
    assert 'Temp' not in {n.attr_name for n in dt.iter_nodes(tree) if isinstance(n, dt.DecisionFork)}


def test_holdout_examples_are_not_in_training_set():
    train, test, _, _ = dt.test_tennis_tree('tennis')
    assert len(train.examples) + len(test.examples) == 14
    assert not any(e in train.examples for e in test.examples)


def test_loads_csv_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(dt.DataSet(name='tennis', target='Play').examples) == 14


def test_synthetic_test_set_is_labelled_by_true_tree():
    test = dt.SyntheticRestaurantTest()
    assert dt.err_ratio(dt.waiting_decision_tree, test) == 0


def test_label_noise_rate():
    noisy = dt.SyntheticRestaurant(2000, seed=1, noise=0.2)
    assert dt.err_ratio(dt.waiting_decision_tree, noisy) == pytest.approx(0.2, abs=0.03)


def test_deviation_zero_when_child_matches_parent_ratio():
    leaf = dt.DecisionLeaf('Yes')
    leaf.pos, leaf.neg = 3, 1
    assert dt.deviation(leaf, 30, 10) == pytest.approx(0.0)
    leaf.pos, leaf.neg = 4, 0
    # expected (3, 1): (4-3)^2/3 + (0-1)^2/1
    assert dt.deviation(leaf, 30, 10) == pytest.approx(1 / 3 + 1)


def test_pruning_removes_irrelevant_split_and_keeps_relevant_one():
    # Root splits on Patrons (relevant); under 'Full' a split on Raining is noise.
    data = dt.SyntheticRestaurant(400, seed=7)
    t = dt.DecisionFork(4, 'Patrons', dt.DecisionLeaf('No'))
    t.add('None', dt.DecisionLeaf('No'))
    t.add('Some', dt.DecisionLeaf('Yes'))
    t.add('Full', dt.DecisionFork(5, 'Raining', dt.DecisionLeaf('No'),
                                  {'Yes': dt.DecisionLeaf('Yes'), 'No': dt.DecisionLeaf('No')}))
    _, _, _, pruned, _ = dt.prune_tree(t, data, verbose=False)
    assert pruned.attr_name == 'Patrons'
    assert isinstance(pruned.branches['Full'], dt.DecisionLeaf)


def test_pruned_error_is_measured_after_pruning():
    train = dt.SyntheticRestaurant(200, seed=3, noise=0.2)
    val = dt.SyntheticRestaurant(300, seed=4, noise=0.2)
    tree = dt.DecisionTreeLearner(train)
    _, _, _, pruned, err = dt.prune_tree(copy.deepcopy(tree), val, verbose=False)
    assert dt.tree_size(pruned) < dt.tree_size(tree)
    assert err == pytest.approx(dt.err_ratio(pruned, val))


def test_samples_required_reaches_final_error():
    train, test = dt.genSyntheticTrainSet(), dt.genSyntheticTestSet()
    tree, n = dt.train_restaurant_tree(train, test)
    small = dt.DecisionTreeLearner(train.subset(train.examples[:n]))
    assert dt.err_ratio(small, test) <= dt.err_ratio(tree, test)

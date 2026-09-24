"""Tests for dataset.PatientSplitter (patient-level K-fold split)."""

import pytest

from dataset import PatientSplitter

PATIENTS = [f"TCGA_XX_{i:04d}" for i in range(110)]


def test_splits_are_disjoint_and_cover_all_patients():
    splits = PatientSplitter.kfold(PATIENTS, n_folds=5, fold=0, n_validation=10, seed=42)
    train, val, test = map(set, (splits["train"], splits["validation"], splits["test"]))

    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(PATIENTS)
    assert len(test) == 22 and len(val) == 10 and len(train) == 78


def test_test_folds_partition_the_cohort():
    test_sets = [set(PatientSplitter.kfold(PATIENTS, 5, k, 10, 42)["test"]) for k in range(5)]
    assert sum(len(s) for s in test_sets) == len(PATIENTS)
    assert set().union(*test_sets) == set(PATIENTS)


def test_split_is_deterministic_and_order_independent():
    a = PatientSplitter.kfold(PATIENTS, 5, 2, 10, 42)
    b = PatientSplitter.kfold(list(reversed(PATIENTS)), 5, 2, 10, 42)
    assert a == b


def test_different_seed_changes_split():
    a = PatientSplitter.kfold(PATIENTS, 5, 0, 10, 42)
    b = PatientSplitter.kfold(PATIENTS, 5, 0, 10, 43)
    assert a["test"] != b["test"]


def test_outputs_are_sorted():
    splits = PatientSplitter.kfold(PATIENTS, 5, 1, 10, 42)
    for ids in splits.values():
        assert ids == sorted(ids)


@pytest.mark.parametrize("fold", [-1, 5])
def test_invalid_fold_raises(fold):
    with pytest.raises(ValueError):
        PatientSplitter.kfold(PATIENTS, 5, fold, 10, 42)


def test_too_many_validation_patients_raises():
    with pytest.raises(ValueError):
        PatientSplitter.kfold(PATIENTS, 5, 0, n_validation=88, seed=42)

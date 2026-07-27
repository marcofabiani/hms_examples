"""Discovery, manifest, dataset and splitting."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import hmslib as hm
from hmslib import io


def test_scan_folder_pairs_flat_layout(tmp_csv_folder):
    manifest = io.scan_folder(tmp_csv_folder, verbose=False)
    ops = manifest["operating_points"]
    assert set(ops) == {"OP_60", "OP_80"}          # original case preserved
    assert not manifest["unmatched"]
    for pair in ops.values():
        assert "nominal" in pair["nominal"].lower()
        assert "failure" in pair["failures"].lower()


def test_scan_folder_pairs_per_op_folders(tmp_path):
    root = str(tmp_path / "nested")
    hm.synth.write_dataset(root, n_ops=2, n_sensors=5, n_nominal=200,
                           n_classes=2, n_per_class=60, per_op_folder=True,
                           random_state=2)
    manifest = io.scan_folder(root, verbose=False)
    assert set(manifest["operating_points"]) == {"OP_60", "OP_80"}


def test_scan_folder_reports_unmatched_files(tmp_csv_folder):
    pd.DataFrame({"a": [1, 2]}).to_csv(os.path.join(tmp_csv_folder, "notes.csv"),
                                       index=False)
    manifest = io.scan_folder(tmp_csv_folder, verbose=False)
    assert "notes.csv" in manifest["unmatched"]


def test_manifest_roundtrip(tmp_csv_folder, tmp_path):
    path = str(tmp_path / "manifest.json")
    io.scan_folder(tmp_csv_folder, write=path, verbose=False)
    assert os.path.isfile(path)
    manifest = io.load_manifest(path)
    ds = io.Dataset.from_manifest(manifest, verbose=False)
    assert len(ds) == 2


def test_manifest_column_overrides_are_honoured(tmp_csv_folder, tmp_path):
    path = str(tmp_path / "manifest.json")
    manifest = io.scan_folder(tmp_csv_folder, verbose=False)
    manifest["columns"]["intensity"] = "none"
    io.save_manifest(manifest, path)
    ds = io.Dataset.from_manifest(path, verbose=False)
    op = ds[ds.operating_points[0]]
    assert op.intensity is None


def test_load_csv_drops_non_finite_rows_and_reports(tmp_path):
    path = str(tmp_path / "t.csv")
    pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [1.0, 2.0, np.nan]}).to_csv(
        path, index=False)
    df, info = io.load_csv(path)
    assert len(df) == 1
    assert info["rows_dropped"] == 2
    assert np.all(np.isfinite(df.to_numpy()))


def test_dataset_accessors(dataset):
    ds, truths = dataset
    assert len(ds) == 2
    op = ds[ds.operating_points[0]]
    assert len(op.classes) == 3
    assert op.class_counts().sum() == len(op.failures)
    assert op.X_nominal().shape[1] == len(op.sensors)
    assert np.isfinite(op.intensity_values()).all()
    with pytest.raises(KeyError):
        ds["does_not_exist"]


def test_dataset_common_sensors_when_one_op_lacks_a_sensor(dataset):
    ds, _ = dataset
    names = ds.operating_points
    op = ds[names[1]]
    victim = op.sensors[0]
    op.schema.sensors = [s for s in op.sensors if s != victim]
    common = ds.common_sensors()
    assert victim not in common
    avail = ds.sensor_availability()
    assert not bool(avail.loc[victim, names[1]])
    assert bool(avail.loc[victim, names[0]])


def test_dataset_from_frames(op_pair):
    nominal, failures, _ = op_pair
    ds = io.Dataset.from_frames(nominal, failures, name="OP_X")
    assert ds.operating_points == ["OP_X"]
    assert ds["OP_X"].label == "Failure"


def test_split_frame_partitions_without_overlap(op_pair):
    nominal, _, _ = op_pair
    train, test = io.split_frame(nominal, test_size=0.25, random_state=0)
    assert len(train) + len(test) == len(nominal)
    assert abs(len(test) / len(nominal) - 0.25) < 0.01
    assert not set(train.index) & set(test.index)


def test_stratified_split_balances_class_and_intensity(op_pair):
    _, failures, _ = op_pair
    train, test = io.stratified_split(
        failures, label="Failure", intensity="Intensity",
        test_size=0.3, n_bins=5, random_state=0,
    )
    assert len(train) + len(test) == len(failures)
    assert not set(train.index) & set(test.index)

    # every class present on both sides, in similar proportion
    ptr = train["Failure"].value_counts(normalize=True).sort_index()
    pte = test["Failure"].value_counts(normalize=True).sort_index()
    assert set(ptr.index) == set(pte.index)
    assert np.abs(ptr - pte).max() < 0.05

    # and, the point of stratifying: comparable intensity distributions
    for cls in failures["Failure"].unique():
        a = train.loc[train["Failure"] == cls, "Intensity"]
        b = test.loc[test["Failure"] == cls, "Intensity"]
        assert abs(a.mean() - b.mean()) < 0.15 * failures["Intensity"].std()


def test_stratified_split_survives_tiny_strata():
    df = pd.DataFrame({
        "Failure": ["a", "a", "b"],
        "Intensity": [1.0, 2.0, 3.0],
        "s": [0.0, 1.0, 2.0],
    })
    train, test = io.stratified_split(df, "Failure", "Intensity", test_size=0.5)
    assert len(train) + len(test) == 3
    assert "b" in set(train["Failure"])  # single-row stratum goes to train

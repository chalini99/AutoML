import pandas as pd
import pytest

from src.data_utils import (
    DatasetValidationError,
    detect_csv_delimiter,
    load_csv,
    prepare_target,
    select_features,
    suggest_task,
)


@pytest.mark.parametrize(
    ("text", "delimiter"),
    [
        ("age;score\n1;10\n2;20\n", ";"),
        ("age,score\n1,10\n2,20\n", ","),
        ("age\tscore\n1\t10\n2\t20\n", "\t"),
    ],
)
def test_detect_csv_delimiter(text, delimiter):
    assert detect_csv_delimiter(text) == delimiter


def test_load_semicolon_csv():
    df, delimiter = load_csv("school;G3\nA;12\nB;8\n".encode("utf-8"))
    assert delimiter == ";"
    assert df.columns.tolist() == ["school", "G3"]


def test_load_malformed_one_column_csv():
    with pytest.raises(DatasetValidationError, match="fewer than two columns"):
        load_csv(b"G3\n12\n8\n")


def test_load_single_feature_prediction_csv():
    df, delimiter = load_csv(b"school\nGP\nMS\n", minimum_columns=1)
    assert delimiter == ","
    assert df["school"].tolist() == ["GP", "MS"]


def test_prepare_and_derive_pass_fail_target():
    df = pd.DataFrame({"G1": [15, 8], "G2": [14, 9], "G3": [13, 7]})
    X, y = prepare_target(df, "G3", "Classification", derive_pass_fail=True)
    assert X.columns.tolist() == ["G1", "G2"]
    assert y.tolist() == ["Pass", "Fail"]


def test_scenario_feature_selection_is_explicit():
    df = pd.DataFrame({"G1": [1, 2], "G2": [2, 3], "age": [15, 16], "G3": [10, 12]})
    X, _ = prepare_target(df, "G3", "Regression")
    assert select_features(X, "G3", "Early prediction").columns.tolist() == ["age"]
    assert select_features(X, "G3", "Grade-assisted prediction").columns.tolist() == ["G1", "G2", "age"]
    assert select_features(X, "G3", "General dataset").columns.tolist() == ["G1", "G2", "age"]


def test_suggest_task_does_not_overstate_numeric_targets():
    assert suggest_task(pd.Series([0, 1, 0, 1])) == "Classification"
    assert suggest_task(pd.Series([10.1, 11.2, 12.3, 13.4])) == "Regression"
    assert suggest_task(pd.Series(["A", "B", "A"])) == "Classification"

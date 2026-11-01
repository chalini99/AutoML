"""Data loading and validation helpers for the Student AutoML application."""

from __future__ import annotations

import csv
import io
from typing import BinaryIO, TextIO

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be safely used for training."""


def _decode_csv(raw_bytes: bytes) -> tuple[str, str]:
    if not raw_bytes:
        raise DatasetValidationError("The uploaded CSV file is empty.")

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise DatasetValidationError("The file could not be decoded using a supported text encoding.")


def detect_csv_delimiter(text: str) -> str:
    """Detect a delimiter while avoiding a single-column fallback."""
    sample = text[:50_000]
    header = sample.splitlines()[0] if sample.splitlines() else ""
    if header and not any(delimiter in header for delimiter in ",;\t|"):
        return ","
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        if delimiter in header:
            return delimiter
    except csv.Error:
        pass

    try:
        candidate = pd.read_csv(io.StringIO(text), sep=None, engine="python", nrows=20)
        if candidate.shape[1] > 1:
            return str(candidate.attrs.get("delimiter", "")) or ","
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        pass

    raise DatasetValidationError(
        "Could not reliably detect a comma, semicolon, tab, or pipe delimiter. "
        "Please upload a standard delimited CSV file."
    )


def load_csv(
    upload: BinaryIO | TextIO | bytes,
    minimum_columns: int = 2,
) -> tuple[pandas_dataframe, str]:
    """Read a CSV from a file object or bytes and report its delimiter."""
    if isinstance(upload, bytes):
        raw_bytes = upload
    else:
        position = upload.tell() if hasattr(upload, "tell") else None
        raw = upload.read()
        if hasattr(upload, "seek") and position is not None:
            upload.seek(position)
        raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw

    text, _ = _decode_csv(raw_bytes)
    delimiter = detect_csv_delimiter(text)

    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetValidationError(f"The CSV could not be parsed: {exc}") from exc

    validate_dataframe(df, minimum_columns=minimum_columns)
    return df, delimiter


def validate_dataframe(df: pd.DataFrame, minimum_columns: int = 2) -> None:
    if df.empty:
        raise DatasetValidationError("The dataset has no usable rows.")
    if minimum_columns < 1:
        raise ValueError("minimum_columns must be at least one.")
    if len(df.columns) < minimum_columns:
        if minimum_columns == 2:
            raise DatasetValidationError(
                "The dataset has fewer than two columns. Confirm that the CSV delimiter is correct."
            )
        raise DatasetValidationError(
            f"The dataset has fewer than {minimum_columns} columns. "
            "Confirm that the CSV delimiter is correct."
        )
    if df.shape[1] != len(set(df.columns)):
        raise DatasetValidationError("Column names must be unique.")


def suggest_task(series: pd.Series) -> str:
    """Suggest a task without silently changing the user's explicit choice."""
    unique_count = int(series.nunique(dropna=True))
    is_discrete_numeric = pd.api.types.is_integer_dtype(series) and unique_count <= 30
    if is_discrete_numeric:
        return "Classification"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return "Classification"
    return "Regression"


def prepare_target(
    df: pd.DataFrame,
    target: str,
    task: str,
    derive_pass_fail: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise DatasetValidationError("The selected target column is missing.")
    if task not in {"Classification", "Regression"}:
        raise DatasetValidationError("Choose Classification or Regression.")

    data = df.dropna(subset=[target]).copy()
    y = data[target].copy()

    if derive_pass_fail:
        if target != "G3" or not pd.api.types.is_numeric_dtype(y):
            raise DatasetValidationError("G3 pass/fail requires a numeric G3 target.")
        y = (y >= 10).map({True: "Pass", False: "Fail"})

    if task == "Classification":
        if y.nunique(dropna=True) < 2:
            raise DatasetValidationError("The target needs at least two distinct values.")
    elif not pd.api.types.is_numeric_dtype(y):
        raise DatasetValidationError("Regression targets must be numeric.")

    X = data.drop(columns=[target])
    if len(X) < 2:
        raise DatasetValidationError("The dataset needs at least two usable rows.")
    if X.shape[1] == 0:
        raise DatasetValidationError("Select a target that leaves at least one input column.")

    return X, y


def select_features(
    X: pd.DataFrame,
    target: str,
    scenario: str,
) -> pd.DataFrame:
    features = X.drop(columns=[target], errors="ignore")
    if scenario == "Early prediction" and "G1" in features.columns and "G2" in features.columns:
        features = features.drop(columns=["G1", "G2"])
    elif scenario not in {
        "Early prediction",
        "Grade-assisted prediction",
        "General dataset",
    }:
        raise DatasetValidationError("Select a valid prediction scenario.")
    return features


pandas_dataframe = pd.DataFrame

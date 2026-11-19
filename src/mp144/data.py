"""Data loading, cleaning, and quality investigation for MP144."""

import hashlib
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_ownership(series: pd.Series) -> pd.Series:
    """Standardize spelling variants in ownership column to 3 true levels:
    - cooperative
    - public_sector
    - private
    """
    def _map_val(val: Any) -> str:
        if pd.isna(val):
            return "public_sector"
        s = str(val).strip().lower()
        if "coop" in s or s == "coo.":
            return "cooperative"
        elif "pub" in s or "sector" in s:
            return "public_sector"
        elif "pri" in s:
            return "private"
        return s

    return series.apply(_map_val)


def clean_dataset(
    df: pd.DataFrame,
    is_train: bool = True,
    target_col: str = "litres_per_day",
    typo_threshold: float = 50000.0,
    leaking_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Clean the raw dataframe according to findings:
    1. Fix target typos (divided by 10 if above typo_threshold)
    2. Standardize ownership spelling variants
    3. Drop leaking columns if present
    """
    df_clean = df.copy()

    # Clean ownership
    if "ownership" in df_clean.columns:
        df_clean["ownership"] = clean_ownership(df_clean["ownership"])

    # Fix target typos in training data
    if is_train and target_col in df_clean.columns:
        typo_mask = df_clean[target_col] > typo_threshold
        df_clean.loc[typo_mask, target_col] = df_clean.loc[typo_mask, target_col] / 10.0

    # Drop leaking columns
    if leaking_cols is None:
        leaking_cols = ["litres_from_tank_dip_next_day", "liters_from_tank_dip_next_day"]
    
    cols_to_drop = [c for c in leaking_cols if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)

    return df_clean


def detect_seven_issues(
    train_df: pd.DataFrame,
    holdout_df: Optional[pd.DataFrame] = None,
    target_col: str = "litres_per_day",
) -> Dict[str, Any]:
    """Examine train_df for the seven data quality issues described in the student handbook:
    1. leaking_column
    2. unit_mixup
    3. sentinel_value
    4. duplicate_rows
    5. stuck_sensor
    6. target_typos
    7. spelling_variants
    """
    issues: Dict[str, Any] = {}

    # 1. Leaking column
    leaking_candidates = []
    if holdout_df is not None:
        for c in train_df.columns:
            if c not in holdout_df.columns and c != target_col:
                leaking_candidates.append(c)
    else:
        for c in ["litres_from_tank_dip_next_day", "liters_from_tank_dip_next_day"]:
            if c in train_df.columns:
                leaking_candidates.append(c)

    if leaking_candidates:
        issues["leaking_column"] = {
            "present": True,
            "column": leaking_candidates[0],
        }
    else:
        issues["leaking_column"] = {"present": False, "column": None}

    # 2. Unit mixup (check numeric features grouped by categorical columns)
    unit_mixup_found = False
    mixup_details = {"present": False, "column": None, "group_value": None, "factor": None}
    for group_col in ["zone", "ownership"]:
        if group_col not in train_df.columns:
            continue
        for num_col in train_df.select_dtypes(include=[np.number]).columns:
            if num_col in [target_col] + leaking_candidates:
                continue
            stats = train_df.groupby(group_col)[num_col].mean()
            if len(stats) > 1 and stats.min() > 0:
                ratio = stats.max() / stats.min()
                for factor in [10, 12, 60, 100, 1000]:
                    if 0.8 * factor <= ratio <= 1.2 * factor:
                        unit_mixup_found = True
                        affected_grp = stats.idxmax() if stats.max() > stats.median() else stats.idxmin()
                        mixup_details = {
                            "present": True,
                            "column": num_col,
                            "group_value": affected_grp,
                            "factor": factor,
                        }
                        break
            if unit_mixup_found:
                break
        if unit_mixup_found:
            break
    issues["unit_mixup"] = mixup_details

    # 3. Sentinel value
    sentinel_found = False
    sentinel_details = {"present": False, "column": None, "value": None, "count": None}
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col in [target_col] + leaking_candidates:
            continue
        for val in [-1, -999, 999, 9999]:
            cnt = int((train_df[col] == val).sum())
            if cnt > 5:
                sentinel_found = True
                sentinel_details = {
                    "present": True,
                    "column": col,
                    "value": val,
                    "count": cnt,
                }
                break
        if sentinel_found:
            break
    issues["sentinel_value"] = sentinel_details

    # 4. Duplicate rows
    dup_count = int(train_df.duplicated().sum())
    issues["duplicate_rows"] = {
        "present": dup_count > 0,
        "count": dup_count if dup_count > 0 else None,
    }

    # 5. Stuck sensor
    stuck_found = False
    stuck_details = {"present": False, "column": None, "start_date": None, "end_date": None}
    if "date" in train_df.columns:
        df_sorted = train_df.sort_values("date").reset_index(drop=True)
        for col in train_df.select_dtypes(include=[np.number]).columns:
            if col in [target_col] + leaking_candidates:
                continue
            daily_nunique = df_sorted.groupby("date")[col].nunique()
            dates_single = daily_nunique[daily_nunique == 1].index
            if len(dates_single) >= 7:
                daily_val = df_sorted.groupby("date")[col].first()
                s = pd.Series(dates_single)
                diffs = pd.to_datetime(s).diff().dt.days
                streak = (diffs != 1).cumsum()
                for _, grp in s.groupby(streak):
                    if len(grp) >= 7:
                        vals = set(daily_val[grp].values)
                        if len(vals) == 1:
                            stuck_found = True
                            stuck_details = {
                                "present": True,
                                "column": col,
                                "start_date": grp.iloc[0],
                                "end_date": grp.iloc[-1],
                            }
                            break
            if stuck_found:
                break
    issues["stuck_sensor"] = stuck_details

    # 6. Target typos
    typo_count = 0
    if target_col in train_df.columns:
        typo_count = int((train_df[target_col] > 50000).sum())
    issues["target_typos"] = {
        "present": typo_count > 0,
        "count": typo_count if typo_count > 0 else None,
    }

    # 7. Spelling variants
    spelling_found = False
    spelling_details = {"present": False, "column": None, "true_levels": None}
    for col in train_df.select_dtypes(include=["object"]).columns:
        if col in ["record_id", "date"]:
            continue
        cleaned = clean_ownership(train_df[col]) if col == "ownership" else train_df[col].astype(str).str.strip().str.lower()
        raw_levels = train_df[col].nunique()
        clean_levels = cleaned.nunique()
        if raw_levels > clean_levels:
            spelling_found = True
            spelling_details = {
                "present": True,
                "column": col,
                "true_levels": int(clean_levels),
            }
            break
    issues["spelling_variants"] = spelling_details

    return issues

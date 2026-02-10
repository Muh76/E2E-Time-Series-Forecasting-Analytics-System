"""
Rossmann store sales ETL: load, merge, normalize, and clean.

No file writing, feature engineering, or forecasting logic.
See docs/DATA_CONTRACT.md §9 for the Rossmann data contract.
"""

import pandas as pd


# Required columns for each step (pre-rename names where applicable)
TRAIN_REQUIRED = {"Date", "Store", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"}
STORE_REQUIRED = {"Store", "StoreType", "Assortment", "CompetitionDistance"}
NORMALIZE_REQUIRED = TRAIN_REQUIRED | STORE_REQUIRED
CATEGORICAL_COLUMNS = ("state_holiday", "store_type", "assortment")


class RossmannETL:
    """
    ETL for Rossmann train + store data: load CSVs, left-join, normalize schema,
    and apply cleaning rules. No file writing; callers persist output (e.g. to Parquet).
    """

    def load_train(self, path: str) -> pd.DataFrame:
        """
        Load train CSV (historical sales and store-day attributes).

        Args:
            path: Path to train.csv.

        Returns:
            DataFrame with at least Date, Store, Sales, Open, Promo, StateHoliday, SchoolHoliday.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If any required column is missing (Date, Store, Sales, Open, Promo, StateHoliday, SchoolHoliday).
        """
        df = pd.read_csv(path)
        missing = TRAIN_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"train CSV missing required columns: {sorted(missing)}")
        return df

    def load_store(self, path: str) -> pd.DataFrame:
        """
        Load store CSV (store master: type, assortment, competition distance).

        Args:
            path: Path to store.csv.

        Returns:
            DataFrame with at least Store, StoreType, Assortment, CompetitionDistance.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If any required column is missing (Store, StoreType, Assortment, CompetitionDistance).
        """
        df = pd.read_csv(path)
        missing = STORE_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"store CSV missing required columns: {sorted(missing)}")
        return df

    def merge(self, train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
        """
        Left-join train on store by Store. All train rows kept; store attributes attached where key matches.

        Args:
            train_df: Output of load_train (must have column Store).
            store_df: Output of load_store (must have column Store).

        Returns:
            Single DataFrame with train columns plus store columns (StoreType, Assortment, CompetitionDistance, etc.).

        Raises:
            ValueError: If train_df or store_df is missing the join key column 'Store'.
        """
        if "Store" not in train_df.columns:
            raise ValueError("train_df missing required column: Store")
        if "Store" not in store_df.columns:
            raise ValueError("store_df missing required column: Store")
        return train_df.merge(store_df, on="Store", how="left", suffixes=("", "_store"))

    def normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to internal names and add target_cleaned. No cleaning logic; target_cleaned is a copy of target_raw.

        Renames: Date -> date, Store -> store_id, Sales -> target_raw; Open -> open, Promo -> promo,
        StateHoliday -> state_holiday, SchoolHoliday -> school_holiday, StoreType -> store_type,
        Assortment -> assortment, CompetitionDistance -> competition_distance.
        Adds: target_cleaned (initialized from target_raw; clean() will apply rules).
        Ensures date is datetime and numeric/bool types are set for downstream contract.

        Args:
            df: Merged DataFrame (train + store) with columns Date, Store, Sales, Open, Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance.

        Returns:
            DataFrame with normalized column names and target_cleaned added.

        Raises:
            ValueError: If any required column is missing.
        """
        missing = NORMALIZE_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns for normalization: {sorted(missing)}")

        out = df.copy()

        # Explicit renames (contract: Date->date, Store->store_id, Sales->target_raw)
        rename = {
            "Date": "date",
            "Store": "store_id",
            "Sales": "target_raw",
            "Open": "open",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday",
            "StoreType": "store_type",
            "Assortment": "assortment",
            "CompetitionDistance": "competition_distance",
        }
        out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

        # date -> datetime (daily); target_raw and target_cleaned -> float
        out["date"] = pd.to_datetime(out["date"])
        out["target_raw"] = out["target_raw"].astype(float)
        out["target_cleaned"] = out["target_raw"].copy()

        # Bools
        out["open"] = out["open"].astype(bool)
        out["promo"] = out["promo"].astype(bool)
        out["school_holiday"] = out["school_holiday"].astype(bool)

        return out

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning rules: Open==0 -> target_cleaned=0; fill missing CompetitionDistance with median;
        cast categorical columns; sort by (store_id, date); ensure no duplicate (store_id, date).

        Args:
            df: DataFrame from normalize_schema (must have date, store_id, target_cleaned, open, competition_distance, state_holiday, store_type, assortment).

        Returns:
            Single DataFrame sorted by (store_id, date) with no duplicate (store_id, date).

        Raises:
            ValueError: If required columns are missing.
        """
        required = {"date", "store_id", "target_cleaned", "open", "competition_distance", "state_holiday", "store_type", "assortment"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns for cleaning: {sorted(missing)}")

        out = df.copy()

        # Rows with Open == 0: target_cleaned = 0 (preserve grain; closed days are not demand)
        out.loc[~out["open"], "target_cleaned"] = 0.0

        # Missing CompetitionDistance: fill with median (global)
        median_dist = out["competition_distance"].median()
        if pd.isna(median_dist):
            median_dist = 0.0
        out["competition_distance"] = out["competition_distance"].fillna(median_dist).astype(float)

        # Categorical columns cast explicitly
        for col in CATEGORICAL_COLUMNS:
            if col in out.columns:
                out[col] = out[col].astype("category")

        # Sort by (store_id, date)
        out = out.sort_values(["store_id", "date"], ignore_index=True)

        # No duplicate (store_id, date)
        out = out.drop_duplicates(subset=["store_id", "date"], keep="first")

        return out

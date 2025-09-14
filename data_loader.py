import glob
import os
import pandas as pd


def load_data(folder: str = "data") -> pd.DataFrame:
    """Load and clean raw production CSV files from *folder* into a DataFrame.

    The function concatenates all ``*.csv`` files, renames key columns and
    returns a cleaned DataFrame sorted by well and date with negative rates
    removed.
    """
    all_csv = glob.glob(os.path.join(folder, "*.csv"))
    df_0 = pd.concat((pd.read_csv(f) for f in all_csv), ignore_index=True)
    print(f"Считано файлов: {len(all_csv)}")
    df_0 = df_0.rename(
        columns={
            "BBLS_OIL_COND": "oil",
            "MCF_GAS": "gas",
            "BBLS_WTR": "water",
            "API_WellNo": "well_name",
            "RptDate": "date",
            "DAYS_PROD": "days_prod",
        }
    )
    df_0["date"] = pd.to_datetime(df_0["date"])
    df = df_0.drop(columns=["Lease_Unit", "Formation"])
    df = df.sort_values(by=["well_name", "date"]).reset_index(drop=True)
    df = df[(df["oil"] >= 0) & (df["gas"] >= 0) & (df["water"] >= 0)]
    return df

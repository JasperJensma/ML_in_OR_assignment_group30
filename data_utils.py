import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def read_unprocessed_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    n = len(df)
    n_cols = len(df.columns) - 2 #two response variables
    new_col_names = ["y1", "y2"]
    for i in range(n_cols):
        new_col_names.append(f"x{i+1}")
    df.columns = new_col_names
    df.drop(columns=["x27", "x33"], axis=1, inplace=True)
    df["x3_0"] = np.where(df["x3"] == 0, 1, 0)
    df["x4_0"] = np.where(df["x4"] == 0, 1, 0)
    return df

def make_cv_splits(data: pd.DataFrame, k: int=8) -> None:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    data['cv_fold'] = -1

    for fold_number, (train_idx, val_idx) in enumerate(kf.split(data)):
        data.loc[val_idx, 'cv_fold'] = fold_number
    


def main():
    path = "documents/data/GroupAssignment-Data.csv"
    df = read_unprocessed_data(path=path)
    make_cv_splits(df)
    file_name = "documents/data/processed_data.csv"
    df.to_csv(file_name, index=False)



if __name__ == "__main__":
    main()

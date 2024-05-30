import os
from typing import Literal, AnyStr, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold  # type: ignore


def get_dates(
    topdir: str,
    split_name: Literal["training", "validation", "test", "bad"],
    split_id: str | int,
    n_split: str | int,
) -> np.ndarray:
    if split_name != "bad":
        file_path = os.path.join(
            topdir, "split", f"split_{split_id}_{n_split}_{split_name}_dates.csv"
        )
    else:
        file_path = os.path.join(topdir, "split", "bad_test_events.csv")
    assert os.path.exists(file_path), f"File {file_path} does not exist"
    return pd.read_csv(file_path, header=None).iloc[:, 0].to_numpy()


def get_model(
    topdir: str, model_name: str, date: str, case_study_max: float
) -> np.ndarray:
    file_path = os.path.join(topdir, "models", f"{model_name}_{date}_0024_regrid.csv")
    assert os.path.exists(file_path), f"File {file_path} does not exist"
    model_data = pd.read_csv(file_path, sep=";", header=None).to_numpy()
    return model_data / case_study_max


def get_mask_indices(topdir: str, ispadded: bool):
    cs = topdir.stem.split("_")[-1]  # OI | radar
    if cs == 'RYDL':
        mask = np.ones((96, 128))
    else:
        file_path = next(topdir.glob(f"{cs}*regrid*piem_vda.csv"))
        if not file_path.exists():
            file_path = next(topdir.glob(f"{cs}*piem_vda.csv"))
        # file_path = os.path.join(topdir, "OI_regrid_mask_piem_vda_unet.csv")

        mask = pd.read_csv(file_path, sep=";", header=None).to_numpy()
        if ispadded:
            mask = np.pad(
                mask, ((0, 0), (0, 128 - mask.shape[1])), mode="constant", constant_values=0
            )

    indices = np.where(mask == 1)
    indices_zero = np.where(mask == 0)
    return indices, indices_zero, mask


def get_obs(topdir: str, date: str, case_study_max: float) -> np.ndarray:
    # file_path = topdir / "obs" / "data" / f"OI_{date}_regrid.csv"
    case_study = topdir.stem.split("_")[-1]
    file_path = list((topdir / "obs" / "data").glob(f"{case_study}*{date}*csv"))
    if len(file_path) > 1:
        file_path = [f for f in file_path if f.stem.endswith("regrid")][0]
    elif len(file_path) == 1:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f"File '{case_study}*{date}*.csv' does not exist")
    obs_data = pd.read_csv(file_path, sep=";", header=None).to_numpy()
    return obs_data / case_study_max


def load_data(
    topdir: str,
    dates: np.ndarray,
    case_study_max: float,
    indices: np.ndarray,
    indices_zero: np.ndarray,
    available_models: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    from matplotlib import pyplot as plt  # type: ignore

    models_data = []
    obs_data = []
    for date in dates:
        _models_tmp = []
        # print(date)
        for model_name in available_models:
            _mdl_data = get_model(topdir, model_name, date, case_study_max)
            _mdl_data = np.hstack(
                (
                    _mdl_data,
                    np.zeros(
                        (
                            _mdl_data.shape[0],
                            int(2 ** np.ceil(np.log2(_mdl_data.shape[1])))
                            - _mdl_data.shape[1],
                        )
                    ),
                )
            )  # **update:** this was hardcoded as (96, 12) [12 supposed to be 128]
            # _mdl_data[indices_zero] = 0  #apply the mask to the input models
            # print(model_name)
            # plt.imshow(_mdl_data)
            # plt.show()
            if _mdl_data.shape[0] % 2 != 0:
                _mdl_data = _mdl_data[:-1]
            _models_tmp.append(_mdl_data)
        models_data.append(_models_tmp)
        _obs_data = get_obs(topdir, date, case_study_max)
        _obs_data = np.hstack(
            (
                _obs_data,
                np.zeros(
                    (
                        _obs_data.shape[0],
                        int(2 ** np.ceil(np.log2(_obs_data.shape[1])))
                        - _obs_data.shape[1],
                    )
                ),
            )
        )
        _obs_data[indices_zero] = 0
        if _obs_data.shape[0] % 2 != 0:
            _obs_data = _obs_data[:-1]
        # print("obs")
        # plt.imshow(_obs_data)
        # plt.show()
        obs_data.append(_obs_data)
    x = np.stack(models_data, axis=0).astype(np.float32)
    y = np.stack(obs_data, axis=0).astype(np.float32)
    return x, y, x.shape[1], 1


# def get_casestudy_stuff(input_path:str, split_idx:str, n_split: int, case_study:str, ispadded:bool):
#     # if case_study=="24h_10mmMAX_OI":
#     case_study_max=483.717752
#     available_models = ["bol00", "e1000", "c2200", "c5m00"]
#     train_dates = get_dates(input_path, "training", split_idx, n_split)
#     val_dates = get_dates(input_path, "validation", split_idx, n_split)
#     test_dates = get_dates(input_path, "test", split_idx, n_split)
#     try:
#         bad_dates = get_dates(input_path, "bad", "", 0)
#         test_dates=np.concatenate((test_dates, bad_dates), axis=None)
#     except:
#         pass

#     indices_one, indices_zero, mask = get_mask_indices(input_path,ispadded)
#     nx=mask.shape[0]
#     ny=mask.shape[1]
#     return case_study_max,available_models, train_dates, val_dates, test_dates, indices_one, indices_zero, mask, nx, ny


def get_casestudy_stuff(
    input_path: str, n_split: int, case_study: str, ispadded: bool, seed: int
):  
    if case_study == 'RYDL':
        case_study_max = 40.9375
        available_models = ["1", "2", "3"]
    else:
        case_study_max = 483.717752
        available_models = ["bol00", "e1000", "c2200", "c5m00"]

    dates = pd.read_csv(input_path / "split/cluster_all_dates.csv", sep=";")
    skf = StratifiedKFold(n_splits=9, random_state=seed, shuffle=True)
    train_index, test_index = list(skf.split(dates, dates.NAME))[n_split]
    val_index, train_index = np.split(train_index, [len(test_index)])
    train_dates = dates.iloc[train_index].DATA.values
    val_dates = dates.iloc[val_index].DATA.values
    test_dates = dates.iloc[test_index].DATA.values

    indices_one, indices_zero, mask = get_mask_indices(input_path, ispadded)
    nx, ny = mask.shape
    return (
        case_study_max,
        available_models,
        train_dates,
        val_dates,
        test_dates,
        indices_one,
        indices_zero,
        mask,
        nx,
        ny,
    )


def date_features(dates: List[AnyStr]):
    """
    Extract temporal features from a date expressed in the form yyyymmdd (e.g. 20200612)
    """
    df = pd.to_datetime(dates, format="%Y%m%d").rename("date").to_frame()
    season = df.date.dt.month.apply(
        lambda x: 0
        if x in [12, 1, 2]
        else 1
        if x in [3, 4, 5]
        else 2
        if x in [6, 7, 8]
        else 3
    )
    df["sin_season"] = np.sin(2 * np.pi * season / 4)
    df["cos_season"] = np.cos(2 * np.pi * season / 4)
    df["sin_month"] = np.sin(2 * np.pi * df.date.dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * df.date.dt.month / 12)
    df["sin_week"] = np.sin(2 * np.pi * df.date.dt.isocalendar().week / 52)
    df["cos_week"] = np.cos(2 * np.pi * df.date.dt.isocalendar().week / 52)
    df["sin_day"] = np.sin(2 * np.pi * df.date.dt.dayofyear / 365)
    df["cos_day"] = np.cos(2 * np.pi * df.date.dt.dayofyear / 365)

    return df.reset_index(drop=True).drop(columns=["date"]).values.astype(np.float32)
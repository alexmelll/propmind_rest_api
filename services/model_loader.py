import os
import re
import joblib
from datetime import datetime
from typing import Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# def load_latest_models(model_dir: str) -> Tuple:
#     """
#     Loads the most recent regressor and classifier model files from a directory.
#     Expects filenames like 'reg_xgb_YYYYMMDD.pkl' and 'class_lgbm_YYYYMMDD.pkl'.
#     """
#     pattern = {
#         "regressor": re.compile(r"reg_xgb_(\d{8})\.pkl"),
#         "classifier": re.compile(r"class_lgbm_(\d{8})\.pkl"),
#         # "knn": re.compile(r"knn_(\d{8})\.pkl"),
#     }
#
#     def get_latest_model(pattern, files):
#         dated_files = []
#         for f in files:
#             match = pattern.search(f)
#             if match:
#                 date_str = match.group(1)
#                 try:
#                     date = datetime.strptime(date_str, "%Y%m%d")
#                     dated_files.append((date, f))
#                 except ValueError:
#                     continue
#         if not dated_files:
#             raise FileNotFoundError("No model file matching pattern found.")
#         _, latest_file = max(dated_files)
#         return os.path.join(model_dir, latest_file)
#
#     all_files = os.listdir(model_dir)
#
#     reg_path = get_latest_model(pattern["regressor"], all_files)
#     class_path = get_latest_model(pattern["classifier"], all_files)
#     # knn_path = get_latest_model(pattern["knn"], all_files)
#
#     regressor = joblib.load(reg_path)
#     classifier = joblib.load(class_path)
#     # knn = joblib.load(knn_path)
#
#     return classifier, regressor

def get_preprocessor():
    return ColumnTransformer([
        ('num', StandardScaler(), ['floor_area', 'num_rooms', 'energy_eff', 'floor_level']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['built_form', 'tenure'])
    ])
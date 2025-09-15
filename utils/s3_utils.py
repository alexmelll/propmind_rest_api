import os, io, boto3, pandas as pd, joblib, re
from datetime import datetime
from rest_api.services.reporting.report import build_html

USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
s3 = boto3.client("s3") if USE_S3 else None

# --------- Reports ----------
def write_report(data: dict, out_path: str) -> str:
    html = build_html(data)

    if USE_S3:
        bucket = os.getenv("REPORTS_BUCKET", "propmind-reports")
        s3.put_object(
            Bucket=bucket,
            Key=out_path,  # treat out_path as S3 key
            Body=html,
            ContentType="text/html"
        )
        return f"s3://{bucket}/{out_path}"
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        return out_path


# --------- Data ----------
def load_csv(filename: str) -> pd.DataFrame:
    if USE_S3:
        bucket = os.getenv("DATA_BUCKET", "propmind-data")
        obj = s3.get_object(Bucket=bucket, Key=filename)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    else:
        data_dir = os.getenv("DATA_DIR", "../data")
        return pd.read_csv(os.path.join(data_dir, filename))


# --------- Models ----------
def load_latest_models() -> tuple:
    pattern = {
        "regressor": re.compile(r"reg_xgb_(\d{8})\.pkl"),
        "classifier": re.compile(r"class_lgbm_(\d{8})\.pkl"),
    }

    def get_latest(pattern, files):
        dated_files = []
        for f in files:
            m = pattern.search(f)
            if m:
                try:
                    date = datetime.strptime(m.group(1), "%Y%m%d")
                    dated_files.append((date, f))
                except ValueError:
                    pass
        if not dated_files:
            raise FileNotFoundError("No model file found")
        _, latest = max(dated_files)
        return latest

    if USE_S3:
        bucket = os.getenv("MODEL_BUCKET", "propmind-models")
        objects = s3.list_objects_v2(Bucket=bucket).get("Contents", [])
        all_files = [obj["Key"] for obj in objects]

        reg_file = get_latest(pattern["regressor"], all_files)
        class_file = get_latest(pattern["classifier"], all_files)

        reg_obj = s3.get_object(Bucket=bucket, Key=reg_file)
        class_obj = s3.get_object(Bucket=bucket, Key=class_file)

        reg = joblib.load(io.BytesIO(reg_obj["Body"].read()))
        clf = joblib.load(io.BytesIO(class_obj["Body"].read()))
    else:
        model_dir = os.getenv("MODEL_DIR", "../models")
        all_files = os.listdir(model_dir)

        reg_file = get_latest(pattern["regressor"], all_files)
        class_file = get_latest(pattern["classifier"], all_files)

        reg = joblib.load(os.path.join(model_dir, reg_file))
        clf = joblib.load(os.path.join(model_dir, class_file))

    return clf, reg


from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


def limpiar_dataset_pavimento(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = ["tramo_id", "fecha", "antiguedad", "transito", "tipo_falla", "severidad", "pci_observado"]
    df = df.dropna(subset=req).copy()

    df["antiguedad"] = pd.to_numeric(df["antiguedad"], errors="coerce")
    df["transito"] = pd.to_numeric(df["transito"], errors="coerce")
    df["pci_observado"] = pd.to_numeric(df["pci_observado"], errors="coerce")
    df = df.dropna(subset=["antiguedad", "transito", "pci_observado"])

    df = df[
        (df["antiguedad"] >= 0) & (df["antiguedad"] <= 60) &
        (df["transito"] >= 0) &
        (df["pci_observado"] >= 0.0) & (df["pci_observado"] <= 100.0)
    ].copy()

    df["tramo_id"] = df["tramo_id"].astype(str).str.strip()
    df["tipo_falla"] = df["tipo_falla"].astype(str).str.strip()
    df["severidad"] = df["severidad"].astype(str).str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def entrenar_modelo_pci(csv_path: Path = None) -> dict:
    base_dir = Path(__file__).resolve().parent.parent
    if csv_path is None:
        csv_path = base_dir / "data" / "pavimento_huanuco_dataset.csv"

    df = limpiar_dataset_pavimento(csv_path)
    X = df[["antiguedad"]].copy()
    y = df["pci_observado"].copy()
    groups = df["tramo_id"].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred_test = modelo.predict(X_test)
    r2 = float(r2_score(y_test, y_pred_test))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    rmse = float(root_mean_squared_error(y_test, y_pred_test))

    artefacto = {
        "modelo": modelo,
        "pendiente": float(modelo.coef_[0]),
        "intercepto": float(modelo.intercept_),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "antiguedad_min": float(df["antiguedad"].min()),
        "antiguedad_max": float(df["antiguedad"].max()),
    }

    out_model = Path(__file__).resolve().parent / "pavement_lr_model.joblib"
    joblib.dump(artefacto, out_model)
    print(f"Modelo guardado en {out_model} | R2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    return artefacto


if __name__ == "__main__":
    entrenar_modelo_pci()

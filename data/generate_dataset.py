from pathlib import Path
import numpy as np
import pandas as pd


def generar_dataset_pavimento(n_tramos: int = 16, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tramos = [
        "JR-28-DE-JULIO-T1", "JR-28-DE-JULIO-T2",
        "AV-ALAMEDA-PERU-T1", "AV-ALAMEDA-PERU-T2",
        "JR-HUALLAYCO-T1", "JR-HUALLAYCO-T2",
        "AV-MARANON-T1", "AV-MARANON-T2",
        "JR-GENERAL-PRADO-T1", "JR-GENERAL-PRADO-T2",
        "AV-COLONIAL-T1", "AV-COLONIAL-T2",
        "JR-DOS-DE-MAYO-T1", "JR-DOS-DE-MAYO-T2",
        "AV-AYACUCHO-T1", "AV-AYACUCHO-T2",
    ][:n_tramos]

    patologias = [
        "Grietas longitudinales",
        "Grietas transversales",
        "Grietas de esquina",
        "Losa dividida",
        "Despostillamiento",
        "Escalonamiento",
    ]
    severidades = ["Baja", "Media", "Alta"]
    pesos_sev = {"Baja": 2.0, "Media": 6.0, "Alta": 12.0}

    filas = []
    fechas = pd.date_range("2022-01-10", "2026-10-25", periods=64)
    idx_reg = 0

    for tramo in tramos:
        for _ in range(4):
            antiguedad = float(np.clip(rng.integers(1, 13), 1, 12))
            transito = int(rng.integers(400, 6500))
            tipo_falla = str(rng.choice(patologias))
            severidad = str(rng.choice(severidades, p=[0.4, 0.38, 0.22]))

            pci_sim = (
                102.0
                - 6.4 * antiguedad
                - 0.0018 * transito
                - pesos_sev[severidad]
                + float(rng.normal(0.0, 3.2))
            )
            pci_observado = round(float(np.clip(pci_sim, 0.0, 100.0)), 1)

            filas.append({
                "tramo_id": tramo,
                "fecha": fechas[idx_reg % len(fechas)].strftime("%Y-%m-%d"),
                "antiguedad": antiguedad,
                "transito": transito,
                "tipo_falla": tipo_falla,
                "severidad": severidad,
                "pci_observado": pci_observado,
                "origen_dato": "simulado_educativo_huanuco_ref",
            })
            idx_reg += 1

    return pd.DataFrame(filas)


if __name__ == "__main__":
    out_csv = Path(__file__).resolve().parent / "pavimento_huanuco_dataset.csv"
    df = generar_dataset_pavimento()
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Dataset generado: {len(df)} registros en {out_csv}")

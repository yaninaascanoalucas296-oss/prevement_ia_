# Pavement AI

Aplicación educativa en Streamlit para explorar y estimar el estado de pavimentos en Huánuco. Incluye un conjunto de datos simulado, un modelo de regresión lineal y herramientas para registrar patologías.

## Requisitos

- Python 3.10 o superior
- `pip`

## Instalación y ejecución

Desde la carpeta del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/app.py
```

La aplicación se abrirá en el navegador. Las imágenes cargadas y los registros de patologías se guardan localmente y no se incluyen en Git.

## Estructura

- `app/app.py`: aplicación web Streamlit.
- `data/pavimento_huanuco_dataset.csv`: datos educativos simulados.
- `data/generate_dataset.py`: generador del conjunto de datos.
- `model/pavement_lr_model.joblib`: modelo entrenado.
- `model/train_model.py`: script de entrenamiento del modelo.

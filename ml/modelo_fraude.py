"""Fraud model wrapper, registered together with the model via `code_paths`.

It lives in a module, not in the notebook, because cloudpickle stores the code
of a class defined in the notebook by value, and serialized code doesn't cross
Python versions: the model saved on serverless with 3.10 wouldn't load in the
job with 3.12. Imported from a module, it is stored by reference.
"""

import mlflow
import numpy as np
import pandas as pd


class ModeloFraude(mlflow.pyfunc.PythonModel):
    """Returns the calibrated probability of fraud, not the class."""

    def __init__(self, modelo, colunas_cat, colunas_num):
        self.modelo, self.colunas_cat, self.colunas_num = modelo, colunas_cat, colunas_num

    def predict(self, context, entrada: pd.DataFrame, params=None) -> pd.DataFrame:
        x = entrada.copy()
        for c in self.colunas_num:
            x[c] = pd.to_numeric(x[c].astype("float64"), errors="coerce") if c in x else np.nan
        for c in self.colunas_cat:
            x[c] = x[c].astype("string").fillna("ausente") if c in x else "ausente"
        p = self.modelo.predict_proba(x[self.colunas_cat + self.colunas_num])[:, 1]
        return pd.DataFrame({"score_fraude": p})

# src/preprocessor.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_base_cost(resid_returns, sectors, adv_tl):
    # korelasyon + sektör + likidite
    # ... (önceki kod)
    return C_base
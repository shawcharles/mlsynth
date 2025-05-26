"""
FDID Estimation
================
FDID Plot
"""

import pandas as pd
from mlsynth import FDID
from mlsynth.config_models import FDIDConfig # Import FDIDConfig
import os

# Load the CSV file using pandas
df = pd.read_csv('https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/HongKong.csv')

treat = "Integration"
outcome = "GDP"
unitid = "Country"
time = "Time"

config = FDIDConfig(
    df=df,
    treat=treat,
    time=time,
    outcome=outcome,
    unitid=unitid,
    counterfactual_color="blue",
    treated_color="black",
    display_graphs=True
)

model = FDID(config=config)

FMAest = model.fit()

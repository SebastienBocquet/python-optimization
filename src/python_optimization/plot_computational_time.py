import numpy as np
import plotly.express as px
import pandas as pd

CPU_LABEL = "time (s) on CPU"
GPU_LABEL = "time (s) on GPU"

# Computational time in s
data = {
    "label": ["naive", "numba", "vectorized-numpy", "vectorized-jax-ops", "C-cuda"],
    CPU_LABEL: [30.*60, 30., 30., 15., np.nan],
    GPU_LABEL: [30. * 60, 30., 10., 1., 0.16],
}

fig = px.bar(pd.DataFrame.from_dict(data), x="label", y=[CPU_LABEL, GPU_LABEL], log_y=True, barmode="group")
fig.show()

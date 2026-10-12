"""Benchmark helpers that report measured, rather than fabricated, metrics."""
from __future__ import annotations
import time
import numpy as np

def relative_l2(prediction, reference, eps=1e-12):
    p,r=np.asarray(prediction),np.asarray(reference)
    return float(np.linalg.norm(p-r)/max(np.linalg.norm(r),eps))

def benchmark_model(model, inputs, reference=None):
    start=time.perf_counter(); result=model(inputs); elapsed=time.perf_counter()-start
    out={"runtime_seconds":elapsed,"prediction":result.detach().cpu().numpy()}
    if reference is not None: out["relative_l2"]=relative_l2(out["prediction"],reference)
    return out

def summarize(values):
    values=np.asarray(values,dtype=float)
    return {"mean":float(values.mean()),"std":float(values.std(ddof=1)) if values.size>1 else 0.,
            "min":float(values.min()),"max":float(values.max())}

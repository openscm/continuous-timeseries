# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import concurrent.futures
import multiprocessing

from tqdm.auto import tqdm

from continuous_timeseries.scratch import progresser

# %%
L = list(range(9))
n_processes = 4

# %%
mp_context = multiprocessing.get_context("fork")
with concurrent.futures.ProcessPoolExecutor(
    max_workers=n_processes, mp_context=mp_context
) as pool:
    futures = [pool.submit(progresser, n) for n in tqdm(L, desc="submitting to pool")]
    res = [
        future.result()
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            desc="Retrieving parallel results",
            total=len(futures),
        )
    ]

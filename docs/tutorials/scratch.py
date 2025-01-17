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

import time
import random
from tqdm.auto import tqdm
from multiprocessing import Pool, freeze_support, RLock, cpu_count, set_start_method
# from multiprocessing import set_start_method

def func(pid, n):
    
    print(end="\r")
    tqdm_text = "#" + "{}".format(pid).zfill(3)

    current_sum = 0
    with tqdm(total=n, desc=tqdm_text, position=pid+1) as pbar:
        for i in range(1, n+1):
            current_sum += i
            time.sleep(0.05)
            pbar.update(1)

    return current_sum

# num_processes = cpu_count()
num_jobs = 10
random_seed = 0
random.seed(random_seed) 
# set_start_method("fork")

pool = Pool(processes=50, initargs=(RLock(),), initializer=tqdm.set_lock)

argument_list = [random.randint(0, 100) for _ in range(num_jobs)]

jobs = [pool.apply_async(func, args=(i,n,)) for i, n in enumerate(argument_list)]
pool.close()
results =[]
for job in tqdm(jobs, desc='Outer loop'):
    results.append(job.get())

#print("\n" * (len(argument_list) + 1))


# %%
from multiprocessing import Pool, RLock
from time import sleep

from tqdm.auto import tqdm, trange

from continuous_timeseries.scratch import progresser

L = list(range(9))


# def progresser(n):
#     interval = 0.001 / (n + 2)
#     total = 5000
#     text = f"#{n}, est. {interval * total:<04.2}s"
#     for _ in trange(total, desc=text, position=n):
#         sleep(interval)




# %%
tqdm.set_lock(RLock())
p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
p.map(progresser, L)


# %%

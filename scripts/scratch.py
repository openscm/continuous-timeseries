from multiprocessing import Pool, RLock
from time import sleep

from tqdm.auto import tqdm, trange

L = list(range(9))[::-1]


def progresser(n):
    interval = 0.005 / (n + 2)
    total = 5000
    text = f"#{n}, est. {interval * total:<04.2}s"
    for _ in trange(total, desc=text, position=n):
        sleep(interval)


if __name__ == "__main__":
    tqdm.set_lock(RLock())
    p = Pool(processes=4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    p.map(progresser, L)

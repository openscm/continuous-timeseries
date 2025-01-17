from time import sleep

import tqdm.notebook


def progresser(n):
    print(end=" ")
    interval = 0.001 / (n + 2)
    total = 5000
    text = f"#{n}, est. {interval * total:<04.2}s"
    # for _ in trange(total, desc=text, position=n):
    for _ in tqdm.notebook.trange(total, desc=text, position=n):
        sleep(interval)

# %%
import random
from animals import Herbivore, Island, HarshIsland
from matplotlib import pylab as plt

# %%
A = Herbivore(10)

# %%
A.age

# %%
A._age()
# %%
A.age

# %%
random.seed(123)
A2 = A.breed()
A2.survival_skill

# %%
I = Island(initial_pop=10, max_pop=100)

# %%
len(I.animals)

# %%
stats = I.compute_epoches(15)
stats[14]

# %%
params = {"initial_pop": 10, "max_pop": 100}
years, N_islands = 15, 1000

islands = [Island(**params) for _ in range(N_islands)]
stats = [island.compute_epoches(years) for island in islands]

# %%
params = {"initial_pop": 10, "max_pop": 40, "env_range": [20, 80]}
years, N_islands = 15, 1000

h_islands = [HarshIsland(**params) for _ in range(N_islands)]
h_stats = [island.compute_epoches(years) for island in h_islands]

# %%
datas = {"Heaven Islands": stats, "Harsh Islands": h_stats}

colors = {"Heaven Islands": "blue", "Harsh Islands": "red"}


# %%
fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)

for i, title in enumerate(
    ("Population", "Average age", "Average Survival Skill", "% with SSK > 75")
):
    axes[i][0].set_ylabel(title)

    axes[i][0].set_xlim(0, 15)
    axes[i][1].set_xlim(0, 15)

for i, (k, v) in enumerate(datas.items()):
    axes[0][i].set_title(k, fontsize=14)

    for s in v:
        years = list(s.keys())

        axes[0][i].plot(
            years, [v["pop"] for v in s.values()], c=colors[k], label=k, alpha=0.007
        )
        axes[1][i].plot(
            years,
            [v.get("mean_age", None) for v in s.values()],
            c=colors[k],
            label=k,
            alpha=0.007,
        )
        axes[2][i].plot(
            years,
            [v.get("mean_skill", None) for v in s.values()],
            c=colors[k],
            label=k,
            alpha=0.007,
        )
        axes[3][i].plot(
            years,
            [v.get("75_skill", None) for v in s.values()],
            c=colors[k],
            label=k,
            alpha=0.007,
        )

# %%

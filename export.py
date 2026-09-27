import json, numpy as np, sim
teams = sim.make_teams(); n = len(teams)
budget = 2.0 * sim.G * sim.T / n
kind = {"trading": 0, "research": 1, "backtest": 2}
# pick a seed whose priority vs backfill gap is close to the average, so the picture is typical
base_seed, best = None, 1e9
for s in range(120):
    b = sim.generate_jobs(np.random.default_rng(s), teams)
    gv = sim.run("greedy", sim.clone(b), n, budget)["value"]
    p = sim.run("priority", sim.clone(b), n, budget)
    q = sim.run("priority", sim.clone(b), n, budget, backfill=True)
    d = abs((q["util"] - p["util"]) - 0.104) + abs((q["value"] - p["value"]) / gv - 0.102) + abs(p["value"]/gv - 0.693)
    if d < best: best, base_seed = d, s
base = sim.generate_jobs(np.random.default_rng(base_seed), teams)
g = sim.run("greedy", sim.clone(base), n, budget)["value"]
out = {"G": sim.G, "T": sim.T, "seed": base_seed, "mechs": {}}
for name, m, kw in [("priority","priority",{}),("backfill","priority",{"backfill":True}),
                    ("auction","auction",{}),("fairshare","fairshare",{})]:
    sim.LOG = []
    r = sim.run(m, sim.clone(base), n, budget, **kw)
    grid = []
    for slot in sim.LOG:
        col = []
        for team, gg in sorted(slot, key=lambda x: kind[teams[x[0]]]):
            col += [kind[teams[team]]] * gg
        col += [-1] * (sim.G - len(col))
        grid.append(col)
    out["mechs"][name] = {"grid": grid, "util": r["util"], "value": r["value"] / g}
    sim.LOG = None
json.dump(out, open("grids.json", "w"), separators=(",", ":"))
print(base_seed, {k: (round(v["util"],3), round(v["value"],3)) for k, v in out["mechs"].items()})

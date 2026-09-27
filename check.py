import numpy as np, sim
teams = sim.make_teams(); n = len(teams); liar = teams.index("research")
budget = 2.0 * sim.G * sim.T / n
res = {k: [] for k in ["greedy","priority","priority+backfill","util_p","util_pb",
       "sel_gain","sel_welf","sel_gain_bf","auc15_gain","auc15_welf"]}
for s in range(200):
    base = sim.generate_jobs(np.random.default_rng(s), teams)
    C = lambda: sim.clone(base)
    g = sim.run("greedy", C(), n, budget)
    p = sim.run("priority", C(), n, budget)
    pb = sim.run("priority", C(), n, budget, backfill=True)
    ps = sim.run("priority", C(), n, budget, liar=liar, lie="selective")
    pbs = sim.run("priority", C(), n, budget, liar=liar, lie="selective", backfill=True)
    a = sim.run("auction", C(), n, budget)
    a15 = sim.run("auction", C(), n, budget, liar=liar, inflate=1.5)
    res["greedy"].append(g["value"]); res["priority"].append(p["value"]); res["priority+backfill"].append(pb["value"])
    res["util_p"].append(p["util"]); res["util_pb"].append(pb["util"])
    res["sel_gain"].append(ps["team_value"][liar]-p["team_value"][liar]); res["sel_welf"].append(ps["value"]-p["value"])
    res["sel_gain_bf"].append(pbs["team_value"][liar]-pb["team_value"][liar])
    res["auc15_gain"].append(a15["team_value"][liar]-a["team_value"][liar]); res["auc15_welf"].append(a15["value"]-a["value"])
R = {k: np.array(v) for k,v in res.items()}; ref = R["greedy"].mean()
se = lambda x: x.std(ddof=1)/np.sqrt(len(x))
print(f"priority          value {R['priority'].mean()/ref:.1%}  util {R['util_p'].mean():.1%}")
print(f"priority+backfill value {R['priority+backfill'].mean()/ref:.1%}  util {R['util_pb'].mean():.1%}")
d = R['priority+backfill']-R['priority']; print(f"  backfill gain {d.mean()/ref:+.1%} of greedy (se {se(d)/ref:.1%}), worse in {np.mean(d<0):.0%} of runs")
for k in ["sel_gain","sel_gain_bf","auc15_gain"]:
    print(f"{k:<12} {R[k].mean():+7.1f} se {se(R[k]):.1f}   liar better off in {np.mean(R[k]>0):.0%} of runs")
print(f"selective lie welfare {R['sel_welf'].mean()/ref:+.1%}; auction 1.5x welfare {R['auc15_welf'].mean()/ref:+.1%}")

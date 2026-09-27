import numpy as np, sim
base_rates = {k: v["rate"] for k, v in sim.ARCHETYPES.items()}
for scale in (0.8, 1.0, 1.4):
    for k in sim.ARCHETYPES: sim.ARCHETYPES[k]["rate"] = base_rates[k] * scale
    out, dr = sim.experiment(n_seeds=150)
    ref = np.mean(out["greedy"]["value"])
    print(f"\n== demand/capacity {dr:.2f}x ==")
    for m, r in out.items():
        kinds = [np.mean(r["by_kind"][k]) / np.mean(out['greedy']['by_kind'][k]) for k in sim.ARCHETYPES]
        line = f"{m:<10} value {np.mean(r['value'])/ref:6.1%} util {np.mean(r['util']):6.1%}  " + " ".join(f"{k[:5]} {x:6.1%}" for k, x in zip(sim.ARCHETYPES, kinds))
        if m in ("priority","auction"):
            g=np.array(r["liar_gain"]); w=np.array(r["welfare_change"])
            line += f"  | liar {g.mean():+7.1f}±{g.std(ddof=1)/np.sqrt(len(g)):.1f} welfare {w.mean()/ref:+.1%}"
        print(line)

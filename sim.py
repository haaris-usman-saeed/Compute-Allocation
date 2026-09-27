"""
Compute allocation under scarcity: comparing internal allocation mechanisms.

Model
  - A GPU cluster with fixed capacity G, simulated over T time slots.
  - Teams submit jobs. Each job needs (width x duration) GPU-slots, can use at most
    `width` GPUs per slot, and pays off `value` only if it finishes by its deadline.
  - Each team gets an equal budget of internal currency ("bucks") per run.

Mechanisms
  greedy     : central planner who knows true values, schedules by value density.
               Not a true optimum, but a strong reference point. No budgets.
  priority   : stylized "priority tier" bidding. Teams tag jobs tier 1/2/3; higher
               tiers cost more bucks per GPU-slot and are served first (EDF within tier).
  auction    : uniform clearing price auction each slot. Jobs bid bucks per GPU-slot,
               capped by what the team's remaining budget can support. All winners pay
               the marginal (highest losing) bid.
  fairshare  : each team is guaranteed an equal slice of the cluster each slot; unused
               slices are pooled and handed out earliest-deadline-first. No bidding.

Strategic test
  One research team misreports: declares every job top tier (priority) or inflates
  bids 3x (auction). Fair share has nothing to misreport.
"""

import numpy as np
from dataclasses import dataclass

G = 64          # GPUs
T = 72          # time slots
TIER_COST = {1: 1.0, 2: 2.0, 3: 4.0}
TIER_CUTOFFS = (2.0, 5.0)   # value density thresholds for honest tier 2 / tier 3

# team archetypes: (count, arrival rate per slot, width range, duration range,
#                   deadline slack range, value density lognormal (mean log, sigma))
ARCHETYPES = {
    "trading":  dict(n=3, rate=0.45, width=(2, 8),  dur=(1, 3),  slack=(0, 2),  dens=(1.3, 0.6)),
    "research": dict(n=3, rate=0.125, width=(8, 32), dur=(4, 12), slack=(8, 30), dens=(0.5, 1.0)),
    "backtest": dict(n=2, rate=0.25, width=(4, 16), dur=(2, 6),  slack=(3, 10), dens=(0.2, 0.5)),
}


@dataclass
class Job:
    team: int
    arrival: int
    width: int
    need: int          # GPU-slots remaining
    total: int
    deadline: int
    density: float     # true value per GPU-slot
    value: float
    done: bool = False


def make_teams():
    teams = []
    for kind, a in ARCHETYPES.items():
        teams += [kind] * a["n"]
    return teams


def generate_jobs(rng, teams):
    jobs = []
    for i, kind in enumerate(teams):
        a = ARCHETYPES[kind]
        for t in range(T):
            for _ in range(rng.poisson(a["rate"])):
                w = int(rng.integers(a["width"][0], a["width"][1] + 1))
                d = int(rng.integers(a["dur"][0], a["dur"][1] + 1))
                s = int(rng.integers(a["slack"][0], a["slack"][1] + 1))
                dens = float(rng.lognormal(*a["dens"]))
                total = w * d
                jobs.append(Job(i, t, w, total, total, t + d + s, dens, dens * total))
    return jobs


def honest_tier(j):
    if j.density >= TIER_CUTOFFS[1]:
        return 3
    if j.density >= TIER_CUTOFFS[0]:
        return 2
    return 1


def feasible(j, t):
    """Can the job still finish by its deadline if given full width from now on?"""
    return j.need <= j.width * (j.deadline - t)


LOG = None   # set to a list to record per-slot allocations for visualization


def give(j, cap):
    g = min(j.width, j.need, cap)
    if LOG is not None and g > 0:
        LOG[-1].append((j.team, int(g)))
    j.need -= g
    if j.need == 0:
        j.done = True
    return g


def run(mechanism, jobs, n_teams, budget, liar=None, inflate=3.0, backfill=False, lie="all"):
    budgets = np.full(n_teams, budget, dtype=float)
    used = 0
    for t in range(T):
        if LOG is not None:
            LOG.append([])
        active = [j for j in jobs if j.arrival <= t and not j.done and feasible(j, t)]
        cap = G

        if mechanism == "greedy":
            for j in sorted(active, key=lambda j: (-j.density, j.deadline)):
                if cap == 0:
                    break
                cap -= give(j, cap)

        elif mechanism == "priority":
            def declared(j):
                if j.team != liar:
                    return honest_tier(j)
                if lie == "selective":
                    tight = j.deadline - t <= j.need / j.width + 2
                    return min(3, honest_tier(j) + 1) if tight else honest_tier(j)
                return 3
            for j in sorted(active, key=lambda j: (-declared(j), j.deadline)):
                if cap == 0:
                    break
                # drop to the highest tier the team can still afford
                tier = declared(j)
                g = min(j.width, j.need, cap)
                while tier >= 1 and budgets[j.team] < TIER_COST[tier] * g:
                    tier -= 1
                if tier == 0:
                    continue
                g = give(j, cap)
                budgets[j.team] -= TIER_COST[tier] * g
                cap -= g

            if backfill:
                for j in sorted([j for j in active if not j.done], key=lambda j: j.deadline):
                    if cap == 0:
                        break
                    cap -= give(j, cap)

        elif mechanism == "auction":
            team_need = np.zeros(n_teams)
            for j in active:
                team_need[j.team] += j.need
            bids = []
            for j in active:
                claim = j.density * (inflate if j.team == liar else 1.0)
                afford = budgets[j.team] / max(team_need[j.team], 1)
                bids.append((min(claim, afford), j))
            bids.sort(key=lambda b: -b[0])
            winners, price = [], 0.0
            for b, j in bids:
                if b <= 0:
                    continue
                if cap == 0:
                    price = b          # highest losing bid sets the price
                    break
                want = min(j.width, j.need)
                g = give(j, cap)
                cap -= g
                winners.append((j, g))
                if g < want:
                    price = b          # partially filled job is marginal
            for j, g in winners:
                budgets[j.team] -= price * g

        elif mechanism == "fairshare":
            quota = G // n_teams
            left = np.full(n_teams, quota)
            by_team = {}
            for j in active:
                by_team.setdefault(j.team, []).append(j)
            for team, js in by_team.items():
                for j in sorted(js, key=lambda j: (-j.density, j.deadline)):
                    if left[team] == 0:
                        break
                    left[team] -= give(j, left[team])
            cap = G - (quota * n_teams - left.sum())
            cap = int(cap)
            for j in sorted([j for j in active if not j.done], key=lambda j: j.deadline):
                if cap == 0:
                    break
                cap -= give(j, cap)

        used += G - cap

    realized = np.zeros(n_teams)
    for j in jobs:
        if j.done:
            realized[j.team] += j.value
    return dict(value=realized.sum(), team_value=realized, util=used / (G * T))


def clone(jobs):
    return [Job(**{k: getattr(j, k) for k in Job.__dataclass_fields__}) for j in jobs]


def experiment(n_seeds=200, seed0=0):
    teams = make_teams()
    n = len(teams)
    budget = 2.0 * G * T / n     # enough to run every fair-share slot at tier 2
    liar = teams.index("research")
    mechs = ["greedy", "priority", "auction", "fairshare"]
    out = {m: dict(value=[], util=[], by_kind={k: [] for k in ARCHETYPES},
                   liar_gain=[], welfare_change=[]) for m in mechs}
    demand_ratio = []

    for s in range(n_seeds):
        rng = np.random.default_rng(seed0 + s)
        base = generate_jobs(rng, teams)
        demand_ratio.append(sum(j.total for j in base) / (G * T))
        for m in mechs:
            r = run(m, clone(base), n, budget)
            out[m]["value"].append(r["value"])
            out[m]["util"].append(r["util"])
            for k in ARCHETYPES:
                idx = [i for i, t in enumerate(teams) if t == k]
                out[m]["by_kind"][k].append(r["team_value"][idx].sum())
            if m in ("priority", "auction"):
                rl = run(m, clone(base), n, budget, liar=liar)
                out[m]["liar_gain"].append(rl["team_value"][liar] - r["team_value"][liar])
                out[m]["welfare_change"].append(rl["value"] - r["value"])
    return out, np.mean(demand_ratio)


if __name__ == "__main__":
    out, dr = experiment()
    ref = np.mean(out["greedy"]["value"])
    print(f"Demand / capacity: {dr:.2f}x\n")
    print(f"{'mechanism':<11}{'value vs greedy':>16}{'utilization':>13}"
          f"{'trading':>10}{'research':>10}{'backtest':>10}")
    for m, r in out.items():
        v = np.mean(r["value"]) / ref
        kinds = [np.mean(r["by_kind"][k]) / np.mean(out['greedy']['by_kind'][k])
                 for k in ARCHETYPES]
        print(f"{m:<11}{v:>15.1%}{np.mean(r['util']):>13.1%}"
              + "".join(f"{x:>10.1%}" for x in kinds))
    print("\nOne research team misreports (all top tier / 3x bids):")
    for m in ("priority", "auction"):
        g = np.array(out[m]["liar_gain"]); w = np.array(out[m]["welfare_change"])
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
        print(f"  {m:<9} liar gain {g.mean():+8.1f} (se {se(g):.1f})   "
              f"firm welfare change {w.mean():+8.1f} (se {se(w):.1f})   "
              f"= {w.mean()/ref:+.1%} of greedy value")

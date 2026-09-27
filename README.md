# Compute allocation under scarcity

A small simulation comparing ways a trading firm could allocate scarce GPU capacity across
competing teams using an internal currency. Motivated by public reporting that Jane Street
allocates GPUs via an internal bidding currency with priority levels, and has said it cannot
get all the compute it wants in one place.

This is a stylized model with synthetic data. It does not claim to describe Jane Street's
actual system.

## Model
- 64 GPUs, 72 time slots, 8 teams of three types: trading (small, urgent, high value),
  research (large, flexible, high variance), backtest (medium, low value).
- A job pays its value only if it finishes by its deadline.
- Each team gets an equal budget of internal currency.

## Mechanisms
| Mechanism | Rule |
|---|---|
| greedy | Planner with true values, schedules by value density. Reference point, not an optimum. |
| priority | Jobs tagged tier 1/2/3, higher tiers cost more and run first. |
| priority + backfill | Same, but GPUs left idle are given free to waiting jobs. |
| auction | Uniform clearing price auction each slot, bids capped by budget. |
| fairshare | Equal guaranteed slice per team, unused slices pooled. |

## Results (demand about 1.5x capacity, 200 seeds)
| Mechanism | Value vs greedy | Utilization |
|---|---|---|
| priority | 69% | 86% |
| priority + backfill | 80% | 97% |
| fairshare | 83% | 97% |
| auction | 88% | 95% |

1. **Budgeted priority tiers idle GPUs.** When teams exhaust their currency, their jobs cannot
   run even while GPUs sit free. Adding a zero cost backfill queue recovers about 10 points of
   value and was worse in only 3% of runs.
2. **Budgets make lying self punishing.** A team that inflates its priority (or bids) burns
   budget faster and ends up worse off on average, whether it lies about every job or only
   near deadline ones. The currency is doing the incentive work.
3. **Large flexible jobs lose most under every mechanism**, because urgent small jobs win each
   slot. Research teams get 43% (priority) to 76% (auction) of their greedy value.

## Limits and next tests
- Parameters are invented; results should be read as directional.
- Budgets here are fixed per run. With "use it or lose it" budgets that refresh each period,
  inflating priority may start to pay. That is the most important next test.
- Jobs are divisible across slots; real GPU jobs often need gang scheduling.

## Run
    python sim.py      # main comparison
    python sweep.py    # sensitivity to scarcity level
    python check.py    # backfill fix and strategic misreporting

## Site
The write up lives in `site/index.html`, a single static file. `export.py` regenerates the
schedule data embedded in it.

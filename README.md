# arbitrage
Single cryptocurrency exchange arbitrage via Bellman-Ford algorithm.

Bellman-Ford algorithm implemented from Wikipedia pseudo-code. Unlike many others posted online it (a) works on incomplete graphs and (b) reports all negative cycles.

Code for faux executing the trades, labelled DRY RUN, is incomplete.

Output looks like:

```
Best negative-weight cycle [71, 3, 127, 0, 71]
EDU 47103.155911446054 BTC 0.0016969999999999997 GMX 0.019688915140775743 USDT 0.63556 cumulative rate: 1.000253757935036
```

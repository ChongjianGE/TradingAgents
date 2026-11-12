# Trading loop on an Alpaca paper account: record as of 2026-11-11 19:09 ET

A copy of everything the `tradeloop` run has written so far, taken while the run was
still going: the first of three sessions (2026-11-11 to 2026-11-13) had closed, and the
19:00 after-hours tick was the last one in it. Nothing here is from a live account.

Five tickers (NVDA, CSCO, HD, DIS, GEV), one tick an hour, orders sent to an Alpaca
**paper** account.

## At the time of the copy

| | |
|---|---|
| Ticks run | 17 |
| Analyses | 29, none failed |
| Orders sent to the paper account | 13: 5 purchases, all filled (7 fills), and 8 protective stops |
| Sales | none |
| Positions | NVDA 65, DIS 126, GEV 15 |
| Account equity | $99,117 from $100,000, marked after hours |
| Model spend | $23.91 |

Five of the analyses and two of the ticks (03:33 and 03:40) are a trial made before the
run was started, with the simulated book; they are in the same database.

## What is where

- `dashboard/index.html`, `dashboard/day-2026-11-11.html`: the pages the loop builds.
  Open them in a browser; they need nothing from the network. Start here.
- `tradeloop.db`: the loop's SQLite database, copied whole: every tick, what was seen,
  each analysis and its decision, what the rules did with it, orders, fills, prices.
- `runs/<run id>/`: one folder per analysis. `request.json` is what it was told,
  `state.json` what each agent wrote, `steps.jsonl` every model and tool call.
- `run.log`: the loop's output, including each restart.
- `config.toml`: the settings it ran with. It names a key file by path; no key is in it.
- `pipeline/`: what the analysis pipeline itself wrote beside the loop: its memory log,
  the per-ticker state logs, and the report of the single NVDA run of 2026-11-10 made
  with the stock CLI.

## Read with care

- The code changed during the day, and the loop was restarted each time. In order:
  stops wait for their order to finish filling (`7aa5b0c`, 14:48); each analysis is told
  what a target weight buys (`3ba109d`, 16:05); the rest are dashboard changes. Decisions
  before 16:05 were sized at about twice what the model's own text intended.
- The reports, including what they say about companies and people in the news, were
  written by language models from sources they fetched. Nobody has checked them.
- One day of trades says nothing about whether any of this works.

# Reports

Rendered HTML from this replication. Open directly in a browser — each is
self-contained (images embedded as data URIs, no external assets).

| file | what it shows |
|---|---|
| `rank-reconstructions.html` | Original vs SVD vs HadNet vs SVD-at-matched-storage, two images × eight ranks (r=2…128), with error, PSNR and stored-value counts. Interactive |
| `all-tables.html` | The paper's Tables 2/5/6 beside re-measured values, plus the matched-storage column the paper omits, plus before/after for all seven methods |
| `change-ledger.html` | Every change made to the released code, with the paper's numbers alongside |
| `presentation-brief.html` | What to present to a supervisor, what to avoid claiming, and prioritised next steps |

Data behind them:

| file | contents |
|---|---|
| `all-tables.data.json` | Paper tables transcribed from the PDF + matched-storage measurements |
| `measured.json` | Parsed baseline/optimised results from `../bench/*.log` |

Reconstruction images live in `../bench/recon_mri/` and `../bench/recon_cat/`.

**Note:** these pages are also published as artifacts. If you regenerate one,
republish rather than creating a new link.

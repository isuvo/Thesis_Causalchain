# CKG Mining Report

- **Mined from:** `C:\Users\MSHUVO23\Desktop\Thesis Research\Thesis-causal-vul\notebooks\Dataset\train\hetero_ready_gcbert`
- **Graphs used:** 3438
- **Created at:** 2025-10-22 09:18:46

## Edge Priors (counts & probabilities)

| Relation | Count | Prob |
|---|---:|---:|
| CFG | 12970 | 0.4744 |
| CALL | 10670 | 0.3903 |
| ARG2PARAM | 2478 | 0.0906 |
| DFG_THIN | 1055 | 0.0386 |
| DFG | 114 | 0.0042 |
| RET2CALL | 52 | 0.0019 |
| RET2LHS | 1 | 0.0000 |

## Start/End Relation Frequencies

| Relation | Start Count | End Count |
|---|---:|---:|
| DFG | 11 | 27 |
| CFG | 6494 | 1965 |
| CALL | 170 | 3612 |
| ARG2PARAM | 50 | 617 |
| RET2CALL | 50 | 2 |
| RET2LHS | 0 | 1 |
| DFG_THIN | 60 | 611 |

## Hop Histogram (path length in edges)

| Hops | Count |
|---:|---:|
| 0 | 40 |
| 4 | 6835 |

## Top Motifs

| # | Count | Relations (tri-gram) |
|---:|---:|---|
| 1 | 3943 | CFG -> CFG -> CFG |
| 2 | 3489 | CALL -> CALL -> CALL |
| 3 | 3343 | CFG -> CALL -> CALL |
| 4 | 810 | CFG -> ARG2PARAM -> ARG2PARAM |
| 5 | 480 | ARG2PARAM -> ARG2PARAM -> ARG2PARAM |
| 6 | 348 | ARG2PARAM -> ARG2PARAM -> DFG_THIN |
| 7 | 288 | CFG -> CFG -> CALL |
| 8 | 180 | CFG -> CFG -> ARG2PARAM |
| 9 | 135 | CFG -> DFG_THIN -> DFG_THIN |
| 10 | 116 | CFG -> ARG2PARAM -> DFG_THIN |
| 11 | 111 | DFG_THIN -> DFG_THIN -> DFG_THIN |
| 12 | 48 | CFG -> DFG_THIN -> ARG2PARAM |
| 13 | 47 | CFG -> CFG -> DFG_THIN |
| 14 | 38 | DFG_THIN -> ARG2PARAM -> ARG2PARAM |
| 15 | 33 | DFG_THIN -> DFG_THIN -> ARG2PARAM |
| 16 | 32 | ARG2PARAM -> DFG_THIN -> DFG_THIN |
| 17 | 29 | CFG -> DFG -> DFG |
| 18 | 28 | RET2CALL -> CALL -> CALL |
| 19 | 22 | DFG_THIN -> CALL -> CALL |
| 20 | 20 | DFG -> DFG -> DFG |

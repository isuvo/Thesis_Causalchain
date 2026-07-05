# Evaluation Report

- **Model:** `C:\Users\MSHUVO23\Desktop\Thesis Research\Thesis-causal-vul\notebooks\out\cvul\1760989836\model.pt`
- **CKG:** `C:\Users\MSHUVO23\Desktop\Thesis Research\Thesis-causal-vul\notebooks\ckg\ckg.json`
- **Created at:** 2025-10-26 16:24:06

## Split: valid
- Graphs used: **2906**
- Beam avg length (edges): **4.732**
- Inter-procedural ratio: **0.024**
- Motif-hit ratio: **0.931**
- Precision / Recall / F1: n/a (no labels found)
- CFAM mean: **0.024762**
- CCS mean: **0.000000**

## Split: test
- Graphs used: **2915**
- Beam avg length (edges): **4.691**
- Inter-procedural ratio: **0.024**
- Motif-hit ratio: **0.919**
- Precision / Recall / F1: n/a (no labels found)
- CFAM mean: **0.025270**
- CCS mean: **0.000000**

## Calibrated thresholds (by source)

- default: **0.25**

## CKG summary used

- Relations: DFG, CFG, CALL, ARG2PARAM, RET2CALL, RET2LHS, DFG_THIN
- Top edge counts:
  - CFG: 12970
  - CALL: 10670
  - ARG2PARAM: 2478
  - DFG_THIN: 1055
  - DFG: 114
  - RET2CALL: 52
  - RET2LHS: 1
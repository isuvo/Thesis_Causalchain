# Project Structure

High-level layout of code, data, and reports.

- Root: research code, notebooks, reports, utilities, training pipeline.
- `notebooks`: exploratory notebooks and demos.
- `src`: preprocessing, training, inference, and utilities.
- `Report`: figures, evaluation outputs, and the full report.
- `tools`: helper scripts for environment and PDG validation.

```
Thesis-causal-vul/
├─ WORKPLAN.md
├─ README.md
├─ commands.txt
├─ ScripstList.txt
├─ requirements.txt
├─ notebooks/
│  ├─ 01_ReposVul.ipynb
│  ├─ 02_word2vec_training.ipynb
│  ├─ 03_causal_chain_demo.ipynb
│  └─ colab.ipynb
├─ tools/
│  ├─ export_pdg_env.sc
│  └─ validate_pdg_dir.py
├─ Report/
│  ├─ Causalchain_pretty.png
│  ├─ Causalchain_raw.png
│  ├─ eval_colab/
│  │  ├─ eval_report.json
│  │  └─ test_preds.csv
│  └─ ReposVul_report/
│     ├─ ReposVul_CCPP_Full_Report.md
│     └─ _edges/
│        ├─ test_c_cpp_repository2__caller_func_callee.csv
│        ├─ train_c_cpp_repository2__caller_func_callee.csv
│        └─ valid_c_cpp_repository2__caller_func_callee.csv
└─ src/
   ├─ __init__.py
   ├─ build_dataset_jsonl.py
   ├─ infer_one_slice_gcbert.py
   ├─ infer_one_slice_pretty.py
   ├─ step4_train_gnn_attn.py
   ├─ step4_train_gnn_attn_v1.py
   ├─ step5_export_chains.py
   ├─ step6_eval_report.py
   ├─ train.py
   ├─ preprocess/
   │  ├─ __init__.py
   │  ├─ embed_lines_gcbert.py
   │  ├─ slices_to_pyg_gcbert.py
   │  ├─ step1_prepare_diversevul.py
   │  ├─ step2_generate_pdg.py
   │  ├─ step2c_pipeline_validate_and_slice.py
   │  ├─ transformer_cache.py
   │  └─ external/
   │     ├─ program_slice.py
   │     ├─ sensiAPI.txt
   │     ├─ sensiAPI_A.txt
   │     └─ sensiAPI_B.txt
   └─ utils/
      ├─ hashing.py
      └─ pdg_io.py
```


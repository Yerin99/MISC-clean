# MISC-clean

This repository provides a **cleaned and executable implementation** of the ACL 2022 paper:

> *MISC: A MIxed Strategy-Aware Model Integrating COMET for Emotional Support Conversation*  
> (Tu et al., ACL 2022)

The [original github of MISC](https://github.com/morecry/MISC) released codes only via a [Google Drive Link](https://drive.google.com/file/d/1QX4_QhYpoF5k-LeX6s2OfD65CdkEHo7x/view) in an unsorted form.  

This repository reorganizes the codebase, adds installation guides, and ensures reproducibility for easier research and development.


## Quickstart (training → eval → generation)

1) Install dependencies (single command)

```bash
pip install -r requirements.txt
```

2) Prepare data and model

- Data directory `dataset/` should contain the ESConv+COMET files expected by the code
  (`trainWithStrategy_short.tsv`, `devWithStrategy_short.tsv`, `testWithStrategy_short.tsv`,
  `trainComet.txt`, `devComet.txt`, `testComet.txt`, `trainComet_st.txt`, `devComet_st.txt`, `testComet_st.txt`,
  `trainSituation.txt`, `devSituation.txt`, `testSituation.txt`).
- Model snapshot
  - Use a local snapshot at `./blenderbot_small-90M/` (recommended for stability), or
  - Let it be fetched automatically from Hugging Face (cached under `./blender-small/`).
  - To download explicitly:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="facebook/blenderbot_small-90M",
                  local_dir="./blenderbot_small-90M",
                  local_dir_use_symlinks=False)
print("downloaded to ./blenderbot_small-90M")
PY
```

3) Run training (with safer memory settings)

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
from BlenderEmotionalSupport import Args, main
a = Args()
a.no_cuda = False
a.model_name_or_path = "./blenderbot_small-90M"  

# memory-friendly settings
a.per_gpu_train_batch_size = 5
a.per_gpu_eval_batch_size = 10
a.gradient_accumulation_steps = 4

main(a)
PY
```
- Optional: to remove strategy tokens from inputs as well, set `a.strategy = False`.

### Ablation (user-added): OffStrategy disables mixed strategy injection

```bash
CUDA_VISIBLE_DEVICES=1 python3 - <<'PY'
from BlenderEmtionalSupportOffStrategy import Args, main
a = Args()
a.no_cuda = False
a.model_name_or_path = "./blenderbot_small-90M"

# memory-friendly settings
a.per_gpu_train_batch_size = 5
a.per_gpu_eval_batch_size = 10
a.gradient_accumulation_steps = 4

main(a)
PY
```
- Note: this script is an ablation I added to remove the mixed-strategy injection module.
- Optional: to remove strategy tokens from inputs as well, set `a.strategy = False`.

Artifacts:
- checkpoints: `blender_strategy/<TAG>/` (e.g., `off_strategy`)
- generations and summary: `generated_data/<TAG>/`


## Notable code adjustments

- NumPy deprecated alias removal in `evaluate` (file: `BlenderEmotionalSupport.py`)
  - Replaced `np.int` with built-in `int` in two places to support NumPy ≥1.24.
- Debug stopper removed in `generate` (file: `BlenderEmotionalSupport.py`)
  - A leftover `print(1/0)` that halted generation has been commented out.

These edits make the code runnable on modern environments while preserving original behavior.

Notes:
- We use the local `src/transformers` (do not install `transformers` from pip).
- Requirements pin `tokenizers==0.9.4` per local runtime checks.


## Environment notes

- Recommended versions: Python 3.9, CUDA driver ≥ 12.2, PyTorch 2.3.1+cu121.
- The local `src/transformers` performs runtime checks and expects `tokenizers==0.9.4`, `sacremoses`, `psutil`, etc. The provided `requirements.txt` pins compatible versions.
- For METEOR metric (via `metric/pycocoevalcap/meteor`), a Java runtime (OpenJDK) is required.


## Citation

Please refer to the original paper for details of the method and evaluation protocol: `https://arxiv.org/pdf/2203.13560`.

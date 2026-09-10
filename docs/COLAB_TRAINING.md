# Training on Google Colab (IO-VNBD denoise / odom models)

Repo pushes intentionally exclude everything >10 MB (see `.gitignore` + `AGENTS.md`):
data, `*_windows.npz`, per-fold `*fold_*.pt`, `*.graphml`, `cache/`.
So Colab starts from **code only** — you supply data + download weights back.

## 1. Clone + install (Colab cell)

```bash
!git clone https://github.com/zenodefault/GatiNav.git
%cd GatiNav
!pip install -q torch numpy scipy
```

No `requirements.txt` install-all: Colab already ships torch/CUDA. Keep it lean.

## 2. Upload data

The loader reads `data/IO-VNBD/Synchronised V abd S datasets` (V-/S- `.csv` pairs).
NOT in git — upload once per runtime:

1. Zip that folder locally (or reuse `Synchronised V abd S datasets.zip`).
2. Upload via the Colab Files pane to `data/IO-VNBD/`, or mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !mkdir -p data/IO-VNBD
   !cp "/content/drive/MyDrive/Synchronised V abd S datasets.zip" data/IO-VNBD/
   !unzip -q data/IO-VNBD/"Synchronised V abd S datasets.zip" -d data/IO-VNBD/
   ```

## 3. Train (subset first — full 72-session run OOMs even locally)

```bash
# smoke test, 1 driver, streaming cache (~minutes)
!python -m python.ml.denoise.train "Driver A"
# then remaining folds, one at a time (each writes results/audit/denoise_fold_<driver>.pt)
!python -m python.ml.denoise.train "Driver B"
!python -m python.ml.denoise.train "Driver D"
!python -m python.ml.denoise.train "Driver E"
```

Memory notes (from local OOM `Killed` on 194k windows): batch 32, val-chunked,
mean/std chunked, `num_workers=0`. On Colab keep the same; add `--max-windows`
if you add that flag later. If the runtime dies, Runtime → Change runtime type →
High-RAM, then re-run a single driver.

## 4. Validate + export

```bash
!python python/eval/validate_denoised.py --max-windows 10
!python python/ml/export_denoise_onnx.py results/audit/denoise_fold_Driver_A.pt results/audit/denoise.onnx
```

## 5. Download weights back (Drive, NOT git push from Colab)

```bash
!cp results/audit/denoise_fold_*.pt "/content/drive/MyDrive/GatiNav-weights/"
!cp results/audit/denoise.onnx "/content/drive/MyDrive/GatiNav-weights/" 2>/dev/null || true
```

Then copy the small deployed files (`denoise.onnx`, `noisenet.pt`-class files
<10 MB) into the repo locally if you want them tracked. Never
`git add` `*_windows.npz` / `*fold_*.pt` / `*.pkl` — `.gitignore` blocks them;
if one slips in, `git rm --cached <file>` before pushing.

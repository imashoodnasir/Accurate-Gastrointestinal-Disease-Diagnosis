# HMA-DER: Hierarchical Multi‑Attention with Dynamic Expert Routing
> Reference PyTorch implementation for **HMA‑DER**, a framework for accurate and explainable GI disease diagnosis with attention fusion (global/ROI/patch), Cognitive Alignment Score (CAS) regularization, and Dynamic Expert Routing (DER).

<p align="center">
  <img alt="HMA-DER Overview" src="https://user-images.githubusercontent.com/000000/placeholder_hma_der.png" width="75%">
</p>

---

## ✨ Key Features
- **Hierarchical Attention**: Stage‑1 Global attention, Stage‑2 ROI cross‑attention, Stage‑3 Patch micro‑attention.
- **Attention Fusion (A\*)**: Probabilistic fusion of the three attention maps with L1 normalization.
- **Explainability‑Aware Training**: CAS alignment loss (KL + sparsity + TV) using expert/lesion masks.
- **Faithfulness Regularization**: Insertion/Deletion‑style objective for attribution consistency.
- **Dynamic Expert Routing (DER)**: Mixture‑of‑experts head with entropy regularization.
- **Comprehensive Metrics**: Dice/IoU (seg), Acc/Macro‑F1/AUROC + ECE/Brier (cls).
- **Fast & Modular**: Clean, minimal PyTorch codebase with Albumentations transforms.

---

## 📦 Project Structure
```
hma_der/
├─ configs/
│  └─ default.yaml
├─ data/
│  ├─ datasets.py          # Seg/Cls dataset definitions
│  └─ transforms.py        # Albumentations pipelines
├─ losses/
│  ├─ cas.py               # CAS alignment + sparsity + TV
│  ├─ faithfulness.py      # insertion/deletion consistency
│  └─ routing.py           # routing entropy regularizer
├─ metrics/
│  ├─ calibration.py       # ECE & Brier
│  ├─ classification.py    # Acc/F1/AUROC
│  └─ segmentation.py      # Dice/IoU
├─ models/
│  ├─ backbone.py          # dilated residual + tiny transformer
│  ├─ attention_stages.py  # global/ROI/patch + fusion
│  ├─ der.py               # gating + experts
│  └─ hma_der.py           # end‑to‑end model wrapper
├─ utils/
│  ├─ eval_utils.py        # validation loops
│  ├─ train_utils.py       # AMP training
│  ├─ viz.py               # heatmap overlay
│  └─ seed.py
├─ train.py
├─ evaluate.py
├─ requirements.txt
└─ README.md
```

---

## 🛠️ Setup
```bash
# 1) create env
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt
```

> Tested with Python 3.10+, PyTorch 2.2+ and CUDA 11.8/12.x.

---

## 📚 Datasets & Directory Layout
Set `data_root` in `configs/default.yaml` and arrange datasets as follows:

```
data_root/
  ks/  # Kvasir-SEG (segmentation)
    images/{train,val,test}/*.png
    masks/{train,val,test}/*.png
  cvc/ # CVC-ClinicDB (segmentation)
    images/{train,val,test}/*.png
    masks/{train,val,test}/*.png
  hk/  # HyperKvasir (classification)
    images/{train,val,test}/{class_name}/*.jpg
  gv/  # GastroVision (classification)
    images/{train,val,test}/{class_name}/*.jpg
```

- **Segmentation** datasets (KS, CVC) need binary masks aligned by filename.
- **Classification** datasets (HK, GV) should have one subfolder per class under each split.

You can change input sizes per dataset in `configs/default.yaml`.

---

## 🚀 Training

### Segmentation (Kvasir‑SEG)
```bash
python hma_der/train.py ks seg 2
```
### Segmentation (CVC‑ClinicDB)
```bash
python hma_der/train.py cvc seg 2
```
### Classification (HyperKvasir)
```bash
python hma_der/train.py hk cls 8
```
### Classification (GastroVision)
```bash
python hma_der/train.py gv cls 5   # adjust class count as needed
```

Training stops early based on validation (`dice` for seg, `macro_f1` for cls). Best weights are written to
```
hma_der_{dataset}_{task}_best.pt
```

---

## ✅ Evaluation & Visualizations
Run evaluation and (optionally) save saliency overlays:
```bash
# segmentation
python hma_der/evaluate.py ks seg 2 outputs/vis_ks

# classification
python hma_der/evaluate.py hk cls 8 outputs/vis_hk
```
This prints dataset metrics and writes image‑level A\* overlays to the output directory.

---

## ⚙️ Configuration
All key hyperparameters live in `configs/default.yaml`:
- Optimizer/schedule (AdamW, LR, weight decay)
- Loss weights: `lambda_cls`, `lambda_align`, `lambda_faith`, `lambda_route`, `tv_strength`
- DER experts count
- Attention fusion weights `beta1/beta2/beta3`
- Per‑dataset image sizes and epoch budgets
- Loader parameters (batch size, workers)

You can override these by editing the YAML or passing custom paths to `train.py`/`evaluate.py`.

---

## 🧠 Method Summary (Paper → Code)
- **Stage‑1 (Global)** → `models/attention_stages.py::GlobalAttention`
- **Stage‑2 (ROI)** → `models/attention_stages.py::ROIAttention`
- **Stage‑3 (Patch)** → `models/attention_stages.py::PatchAttention`
- **Fusion (A\*)** → `models/attention_stages.py::fuse_attention`
- **DER Head** → `models/der.py::DynamicExpertRouting`
- **CAS Loss** → `losses/cas.py::CASLoss`
- **Faithfulness** → `losses/faithfulness.py::FaithfulnessLoss`
- **Routing Entropy** → `losses/routing.py::RoutingEntropyLoss`

---

## 📊 Reported Metrics
- **Segmentation**: Dice, IoU (computed on A\* > 0.5 as a proxy mask), plus optional boundary metrics if you add them.
- **Classification**: Accuracy, Macro‑F1, AUROC (OvR), ECE, Brier.
- **Explainability**: CAS (lower is better in our KL‑based loss), Insertion/Deletion consistency (lower is better loss).

> Tip: You can export per‑image CSVs of metrics by extending `utils/eval_utils.py`.

---

## 🔍 Reproducibility Tips
- Fix `seed` in `configs/default.yaml`.
- Use the same image sizes/splits across runs.
- Log versions: `torch`, `torchvision`, GPU model, and CUDA driver.
- For cross‑center transfer, train on one dataset and evaluate on another without finetuning.

---

## 🧪 Extending the Code
- Swap the backbone with a larger CNN/Transformer while keeping the attention stages intact.
- Replace the ROI proposal with learned proposals (e.g., top‑K via a small conv detector).
- Add boundary‑aware losses for segmentation or class‑balanced CE for imbalanced classes.
- Integrate temperature scaling for post‑hoc calibration.

---

## 📄 License
This project is released under the MIT License. See `LICENSE` (or include it) for details.

---

## ❓ FAQ
**Q: Do I need masks for classification datasets?**  
A: No. CAS alignment is only enforced when masks are available; otherwise, the model trains with the classification loss and faithfulness regularizer.

**Q: Why do Dice/IoU use A\* as a proxy?**  
A: To evaluate attention‑as‑segmentation. If you have a dedicated seg head, you can plug it in and compute metrics on its logits.

**Q: Where do I change image sizes?**  
A: `configs/default.yaml` → `datasets.<name>.img_size`.

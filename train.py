import yaml, torch, torch.optim as optim
from torch.utils.data import DataLoader
from hma_der.utils.seed import set_seed
from hma_der.models.hma_der import HMADER
from hma_der.data.datasets import SegDataset, ClsDataset
from hma_der.data.transforms import get_seg_train_transforms, get_seg_val_transforms, get_cls_train_transforms, get_cls_val_transforms
from hma_der.losses.cas import CASLoss
from hma_der.losses.faithfulness import FaithfulnessLoss
from hma_der.losses.routing import RoutingEntropyLoss
from hma_der.utils.train_utils import train_one_epoch
from hma_der.utils.eval_utils import eval_epoch

def main(cfg_path="configs/default.yaml", dataset="ks", task="seg", num_classes=2):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["seed"])
    device = cfg["device"]

    if task=="seg":
        train_tf = get_seg_train_transforms(cfg["datasets"][dataset]["img_size"])
        val_tf   = get_seg_val_transforms(cfg["datasets"][dataset]["img_size"])
        train_ds = SegDataset(root=f'{cfg["data_root"]}/{dataset}', split="train", transforms=train_tf)
        val_ds   = SegDataset(root=f'{cfg["data_root"]}/{dataset}', split="val", transforms=val_tf)
    else:
        train_tf = get_cls_train_transforms(cfg["datasets"][dataset]["img_size"])
        val_tf   = get_cls_val_transforms(cfg["datasets"][dataset]["img_size"])
        # classes inferred from folder names under images/train
        train_ds = ClsDataset(root=f'{cfg["data_root"]}/{dataset}', split="train", transforms=train_tf)
        val_ds   = ClsDataset(root=f'{cfg["data_root"]}/{dataset}', split="val", transforms=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["loader"]["batch_size"], shuffle=True, num_workers=cfg["loader"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["loader"]["batch_size"], shuffle=False, num_workers=cfg["loader"]["num_workers"], pin_memory=True)

    model = HMADER(num_classes=num_classes, dk=128, experts=cfg["experts"]).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

    losses = {
        "cas": CASLoss(tv_strength=cfg["loss_weights"]["tv_strength"]),
        "faith": FaithfulnessLoss(keep_ratio=0.2),
        "route": RoutingEntropyLoss()
    }

    max_epochs = cfg["epochs"][dataset]
    best_metric, patience, bad = -1, (cfg["early_stopping"]["patience_seg"] if task=="seg" else cfg["early_stopping"]["patience_cls"]), 0

    for epoch in range(1, max_epochs+1):
        tr = train_one_epoch(model, train_loader, opt, losses, cfg, task=task)
        ev = eval_epoch(model, val_loader, task=task)
        metric_name = "dice" if task=="seg" else "macro_f1"
        metric_val = ev[metric_name]
        print(f"[{epoch:03d}] loss={tr['loss']:.4f} | {metric_name}={metric_val:.4f}")

        if metric_val > best_metric:
            best_metric, bad = metric_val, 0
            torch.save(model.state_dict(), f"hma_der_{dataset}_{task}_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    # Example runs:
    # python -m hma_der.train -- ks seg 2
    import sys
    dataset = sys.argv[1] if len(sys.argv)>1 else "ks"   # ks|cvc|hk|gv
    task    = sys.argv[2] if len(sys.argv)>2 else "seg"  # seg|cls
    ncls    = int(sys.argv[3]) if len(sys.argv)>3 else (2 if task=="seg" else 8)
    main(dataset=dataset, task=task, num_classes=ncls)

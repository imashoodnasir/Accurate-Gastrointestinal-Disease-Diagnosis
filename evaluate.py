import yaml, torch, cv2, os
from torch.utils.data import DataLoader
from hma_der.models.hma_der import HMADER
from hma_der.data.datasets import SegDataset, ClsDataset
from hma_der.data.transforms import get_seg_val_transforms, get_cls_val_transforms
from hma_der.utils.eval_utils import eval_epoch
from hma_der.utils.viz import overlay_heatmap

def main(cfg_path="configs/default.yaml", dataset="ks", task="seg", num_classes=2, ckpt=None, save_vis_dir=None):
    cfg = yaml.safe_load(open(cfg_path))
    device = cfg["device"]

    if task=="seg":
        val_tf = get_seg_val_transforms(cfg["datasets"][dataset]["img_size"])
        ds = SegDataset(root=f'{cfg["data_root"]}/{dataset}', split="val", transforms=val_tf)
    else:
        val_tf = get_cls_val_transforms(cfg["datasets"][dataset]["img_size"])
        ds = ClsDataset(root=f'{cfg["data_root"]}/{dataset}', split="val", transforms=val_tf)

    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model = HMADER(num_classes=num_classes, dk=128, experts=cfg["experts"]).to(device)
    ckpt = ckpt or f"hma_der_{dataset}_{task}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    metrics = eval_epoch(model, loader, task=task)
    print(metrics)

    if save_vis_dir:
        os.makedirs(save_vis_dir, exist_ok=True)
        with torch.no_grad():
            for b in loader:
                x = b["image"].to(device)
                out = model(x)
                vis = overlay_heatmap(x[0], out["A_star"][0:1])
                cv2.imwrite(os.path.join(save_vis_dir, b["id"][0].replace(".png",".jpg")), vis)

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv)>1 else "ks"
    task    = sys.argv[2] if len(sys.argv)>2 else "seg"
    ncls    = int(sys.argv[3]) if len(sys.argv)>3 else (2 if task=="seg" else 8)
    visdir  = sys.argv[4] if len(sys.argv)>4 else None
    main(dataset=dataset, task=task, num_classes=ncls, save_vis_dir=visdir)

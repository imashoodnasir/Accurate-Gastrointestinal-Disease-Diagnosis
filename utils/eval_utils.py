import torch
from ..metrics.segmentation import dice_score, iou_score
from ..metrics.classification import classification_metrics

@torch.no_grad()
def eval_epoch(model, loader, task="seg"):
    model.eval()
    if task=="seg":
        dices, ious = [], []
        for b in loader:
            x = b["image"].cuda()
            y = b["mask"].cuda()
            out = model(x)
            pred = (out["A_star"]>0.5).float()
            dices.append(dice_score(pred, y).item())
            ious.append(iou_score(pred, y).item())
        return {"dice": sum(dices)/len(dices), "iou": sum(ious)/len(ious)}
    else:
        ys, probs = [], []
        for b in loader:
            x = b["image"].cuda()
            out = model(x)
            p = out["yhat"].softmax(1).detach().cpu()
            probs.append(p)
            ys.extend(b["label"])
        import torch as T
        probs = T.cat(probs, dim=0)
        return classification_metrics(ys, probs)

import torch, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, loss_modules, cfg, task="seg"):
    model.train()
    scaler = GradScaler()
    metrics = {"loss": 0.0}
    for batch in loader:
        x = batch["image"].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out = model(x)
            loss = 0.0
            # classification term
            if task=="cls":
                y = batch["label"].cuda()
                loss_cls = F.cross_entropy(out["yhat"], y)
            else:
                # for seg supervision, derive a mask-prob head from A* (proxy)
                y = batch["mask"].cuda()
                # predict mask from attention (simple proxy for demo)
                pred_mask = (out["A_star"] > 0.5).float()
                # encourage attention to match mask via BCE (aux)
                loss_cls = F.binary_cross_entropy_with_logits(out["A_star"], y)

            # CAS alignment
            loss_align = loss_modules["cas"](out["A_star"], y.unsqueeze(1) if task=="cls" else y)

            # faithfulness (stop-grad inside)
            if task=="cls":
                loss_faith = loss_modules["faith"](model, x, y, out["A_star"])
            else:
                # for seg, approximate with foreground as positive class 1
                y_pos = torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
                y_pos[:] = 1 if out["yhat"].shape[1]>1 else 0
                loss_faith = loss_modules["faith"](model, x, y_pos, out["A_star"])

            # routing entropy
            loss_route = loss_modules["route"](out["pi"])

            # sparsity/TV already in CAS (implemented inside)

            lw = cfg["loss_weights"]
            loss = (lw["lambda_cls"]*loss_cls +
                    lw["lambda_align"]*loss_align +
                    lw["lambda_faith"]*loss_faith +
                    lw["lambda_route"]*loss_route)

        scaler.scale(loss).from_float(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        metrics["loss"] += loss.item()*x.size(0)
    metrics["loss"] /= len(loader.dataset)
    return metrics

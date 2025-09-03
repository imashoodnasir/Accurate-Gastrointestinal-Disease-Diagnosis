import cv2, numpy as np, torch

def overlay_heatmap(img_tensor, A_star):
    # img_tensor: (3,H,W) in normalized space -> bring back to uint8
    img = img_tensor.detach().cpu().numpy().transpose(1,2,0)
    img = (img* np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))*255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    H,W,_ = img.shape
    att = A_star.detach().cpu().numpy()[0,0]
    att = cv2.resize(att, (W,H))
    att = (255*(att - att.min())/(att.max()-att.min()+1e-6)).astype(np.uint8)
    heat = cv2.applyColorMap(att, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    return out[..., ::-1]  # RGB

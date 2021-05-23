import numpy as np
import torch

def unnormalize(img_test):
    img_mean = np.array([104, 117, 128]).reshape(1, 1, 3)
    unnormalized = (img_test.cpu().numpy().transpose(1, 2, 0) + img_mean)[:, :, ::-1].astype(np.uint8)
    return unnormalized

class must_transform():
    def __call__(self, img):
        img_mean = np.array([104, 117, 128]).reshape(1, 1, 3)
        bgr_img = img[:, :, ::-1]
        bgr_img = np.ascontiguousarray(bgr_img, dtype= np.float32)
        norm_img = bgr_img - img_mean
        tr_norm_img = norm_img.transpose(2, 0, 1)
        return torch.from_numpy(tr_norm_img).float()


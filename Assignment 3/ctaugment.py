import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class CTAugment:
    def __init__(self, n_bins=17, alpha=1e-2, init_w=1.0):
        
        self._AUGMENTATION_SPACE = {
            "AutoContrast": (lambda n, h, w: None, False),
            "Brightness": (lambda n, h, w: np.linspace(0.0, 1.0, n), True),
            "Color": (lambda n, h, w: np.linspace(0.0, 1.0, n), True),
            "Contrast": (lambda n, h, w: np.linspace(0.0, 1.0, n), True),
            "Equalize": (lambda n, h, w: np.linspace(0.0, 1.0, n), False),
            "Identity": (lambda n, h, w: None, False),
            "Invert": (lambda n, h, w: np.linspace(0.0, 1.0, n), False),
            "Posterize": (lambda n, h, w: np.round(np.linspace(1, 8, n)).astype(int), False),
            "Rescale": (lambda n, h, w: np.linspace(0.5, 1.0, n), False),
            "Rotate": (lambda n, h, w: np.linspace(-45.0, 45.0, n), True),
            "Sharpness": (lambda n, h, w: np.linspace(0.0, 1.0, n), True),
            "ShearX": (lambda n, h, w: np.linspace(-0.3, 0.3, n), True),
            "ShearY": (lambda n, h, w: np.linspace(-0.3, 0.3, n), True),
            "Solarize": (lambda n, h, w: np.linspace(0.0, 256.0, n), False),
            "Smooth": (lambda n, h, w: np.linspace(0.0, 1.0, n), True),
            "TranslateX": (lambda n, h, w: np.linspace(-0.3 * w, 0.3 * w, n), True),
            "TranslateY": (lambda n, h, w: np.linspace(-0.3 * h, 0.3 * h, n), True),
        }

        self._op_names = list(self._AUGMENTATION_SPACE.keys())
        self._n_ops = len(self._op_names)
        self.n_bins = n_bins
        self.alpha = alpha
        self.weights = np.ones((self._n_ops, n_bins), dtype=np.float32) * init_w

    def __call__(self, img):
        return self.apply(img)[0]

    def _sample_bin(self, op_idx):
        p = self.weights[op_idx] / (self.weights[op_idx].sum() + 1e-12)
        return np.random.choice(self.n_bins, p=p)

    def _bin_to_mag(self, op_idx, bin_idx, H, W):
        name = self._op_names[op_idx]
        mag_fn, _ = self._AUGMENTATION_SPACE[name]

        # mag_fn pode ser None (não é função) ou uma função que retorna None/scalar/array
        if mag_fn is None:
            return None

        vals = mag_fn(self.n_bins, H, W)
        if vals is None:
            return None

        arr = np.asarray(vals)

        # se arr for escalar (0-d), devolve o valor escalar
        if arr.ndim == 0:
            return float(arr)
        
        n = arr.shape[0]
        if n == 0:
            return None

        bi = int(bin_idx)
        bi = max(0, min(bi, n - 1))
        
        if bi < n - 1:
            low = float(arr[bi])
            high = float(arr[bi + 1])
            return float(low + np.random.rand() * (high - low))
        else:
            return float(arr[bi])


    def _apply_op(self, img, name, mag):
        C, H, W = img.shape
        device = img.device

        if name == "AutoContrast":
            return TF.autocontrast(img)

        if name == "Brightness":
            return TF.adjust_brightness(img, float(mag))

        if name == "Color":
            return TF.adjust_saturation(img, float(mag))

        if name == "Contrast":
            return TF.adjust_contrast(img, float(mag))
        
        if name == "Equalize":
            blend = 1.0 if mag is None else float(mag)
            
            if torch.max(img) > 1:
                img_u8 = img.clamp(0, 255).to(torch.uint8)
            else:
                img_u8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8)

            eq_u8 = torch.empty_like(img_u8, dtype=torch.uint8, device=device)
            num_pixels = H * W

            for c in range(C):
                ch = img_u8[c].view(-1).to(device).long() 
                hist = torch.bincount(ch, minlength=256).to(torch.float32)
                cdf = hist.cumsum(0)

                # se não houver pixels, mantém canal original
                if (cdf > 0).sum() == 0:
                    eq_u8[c] = img_u8[c]
                    continue

                cdf_min = cdf[cdf > 0][0]
                denom = float(num_pixels) - float(cdf_min)
                if denom <= 0.0:
                    lut = torch.arange(256, dtype=torch.uint8, device=device)
                else:
                    lut = (((cdf - cdf_min) / denom) * 255.0).round().clamp(0, 255).to(torch.uint8).to(device)

                # mapeia usando LUT
                eq_u8[c] = lut[img_u8[c].long()]

            # volta para float 0-1 e blend com a imagem original
            eq_f = eq_u8.to(torch.float32) / 255.0
            return eq_f * blend

        if name == "Identity":
            return img
        
        if name == "Invert":
            return TF.invert(img)

        if name == "Posterize":
            bits = int(round(mag)) if mag is not None else 4
            bits = max(1, min(bits, 8))
            img_u8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8)
            out_u8 = TF.posterize(img_u8, bits)
            out = out_u8.to(torch.float32) / 255.0
            return out

        if name == "Rescale":
            scale = float(mag) if mag is not None else 1.0
            new_h = max(1, int(round(H * scale))); new_w = max(1, int(round(W * scale)))
            out = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
            if scale < 1.0:
                pad_h = H - new_h; pad_w = W - new_w
                ph0 = pad_h // 2; pw0 = pad_w // 2
                out = F.pad(out, (pw0, pad_w - pw0, ph0, pad_h - ph0), value=0.5)
                return out
            else:
                top = max(0, (new_h - H) // 2); left = max(0, (new_w - W) // 2)
                return out[:, top:top + H, left:left + W]

        if name == "Rotate":
            angle = float(mag)
            return TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=128)

        if name == "Sharpness":
            return TF.adjust_sharpness(img, float(mag))

        if name == "ShearX":
            shear_deg = float(mag) * (180.0 / math.pi)
            return TF.affine(img, angle=0.0, translate=(0, 0), scale=1.0, shear=(shear_deg, 0.0), interpolation=TF.InterpolationMode.BILINEAR, fill=128)

        if name == "ShearY":
            shear_deg = float(mag) * (180.0 / math.pi)
            return TF.affine(img, angle=0.0, translate=(0, 0), scale=1.0, shear=(0.0, shear_deg), interpolation=TF.InterpolationMode.BILINEAR, fill=128)

        if name == "Solarize":
            thr = int(mag) if mag is not None else 128
            thr = max(0, min(thr, 255))

            img_u8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8)
            out_u8 = TF.solarize(img_u8, thr)
            out = out_u8.to(torch.float32) / 255.0
            return out
        
        if name == "Smooth":
            radius = float(mag) if mag is not None else 0.0
            if radius <= 0.0:
                return img
            k = max(1, int(round(radius * 2)) | 1)
            return TF.gaussian_blur(img, kernel_size=[k, k], sigma=(radius, radius))

        if name == "TranslateX":
            tx = round(float(mag))
            return TF.affine(img, angle=0.0, translate=(tx, 0), scale=1.0, shear=0.0, interpolation=TF.InterpolationMode.BILINEAR, fill=128)

        if name == "TranslateY":
            ty = round(float(mag))
            return TF.affine(img, angle=0.0, translate=(0, ty), scale=1.0, shear=0.0, interpolation=TF.InterpolationMode.BILINEAR, fill=128)

        return img

    def apply(self, img):
        orig_device = img.device
        orig_dtype = img.dtype

        t = img.detach().cpu()
        
        scale_restore = 1.0
        if orig_dtype == torch.uint8:
            t = t.float() / 255.0
            scale_restore = 255.0
        elif t.dtype in (torch.float32, torch.float64):
            if t.numel() > 0 and float(t.max()) > 1.0:
                t = t.float() / 255.0
                scale_restore = 255.0
            else:
                t = t.float()
        else:
            t = t.float()

        # garante C,H,W
        if t.ndim == 2:
            t = t.unsqueeze(0)

        C, H, W = t.shape
        op_idxs = np.random.choice(self._n_ops, size=2, replace=False)
        out = t
        choices = []
        for oi in op_idxs:
            bi = self._sample_bin(oi)
            mag = self._bin_to_mag(oi, bi, H, W)
            name = self._op_names[oi]
            out = self._apply_op(out, name, mag)
            choices.append((oi, bi, mag))

        # restaura escala/dtype e device
        if scale_restore != 1.0:
            out = out * scale_restore
        if orig_dtype == torch.uint8:
            out = out.round().clamp(0, 255).to(torch.uint8)
        out = out.to(orig_device)
        return out, choices

    def update(self, choices, reward):
        baseline = 0.5
        for oi, bi, _ in choices:
            self.weights[oi, bi] *= math.exp(self.alpha * (float(reward) - baseline))
            self.weights[oi] = np.clip(self.weights[oi], 1e-6, 1e6)
            s = self.weights[oi].sum()
            if s > 0:
                self.weights[oi] /= s

    def get_weights_dict(self):
        return {self._op_names[i]: self.weights[i].copy() for i in range(self._n_ops)}


if __name__ == "__main__":
    augment = CTAugment()

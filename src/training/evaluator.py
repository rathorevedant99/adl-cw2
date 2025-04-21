import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import random
import logging
from PIL import Image, ImageDraw
import torchvision.transforms as T

class Evaluator:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.method = config['model'].get('method','WS').upper()

        device_name = config.get('device', 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        if device_name == 'mps' and not torch.backends.mps.is_available():
            logging.warning("MPS device not available, falling back to CPU")
            device_name = 'cpu'
        if device_name == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA device not available, falling back to CPU")
            device_name = 'cpu'
        self.device = torch.device(device_name)
        logging.info(f"Using device: {self.device}")

        self.model = self.model.to(self.device)
        self.eval_loader = DataLoader(
            dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['evaluation']['num_workers'],
            pin_memory=config['training'].get('pin_memory', False)
        )

        self.viz_dir = Path(config.get('viz_dir', 'experiments/visualizations'))
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self):
        self.model.eval()
        metrics = {'accuracy': [], 'mean_iou': [], 'pixel_accuracy': []}
        logging.info('Starting model evaluation...')

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_loader):
                # unpack paired data in WS mode, or single ds in FS
                if self.method == 'WS':
                    weak_batch, fs_batch = batch
                    images     = weak_batch['image'].to(self.device)
                    cls_labels = weak_batch['mask'].to(self.device)   # scalar class idx
                    seg_labels = fs_batch  ['mask'].to(self.device)   # H×W mask
                else:
                    fs_batch   = batch
                    images     = fs_batch['image'].to(self.device)
                    seg_labels = fs_batch['mask'].to(self.device)
                    cls_labels = None

                outputs = self.model(images)

                if self.method == 'WS':
                    # pick out each sample's CAM channel
                    cams = outputs['segmentation_maps']           # [B, C, H, W]
                    idx  = torch.arange(cams.size(0), device=self.device)
                    pet_cams = cams[idx, cls_labels]             # [B, H, W]

                    # threshold to binary mask
                    prob_maps = torch.sigmoid(pet_cams)
                    seg_preds = (prob_maps > 0.5).long()

                    # compute binary IoU + pixel accuracy
                    mean_iou, pixel_acc = self._calc_binary_metrics(seg_preds, seg_labels)

                    # classification accuracy
                    logits    = outputs['logits']
                    cls_preds = logits.argmax(dim=1)
                    acc       = (cls_preds == cls_labels).float().mean().item()
                    metrics['accuracy'].append(acc)
                else:
                    # fully‑supervised: simple argmax over C logits
                    seg_maps   = outputs['segmentation_maps']
                    seg_preds  = seg_maps.argmax(dim=1)          # [B, H, W]
                    mean_iou, pixel_acc = self._calc_seg_metrics(seg_preds, seg_labels)

                metrics['mean_iou'].append(mean_iou)
                metrics['pixel_accuracy'].append(pixel_acc)
                if batch_idx % 10 == 0:
                    logging.info(f'Evaluated batch {batch_idx}/{len(self.eval_loader)}')

        # aggregate metrics
        final_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        logging.info("\nEvaluation Results:")
        if self.method == 'WS':
            logging.info(f"Classification Accuracy: {final_metrics['accuracy']:.4f}")
        logging.info(f"Mean IoU: {final_metrics['mean_iou']:.4f}")
        logging.info(f"Pixel Accuracy: {final_metrics['pixel_accuracy']:.4f}")

        # visualize CAMs if WS
        if self.method == 'WS':
            self._visualize_cams()
        # visualize segmentation examples
        self._visualize_examples()

        return final_metrics

    def _calc_seg_metrics(self, preds, labels):
        # Compute mean IoU and pixel accuracy for a batch
        batch_size = preds.size(0)
        total_iou, total_acc = 0.0, 0.0
        num_classes = self.config['model']['num_classes']
        for i in range(batch_size):
            pred_mask = preds[i]
            true_mask = labels[i]
            # pixel accuracy
            total_acc += (pred_mask == true_mask).float().mean().item()
            # iou per class
            ious = []
            for c in range(num_classes):
                p = (pred_mask == c)
                t = (true_mask == c)
                union = (p | t).sum().float()
                if union > 0:
                    intersection = (p & t).sum().float()
                    ious.append((intersection / (union + 1e-8)).item())
            if ious:
                total_iou += np.mean(ious)
        return total_iou / batch_size, total_acc / batch_size
    
    def _calc_binary_metrics(self, preds, true_masks):
        """
        preds, true_masks: LongTensors of shape [B, H, W] with values 0 or 1.
        Returns (mean_iou, pixel_acc).
        """
        batch_size = preds.size(0)
        total_iou, total_acc = 0.0, 0.0

        for i in range(batch_size):
            p = preds[i].bool()
            t = true_masks[i].bool()

            inter = (p & t).sum().float()
            uni   = (p | t).sum().float() + 1e-8
            iou   = (inter / uni).item()
            acc   = (p == t).float().mean().item()

            total_iou += iou
            total_acc += acc

        return total_iou / batch_size, total_acc / batch_size    

    def _create_heatmap(self, cam, size):
        cam = np.clip(cam, 0, None)
        cam = cam / (cam.max() + 1e-8)
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS)
        cam_rgb = Image.new('RGB', size)
        for x in range(size[0]):
            for y in range(size[1]):
                v = cam_pil.getpixel((x,y)) / 255.0
                r = int(255 * max(0, min(1, 1.5 - abs(4*v-3))))
                g = int(255 * max(0, min(1, 1.5 - abs(4*v-2))))
                b = int(255 * max(0, min(1, 1.5 - abs(4*v-1))))
                cam_rgb.putpixel((x,y),(r,g,b))
        return cam_rgb

    def _visualize_cams(self, num_samples=5):
        logging.info(f"Generating CAM visualizations for {num_samples} samples...")
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        for idx in indices:
            sample = self.dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            true_label = sample['mask']
            name = sample['image_name']
            with torch.no_grad():
                out = self.model(image)
                logits = out['logits']
                seg_maps = out['segmentation_maps']
            pred_label = logits.argmax(dim=1).item()
            true_cam = seg_maps[0, true_label].cpu().numpy()
            pred_cam = seg_maps[0, pred_label].cpu().numpy()
            orig = image[0].cpu().numpy().transpose(1,2,0)
            orig = (orig * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])) * 255
            orig = orig.astype(np.uint8)
            orig_img = Image.fromarray(orig)
            # build combined
            w,h=orig_img.size
            heat_true = self._create_heatmap(true_cam,(w,h))
            heat_pred = self._create_heatmap(pred_cam,(w,h))
            overlay_true = Image.blend(orig_img, heat_true, alpha=0.5)
            overlay_pred = Image.blend(orig_img, heat_pred, alpha=0.5)
            canvas = Image.new('RGB',(w*4,h))
            canvas.paste(orig_img,(0,0))
            canvas.paste(heat_true,(w,0))
            canvas.paste(overlay_true,(w*2,0))
            canvas.paste(heat_pred,(w*3,0))
            path=self.viz_dir/f"cam_{name}.png"
            canvas.save(path)
            logging.info(f"Saved CAM {path}")

    def _visualize_examples(self, num_samples=3):
        logging.info(f"Saving {num_samples} example segmentations...")
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        for idx in indices:
            sample = self.dataset[idx]
            img_t = sample['image']
            true_mask = sample['mask']
            name = sample['image_name']
            with torch.no_grad():
                out = self.model(img_t.unsqueeze(0).to(self.device))
                seg_map = out['segmentation_maps'][0].argmax(dim=0).cpu().numpy()
            # unnormalize
            img_np = img_t.cpu().numpy().transpose(1,2,0)
            img_np = img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
            img_np = np.clip(img_np,0,1)
            orig_img = Image.fromarray((img_np*255).astype(np.uint8))
            gt_mask_img = Image.fromarray((true_mask.cpu().numpy()*255).astype(np.uint8)).convert('L')
            pred_mask_img = Image.fromarray((seg_map*255).astype(np.uint8)).convert('L')
            # composite grid
            w,h = orig_img.size
            canvas = Image.new('RGB',(w*3,h))
            canvas.paste(orig_img,(0,0))
            canvas.paste(gt_mask_img.convert('RGB'),(w,0))
            canvas.paste(pred_mask_img.convert('RGB'),(w*2,0))
            path=self.viz_dir/f"example_{name}.png"
            canvas.save(path)
            logging.info(f"Saved example {path}")

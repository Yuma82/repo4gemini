import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import List

# Add paths to import DETR modules
sys.path.append('../../detr-main')
sys.path.append('../../detr-main/models')
sys.path.append('../../detr-main/util')

from models.detr import DETR, PostProcess
from models.backbone import build_backbone
from models.transformer import build_transformer
from util.misc import nested_tensor_from_tensor_list
import util.box_ops as box_ops

class DETRODAMNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Build DETR model components
        self.backbone = self._build_backbone()
        self.transformer = self._build_transformer()
        
        # Build DETR model
        self.detr = DETR(
            backbone=self.backbone,
            transformer=self.transformer,
            num_classes=config.num_classes,
            num_queries=config.num_queries,
            aux_loss=False
        )
        
        # Post-processor for converting outputs
        self.postprocessor = PostProcess()
        
    def _build_backbone(self):
        """Build DETR backbone"""
        # Create a mock args object for backbone building
        class BackboneArgs:
            backbone = self.config.backbone
            dilation = False
            position_embedding = self.config.position_embedding
            
        args = BackboneArgs()
        return build_backbone(args)
        
    def _build_transformer(self):
        """Build DETR transformer"""
        # Create a mock args object for transformer building
        class TransformerArgs:
            hidden_dim = self.config.hidden_dim
            dropout = self.config.dropout
            nheads = self.config.nheads
            dim_feedforward = self.config.dim_feedforward
            enc_layers = self.config.enc_layers
            dec_layers = self.config.dec_layers
            pre_norm = self.config.pre_norm
            
        args = TransformerArgs()
        return build_transformer(args)
    
    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained DETR weights"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Load weights, ignoring mismatched keys
            missing_keys, unexpected_keys = self.detr.load_state_dict(state_dict, strict=False)
            print(f"Loaded DETR weights from {checkpoint_path}")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found, using random initialization")
    
    def forward(self, image, im_info, gt_boxes=None):
        """Forward pass compatible with ODAM framework"""
        config = self.config
        
        # Preprocess image
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension if needed
        
        # Normalize image (DETR expects ImageNet normalization)
        # Convert from BGR to RGB and normalize
        if config.to_bgr255:
            image = image[:, [2, 1, 0], :, :]  # BGR to RGB
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        image = (image - mean) / std
        
        # Convert to NestedTensor for DETR
        if not isinstance(image, list):
            image = nested_tensor_from_tensor_list([image[0]])
        
        # Forward through DETR
        outputs = self.detr(image)
        
        if self.training:
            # Training mode - return loss (not implemented for inference-only version)
            raise NotImplementedError("Training mode not implemented for inference-only ODAM DETR")
        else:
            # Inference mode
            return self._process_inference_outputs(outputs, im_info)
    
    def _process_inference_outputs(self, outputs, im_info):
        """Process DETR outputs for ODAM compatibility"""
        pred_logits = outputs['pred_logits']  # [batch_size, num_queries, num_classes+1]
        pred_boxes = outputs['pred_boxes']    # [batch_size, num_queries, 4] in cxcywh format
        
        batch_size = pred_logits.shape[0]
        
        # Convert to image coordinates
        h, w = im_info[0][0].item(), im_info[0][1].item()
        target_sizes = torch.tensor([[h, w]]).to(pred_logits.device)
        
        # Use PostProcess to get final detections
        results = self.postprocessor(outputs, target_sizes)
        
        # Convert to ODAM format: [x1, y1, x2, y2, score, label, cam_features..., cam_h, cam_w]
        detections = []
        attribution_maps = []
        pred_levels = []
        
        for result in results:
            boxes = result['boxes']      # [N, 4] in xyxy format
            scores = result['scores']    # [N]
            labels = result['labels']    # [N]
            
            # Filter by score threshold
            valid_mask = scores > self.config.pred_cls_threshold
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            labels = labels[valid_mask]
            
            if len(boxes) == 0:
                detections.append(torch.empty(0, 8).to(pred_logits.device))
                attribution_maps.append([])
                pred_levels.append(torch.empty(0).to(pred_logits.device))
                continue
            
            # Generate attribution maps for each detection
            cam_features = []
            cam_h, cam_w = 64, 64  # Standard size for attribution maps
            
            for i in range(len(boxes)):
                # Create simple heatmap based on bounding box location
                # This is a simplified approach - can be enhanced with gradient-based methods
                cam = self.create_box_heatmap(boxes[i], (cam_h, cam_w), pred_logits.device)
                cam_features.extend(cam.flatten().tolist())
            
            # Construct detection tensor in ODAM format
            detection_tensor = torch.cat([
                boxes,                                    # [N, 4] - x1, y1, x2, y2
                scores.unsqueeze(1),                     # [N, 1] - score
                labels.unsqueeze(1).float(),             # [N, 1] - label
                torch.tensor(cam_features).view(len(boxes), -1).to(pred_logits.device),  # [N, cam_h*cam_w] - attribution map
                torch.full((len(boxes), 1), cam_h, device=pred_logits.device),  # [N, 1] - cam_h
                torch.full((len(boxes), 1), cam_w, device=pred_logits.device),  # [N, 1] - cam_w
            ], dim=1)
            
            detections.append(detection_tensor)
            attribution_maps.append(cam_features)
            pred_levels.append(torch.zeros(len(boxes)).to(pred_logits.device))  # Dummy level info
        
        # Return in the format expected by ODAM evaluation scripts
        if len(detections) == 1:
            return detections[0], {'detr': outputs}, pred_levels[0]
        else:
            return detections, {'detr': outputs}, pred_levels
    
    def create_box_heatmap(self, box, cam_size, device):
        """Create a simple heatmap based on bounding box location"""
        cam_h, cam_w = cam_size
        x1, y1, x2, y2 = box
        
        # Create coordinate grids
        y_coords = torch.linspace(0, 1, cam_h, device=device)
        x_coords = torch.linspace(0, 1, cam_w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalize box coordinates to [0, 1]
        # Assuming box is already in absolute coordinates
        # For a more accurate implementation, we'd need the original image size
        box_center_x = (x1 + x2) / 2 / 800  # Approximate normalization
        box_center_y = (y1 + y2) / 2 / 600
        box_width = (x2 - x1) / 800
        box_height = (y2 - y1) / 600
        
        # Create Gaussian-like heatmap centered on the box
        sigma_x = max(box_width / 4, 0.05)  # Minimum sigma
        sigma_y = max(box_height / 4, 0.05)
        
        gaussian = torch.exp(
            -((xx - box_center_x) ** 2 / (2 * sigma_x ** 2) +
              (yy - box_center_y) ** 2 / (2 * sigma_y ** 2))
        )
        
        # Normalize to [0, 1]
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min() + 1e-8)
        
        return gaussian

def get_network(args):
    """
    Entry point function compatible with ODAM framework
    This function should be called by the ODAM evaluation scripts
    """
    from config_coco import config
    
    # Create network
    network = DETRODAMNetwork(config)
    
    # Load pretrained weights if specified
    if hasattr(args, 'resume_weights') and args.resume_weights is not None:
        # Construct checkpoint path based on resume_weights number
        checkpoint_path = f"model_dump/model-{args.resume_weights}.pth"
        if not os.path.exists(checkpoint_path):
            # Try alternative paths
            alt_paths = [
                f"checkpoint{args.resume_weights:04d}.pth",
                f"detr-r50-{args.resume_weights}.pth",
                f"model_{args.resume_weights}.pth"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    break
        
        network.load_pretrained_weights(checkpoint_path)
    
    return network

# For compatibility with the existing ODAM framework
Network = DETRODAMNetwork
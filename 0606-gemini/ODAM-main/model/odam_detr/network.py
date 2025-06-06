import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from torchvision import transforms

# Add paths to import DETR modules
sys.path.append('../../detr-main')

from models.backbone import build_backbone
from models.transformer import build_transformer
from models.detr import DETR
from util.misc import nested_tensor_from_tensor_list
import util.box_ops as box_ops



class DETRODAMNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Mock args to build DETR components
        class MockArgs:
            def __init__(self, config):
                self.hidden_dim = config.hidden_dim
                self.dropout = config.dropout
                self.nheads = config.nheads
                self.dim_feedforward = config.dim_feedforward
                self.enc_layers = config.enc_layers
                self.dec_layers = config.dec_layers
                self.pre_norm = config.pre_norm
                self.backbone = config.backbone
                self.dilation = config.dilation
                self.position_embedding = config.position_embedding
                self.lr_backbone = 0 # <<< この行を追加してください
                self.masks = False # <<< この行を追加してください

        args = MockArgs(config)
        
        
        # Build DETR model from components to allow feature map access
        self.backbone = build_backbone(args)
        self.transformer = build_transformer(args)
        
        self.detr = DETR(
            backbone=self.backbone,
            transformer=self.transformer,
            num_classes=config.num_classes,
            num_queries=config.num_queries,
            aux_loss=False
        )

    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained DETR weights"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model']
            missing_keys, unexpected_keys = self.detr.load_state_dict(state_dict, strict=False)
            print(f"Loaded DETR weights from {checkpoint_path}")
            if missing_keys: print(f"Missing keys: {missing_keys}")
            if unexpected_keys: print(f"Unexpected keys: {unexpected_keys}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found.")

    def forward(self, image, im_info, gt_boxes=None):
        """Forward pass compatible with ODAM framework"""
        # DETR expects RGB images normalized by ImageNet mean/std
        # The dataloader already provides images in this format.
        samples = nested_tensor_from_tensor_list(image)
        
        # Manually perform DETR forward pass to get intermediate features
        features, pos = self.detr.backbone(samples)
        src, mask = features[-1].decompose()
        
        projected_src = self.detr.input_proj(src)
        hs = self.detr.transformer(projected_src, mask, self.detr.query_embed.weight, pos[-1])[0]
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        
        outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        # In inference mode, process outputs to include attribution maps
        return self._process_inference_outputs(outputs, im_info, projected_src, pos[-1], mask)

    def _process_inference_outputs(self, outputs, im_info, feature_map, pos_embed, mask):
        """Process DETR outputs to include ODAM and match expected format."""
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # Filter detections by threshold
        keep = scores > self.config.pred_cls_threshold
        
        # Get boxes, scores, and labels for kept detections
        final_boxes = box_ops.box_cxcywh_to_xyxy(out_bbox[keep])
        img_h, img_w = im_info[0,0:2]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
        final_boxes = final_boxes * scale_fct
        
        final_scores = scores[keep]
        final_labels = labels[keep]
        
        if final_scores.shape[0] == 0:
            return torch.empty(0, 8).to(out_logits.device), {}, torch.empty(0).to(out_logits.device)

        # --- ODAM Attribution Map Calculation ---
        cams = []
        # Get query indices for the kept detections
        query_indices = keep.nonzero(as_tuple=False)[:, 1]
        
        # Detach the feature map to make it a leaf variable for grad-cam
        grad_cam_features = feature_map.detach()

        for i in range(final_scores.shape[0]):
            query_idx = query_indices[i]
            label_idx = final_labels[i]

            # Enable grad on the detached feature map
            grad_cam_features.requires_grad_(True)
            
            # Re-compute the transformer output with the detached feature map
            hs_for_grad = self.detr.transformer(grad_cam_features,
                                                mask,
                                                self.detr.query_embed.weight,
                                                pos_embed)[0]
            
            # Re-compute logits using the new transformer output and select the last layer
            outputs_class_for_grad = self.detr.class_embed(hs_for_grad)
            score_for_grad = outputs_class_for_grad[-1, 0, query_idx, label_idx]
            
            # Compute gradient
            grad = torch.autograd.grad(score_for_grad, grad_cam_features, allow_unused=True)[0]
            
            # Compute attribution map (Grad-CAM)
            cam = F.relu_((grad * grad_cam_features).sum(1)).squeeze(0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # Normalize
            cams.append(cam)

            # Disable grad to avoid unnecessary computations in the next iteration
            grad_cam_features.requires_grad_(False)
        
        cams = torch.stack(cams) # Shape: [N_dets, H_feat, W_feat]

        # Assemble final output tensor
        cam_h, cam_w = 64, 64 # Standard size for attribution maps
        resized_cams = transforms.Resize([cam_h, cam_w])(cams.unsqueeze(1))
        
        cam_features = resized_cams.flatten(start_dim=1)
        
        detections = torch.cat([
            final_boxes,
            final_scores.unsqueeze(1),
            final_labels.unsqueeze(1).float(),
            cam_features,
            torch.full((final_scores.shape[0], 1), cam_h, device=out_logits.device),
            torch.full((final_scores.shape[0], 1), cam_w, device=out_logits.device),
        ], dim=1)
        
        # Dummy values for features and levels for compatibility
        features_dict = {}
        levels = torch.zeros(detections.shape[0], device=detections.device)
        
        return detections, features_dict, levels

def get_network(args):
    """Entry point function for ODAM framework."""
    from config_coco import config
    network = DETRODAMNetwork(config)
    
    if hasattr(args, 'resume_weights') and args.resume_weights is not None:
        model_path = os.path.join('../model', args.model_dir, 'dump-{}.pth'.format(args.resume_weights))
        if os.path.exists(model_path):
             network.load_pretrained_weights(model_path)
        else:
             # Fallback to init_weights from config if specific dump not found
             network.load_pretrained_weights(config.init_weights)
    else:
        network.load_pretrained_weights(config.init_weights)
        
    return network

# For compatibility
Network = DETRODAMNetwork
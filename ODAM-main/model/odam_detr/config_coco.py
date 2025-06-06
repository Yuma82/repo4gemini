import os
import sys

import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = '../../'
add_path(os.path.join(root_dir))
add_path(os.path.join(root_dir, 'lib'))

class COCO:
    root_folder = './data/coco'
    train_folder = os.path.join(root_folder, 'train2017')
    eval_folder = os.path.join(root_folder, 'val2017')
    test_folder = os.path.join(root_folder, 'testdev2017')
    train_source = os.path.join(root_folder,'annotations/instances_train2017.json')
    eval_source = os.path.join(root_folder,'annotations/instances_val2017.json')
    test_source = os.path.join(root_folder,'annotations/instances_testdev2017.json')

class Config:
    output_dir = 'coco_model'
    model_dir = output_dir
    # model_dir = os.path.join(output_dir, 'model_dump')
    eval_dir = os.path.join(output_dir, 'eval_dump')
    init_weights = '../data/model/resnet50_fbaug.pth'

    # ----------data config---------- #
    image_mean = np.array([103.530, 116.280, 123.675])
    image_std = np.array([57.375, 57.120, 58.395])
    to_bgr255 = True
    train_image_min_size = 800
    train_image_max_size = 1333
    train_image_min_size_range = (-1,-1) # -1 means disabled and it will use MIN_SIZE_TRAIN
    # If True, each batch should contain only images for which the aspect ratio
    # is compatible. This groups portrait images together, and landscape images
    # are not batched with portrait images.
    aspect_ratio_grouping = True
    size_divisible = 32
    num_workers = 2
    eval_resize = True
    eval_image_min_size = 800
    eval_image_max_size = 1333
    seed_dataprovider = 3
    train_source = COCO.train_source
    eval_source = COCO.eval_source
    test_source = COCO.test_source    
    train_folder = COCO.train_folder
    eval_folder = COCO.eval_folder
    test_folder = COCO.test_folder

    # ----------test config---------- #
    test_nms = 0.5
    test_nms_method = 'normal_nms'
    detection_per_image = 100
    visulize_threshold = 0.3
    pred_cls_threshold = 0.05
    drise_output_format = False

    # ----------dataset config---------- #
    nr_box_dim = 5
    max_boxes_of_image = 100

    # ----------DETR specific config---------- #
    num_classes = 91  # COCO has 91 classes (including background)
    num_queries = 100  # Number of object queries in DETR
    hidden_dim = 256   # Hidden dimension of transformer
    backbone = 'resnet50'
    position_embedding = 'sine'
    enc_layers = 6
    dec_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    nheads = 8
    pre_norm = False

config = Config()
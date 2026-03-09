
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcam.methods import GradCAM
import re
import torchvision.transforms as transforms
from PIL import Image
import os
import csv
import random 
import numpy as np

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_single_image(image_path, input_size=224):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0) 

def load_images_from_folder(folder_path, input_size=224, max_images=None):
   
    images = []
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    for img_path in image_paths:
        try:
            img_tensor = load_single_image(img_path, input_size)
            images.append(img_tensor)
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            image_paths.remove(img_path)
    
    if len(images) == 0:
        raise ValueError("No valid images loaded!")
    images_tensor = torch.cat(images, dim=0)
    return images_tensor, image_paths


def load_labels(label_path, image_paths, default_label=0):
    
    label_map = {}
    
    if label_path and os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f) 
            for row in reader:
                fname = row.get('filename', '').strip()
                label_str = row.get('label', '').strip()
                if fname and label_str.isdigit():
                    label_map[fname] = int(label_str)
    
    labels = []
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        labels.append(label_map.get(fname, default_label))
    
    matched = sum(1 for l in labels if l != default_label)
    print(f"Label loading completed: Total {len(labels)} images, {matched} labels matched, others use default value {default_label}")
    
    labels_tensor = torch.tensor(labels)
    labels_tensor = labels_tensor - 1
    
    print(f"Adjusted labels range: min={labels_tensor.min().item()}, max={labels_tensor.max().item()}")
    labels_tensor = torch.clamp(labels_tensor, 0, 999)
    
    return labels_tensor


def get_cam_target_layer(model):
   
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    
    model_name = _infer_model_name(model)
    
    conv_layers = []
    layer_full_names = []
    layer_short_names = []

    def _recursive_find_conv(module, parent_full_name="", parent_short_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_full_name}.{name}" if parent_full_name else name
            short_name = f"{parent_short_name}_{name}" if parent_short_name else name
            
            if isinstance(child, nn.Conv2d):
                conv_layers.append(child)
                layer_full_names.append(full_name)
                layer_short_names.append(short_name)
            
            elif len(list(child.named_children())) > 0:
                _recursive_find_conv(child, full_name, short_name)
    
    _recursive_find_conv(model)

    if "resnet" in model_name.lower() or "res-101" in model_name.lower():

        resnet_candidates = []
        for idx, full_name in enumerate(layer_full_names):
            if "layer4" in full_name and ("conv2" in full_name or "conv" in full_name):
                resnet_candidates.append(idx)
        if resnet_candidates:
            return conv_layers[resnet_candidates[-1]]
    
    elif "vgg" in model_name.lower():
        vgg_candidates = []
        for idx, full_name in enumerate(layer_full_names):
            if "features" in full_name and "conv" in full_name:
                vgg_candidates.append(idx)
        if vgg_candidates:
            return conv_layers[vgg_candidates[-1]]
    
    elif "inc-v3" in model_name.lower():
        inc3_candidates = []
        for idx, full_name in enumerate(layer_full_names):
            if "Mixed_7c" in full_name and "conv" in full_name:
                inc3_candidates.append(idx)
        if inc3_candidates:
            return conv_layers[inc3_candidates[-1]]
    
    elif "inc-v4" in model_name.lower():
        inc4_candidates = []
        for idx, full_name in enumerate(layer_full_names):
            if "Mixed_8e" in full_name and "conv" in full_name:
                inc4_candidates.append(idx)
        if inc4_candidates:
            return conv_layers[inc4_candidates[-1]]
    
    elif "incres-v2" in model_name.lower():
        incres2_candidates = []
        for idx, full_name in enumerate(layer_full_names):
            if ("Conv2d_7b_1x1" in full_name or "Block8_6" in full_name) and "conv" in full_name:
                incres2_candidates.append(idx)
        if incres2_candidates:
            return conv_layers[incres2_candidates[-1]]
    
    if conv_layers:
        return conv_layers[-1]
    
    raise ValueError(
        f"Model {model_name} has no Conv2d layers! Cannot select GradCAM target layer."
    )


def _infer_model_name(model):

    model_class_name = model.__class__.__name__.lower()
    
    ens_pattern = re.compile(r"(ens\d+|ens)$")

    if "resnet" in model_class_name or "res-101" in model_class_name:
        return "resnet"
    elif "inception3" in model_class_name or "inc-v3" in model_class_name:
        return re.sub(ens_pattern, "", "inc-v3")
    elif "inception4" in model_class_name or "inc-v4" in model_class_name:
        return re.sub(ens_pattern, "", "inc-v4")
    elif "inceptionresnetv2" in model_class_name or "incres-v2" in model_class_name:
        return re.sub(ens_pattern, "", "incres-v2")
    else:
        return re.sub(ens_pattern, "", model_class_name)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_key_mask(model, input_tensor, y):
    
    if isinstance(model, nn.DataParallel):
        original_model = model.module
    else:
        original_model = model
    original_model.eval() 
    device = next(original_model.parameters()).device
    input_tensor = input_tensor.to(device).float()

    
    target_layer = get_cam_target_layer(original_model)

    cam_extractor = GradCAM(model=original_model, target_layer=target_layer)
    
    output = model(input_tensor)
    if isinstance(output, tuple):
        logits = output[0]
    elif isinstance(output, dict):
        logits = output.get('logits', None)
    else:
        logits = output
    
    batch_size = input_tensor.shape[0]
    target_classes = y.detach().cpu().numpy().tolist()

    heatmaps = cam_extractor(scores=logits, class_idx=target_classes)
    if isinstance(heatmaps, list):
        heatmaps = torch.stack(heatmaps, dim=0)
    heatmaps = heatmaps.to(input_tensor.device)

    if heatmaps.dim() == 3: 
        heatmaps = heatmaps.unsqueeze(1)
    elif heatmaps.dim() == 4 and heatmaps.shape[0] == 1 and heatmaps.shape[1] == batch_size:
        heatmaps = heatmaps.permute(1, 0, 2, 3)
   
    heatmap_resized = F.interpolate(
        heatmaps,
        size=input_tensor.shape[2:],
        mode='bilinear',
        align_corners=False
    ) 

    mean = torch.mean(heatmap_resized, dim=(2, 3), keepdim=True)  
    std = torch.std(heatmap_resized, dim=(2, 3), keepdim=True) + 1e-6  
    weight_matrix = (heatmap_resized - mean) / std 
    key_mask = (weight_matrix > 1).float()  

    cam_extractor.remove_hooks()

    return key_mask


def addMask(input_tensor, key_mask, prob): 
    
    device = input_tensor.device
    input_tensor = input_tensor.to(device).float()

    non_key_random_mask = torch.bernoulli(
        torch.full_like(key_mask, prob, dtype=torch.float32)
    ).float()

    final_mask = torch.where(
        key_mask == 1.0,
        torch.ones_like(key_mask),
        non_key_random_mask
    )

    final_mask = final_mask.expand(-1, input_tensor.shape[1], -1, -1) 

    masked_tensor = input_tensor * final_mask

    return masked_tensor
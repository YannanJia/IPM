import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse 
from utils import *
from models import *


def parse_args():
    parser = argparse.ArgumentParser(description='MIM Adversarial Attack with GradCAM Mask')
    parser.add_argument('--epsilon', type=float, default=16./255,
                        help='Linf MI-FGSM attack epsilon')
    parser.add_argument('--k', type=int, default=10,
                        help='MI-FGSM attack iteration number')
    parser.add_argument('--alpha', type=float, default=2./255,
                        help='MI-FGSM attack step size ')
    parser.add_argument('--mu', type=float, default=1.0,
                        help='MI-FGSM momentum factor')
    parser.add_argument('--prob', type=float, default=1-0.001,
                        help='Non-key region keep probability')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Running device (default: auto select cuda/cpu)')
    parser.add_argument('--use_mask', action='store_true', default=False,
                        help='Whether to use GradCAM mask (addMask function) (default: False)')
    parser.add_argument('--N', type=int, default=10, 
                    help='Number of mask copies generated per iteration (default: 10)')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for reproducibility')
    parser.add_argument('--image_folder', type=str, default='./data/val_rs',
                        help='Folder containing test images')
    parser.add_argument('--label_path', type=str, default="./data/val_rs.csv",
                        help='Path to label file')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size for model (default: 224)')
    parser.add_argument('--max_images', type=int, default=1000,
                        help='Maximum number of images to process (default: 100)')
    parser.add_argument('--default_label', type=int, default=0,
                        help='Default label if no label file is provided (default: 0)')
    return parser.parse_args()


args = parse_args()  
# set_seed(args.seed)
epsilon = args.epsilon
k = args.k
alpha = args.alpha
mu = args.mu
prob = args.prob
device = args.device
use_mask = args.use_mask 
N = args.N

net1 = resnet18()
checkpoint = torch.load("./models/checkpoint/resnet18.pth", weights_only=True)
net1.load_state_dict(checkpoint)
net1 = torch.nn.DataParallel(net1).to(device)
net1.eval()


net2 = resnet50()
checkpoint = torch.load("./models/checkpoint/resnet50.pth", weights_only=True)
net2.load_state_dict(checkpoint)
net2 = torch.nn.DataParallel(net2).to(device)
net2.eval()


print(f"Loading images from folder: {args.image_folder}")
images, image_paths = load_images_from_folder(args.image_folder, args.input_size, args.max_images)
labels = load_labels(args.label_path, image_paths, args.default_label) 


test_dataset = ImageDataset(images, labels)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=100,
    shuffle=False 
)

class MIFGSMAttack(object):
    def __init__(self, model, use_mask): 
        self.model = model
        self.model.eval()
        self.use_mask = use_mask
        self.N = N

    def perturb(self, x_natural, y):
        x = x_natural.detach().to(device)

        key_mask = None
        if self.use_mask:
            key_mask = compute_key_mask(self.model, x_natural, y)

        momentum = torch.zeros_like(x_natural).to(device)

        for i in range(k): 
            x = x.detach() 
            grad_list = [] 

            if self.use_mask and key_mask is not None:
                for _ in range(self.N):
                    x_mask = addMask(x, key_mask, prob=prob)
                    x_mask.requires_grad_()
                    with torch.enable_grad():
                        logits = self.model(x_mask)
                        loss = F.cross_entropy(logits, y)
                    grad = torch.autograd.grad(loss, [x_mask])[0]
                    grad_list.append(grad.detach())
                grad = torch.stack(grad_list, dim=0).mean(dim=0)
            else:
                x_mask = x.detach()
                x_mask.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x_mask)
                    loss = F.cross_entropy(logits, y)
                grad = torch.autograd.grad(loss, [x_mask])[0]
            
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8) 
            
            momentum = mu * momentum + grad
            
            x = x + alpha * torch.sign(momentum)

            x = torch.clamp(x, x_natural - epsilon, x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        
        return x

adversary = MIFGSMAttack(net1, use_mask=use_mask)

def attack(x, y):
    adv = adversary.perturb(x, y)
    return adv

def run_eval():
    print(f'\nRunning params: epsilon={epsilon}, k={k}, alpha={alpha}, mu={mu}, prob={1-prob}, use_mask={use_mask}')

    net1.eval()
    net2.eval()
    
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        adv = attack(inputs, targets)
        adv_outputs = net2(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

        
        print(f"Batch {batch_idx}: Accuracy = {predicted.eq(targets).sum().item() / targets.size(0):.4f}")

    print(f"\nFinal attack success rate: {100. * (1 - adv_correct / total):.2f}%")
    print(f"Target model accuracy: {100. * adv_correct / total:.2f}%")


if __name__ == '__main__':
    run_eval()

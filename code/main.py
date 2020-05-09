import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as T
import torchvision.models as models
from torch.autograd import Variable

import copy

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-num_epochs", help="number of epochs.", type=int)
parser.add_argument(
    "-alpha", help="value of alpha (content loss weight).", type=float)
parser.add_argument(
    "-beta", help="value of beta (style loss weight).", type=float)
parser.add_argument("-rand", action="store_true",
                    help="whether or not to initialize with random noise.")
parser.add_argument("-run", nargs=2,
                    help="computes style transfer between content image and and style image. Expects relative file path to content and style image respectively.", type=str)
args = parser.parse_args()


def main():
    alpha = 1
    beta = 1e7
    num_epochs = 500
    if args.alpha:
        alpha = args.alpha
    if args.beta:
        beta = args.beta
    if args.num_epochs:
        num_epochs = args.num_epochs
    if args.run:
        img = style_transfer(cnn, args.run[0], args.run[1],
                             alpha=alpha, beta=beta,
                             num_epochs=num_epochs, init_random=args.rand)
        img = img.cpu().clone()
        img = img.squeeze(0)
        img = tensor_to_PIL(img)
        img.save("results/" + args.run[0].split("/")[-1][:-4] + "_"
                   + args.run[1].split("/")[-1][:-4] + ".jpg")


############################
##### Helper functions #####
############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()
for param in cnn.parameters():
    param.requires_grad = False

size = 512 if torch.cuda.is_available() else 128
loader = T.Compose([
    T.Resize(size),
    T.ToTensor()])

tensor_to_PIL = T.ToPILImage()


def load_image(img_path):
    """
    loads image given relative path
    """
    img = Image.open(img_path)
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)


def content_loss(x, target):
    """
    computes content loss
    """
    return F.mse_loss(x, target)


def gram_matrix(features):
    """
    computes gram matrix
    """
    N, C, H, W = features.size()
    features = features.view(N * C, H * W)
    gram = torch.mm(features, features.t())
    return gram.div(N * C * H * W)


def style_loss(x, target):
    """
    computes style loss
    """
    return F.mse_loss(gram_matrix(x), gram_matrix(target))


def normalize(x, mean, std):
    """
    normalize accorinding to mean and std
    """
    return (x - mean) / std


def extract_features(cnn, x, layers):
    """
    extract feaures from given layers of cnn when given input x
    """
    cnn = copy.deepcopy(cnn)
    features = []
    layer = 0
    stopping_layer = max(layers)
    prev = x
    for module in cnn.children():
        curr = module(prev)
        is_conv = False
        if isinstance(module, nn.Conv2d):
            layer += 1
            is_conv = True
        elif isinstance(module, nn.MaxPool2d):
            temp = nn.AvgPool2d(kernel_size=2, stride=2)
            curr = temp(prev)
        if layer in layers and is_conv:
            features.append(curr)
        if layer == stopping_layer and is_conv:
            break
        prev = curr
    return features


def style_transfer(cnn, content_path, style_path, num_epochs=500,
                   alpha=1, beta=1e6, init_random=False):
    """
    Run the style transfer.
    """
    style_img = load_image(style_path)
    content_img = load_image(content_path)
    if init_random:
        input_img = torch.randn(content_img.data.size(), device=device)
    else:
        input_img = content_img.clone()

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    input_var = Variable(input_img, requires_grad=True)
    content_var = Variable(content_img)
    style_var = Variable(style_img)

    content_var = normalize(content_var, mean, std)
    style_var = normalize(style_var, mean, std)

    content_targets = extract_features(cnn, content_var, [9])
    style_targets = extract_features(cnn, style_var, [1, 3, 5, 9, 13])
    optimizer = optim.LBFGS([input_var])

    epoch = 0
    while epoch <= num_epochs:
        def closure():
            input_var.data.clamp_(0, 1)
            optimizer.zero_grad()

            content_feats = extract_features(
                cnn, normalize(input_var, mean, std), [9])
            style_feats = extract_features(cnn, normalize(
                input_var, mean, std), [1, 3, 5, 9, 13])

            c_loss = sum(content_loss(
                content_feats[i], content_targets[i]) for i in range(len(content_targets)))
            s_loss = sum(style_loss(
                style_feats[i], style_targets[i]) for i in range(len(style_targets)))

            loss = alpha * c_loss + beta * s_loss
            loss.backward()

            return loss

        optimizer.step(closure)
        # steps 20 times
        epoch += 20
        if epoch % 100 == 0:
            print("Iteration " + str(epoch) + "/" + str(num_epochs))
    return input_img.data.clamp(0, 1)

if __name__ == "__main__":
    main()

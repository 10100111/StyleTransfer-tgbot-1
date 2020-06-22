from PIL import Image
import torch
import torchvision


def load_image_as_prepared_tensor(filename, size=None, return_start_size=False):
    img = Image.open(filename).convert('RGB')
    start_size = 0
    if return_start_size:
        start_size = img.size
    t = torchvision.transforms.Compose([
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor()
    ])
    img = t(img)
    img = img.unsqueeze(0)
    img = img.transpose(0, 1)
    (r, g, b) = torch.chunk(img, 3)
    img = torch.cat((b, g, r))
    img = img.transpose(0, 1)
    if return_start_size:
        return img, start_size
    return img


def tensor_save_as_image(tensor, filename, size=None):
    tensor = tensor.squeeze(0)
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    img = tensor.clone().clamp(0, 255).detach().numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    if size:
        img = img.resize(size)
    img.save(filename)

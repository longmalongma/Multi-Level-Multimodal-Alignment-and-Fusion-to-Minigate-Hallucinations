import torch
import torch.nn.functional as F

@torch.no_grad()
def get_DiT_latent_features(image, image_size, vae, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("GPU not found. Using CPU instead.")

    #MA默认为224，预训练的DiT为256/512
    if image_size!=256:
        new_size = (256,256)
        image = F.interpolate(image, size=new_size, mode='bilinear', align_corners=False)
    image_size = 256 #@param [256, 512]
    B,C,H,W = image.shape

    #通过vae获得latent_feature
    # Map input images to latent space + normalize latents:
    x = vae.encode(image).latent_dist.sample().mul_(0.18215) #B*32*32*4
    latent_size = int(image_size) // 8
    assert latent_size ==x.shape[-1], "出错了，latent feature不匹配"
    

    # Set user inputs:
    seed = 0 #@param {type:"number"}
    torch.manual_seed(seed)
    class_labels = torch.tensor([1000] * B, device=device)

    _,latent_features = model.forward(x, torch.zeros(x.shape[0]).to(device), class_labels)
    return latent_features
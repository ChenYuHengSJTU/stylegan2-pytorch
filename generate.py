import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

import os

def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema[0].eval()
        g_ema[1].eval()
        
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample_1, _ = g_ema[0](
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            sample_2, _ = g_ema[1](
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample_1,
                f"sample/1_{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            
            utils.save_image(
                sample_2,
                f"sample/2_{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema_1 = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema_2 = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    _, dirs, ckpts = next(os.walk("checkpoint"))
    print(ckpts)
    assert len(ckpts) == 2, "There should be exactly two directories in the checkpoint folder"
    
    checkpoint_1 = torch.load("checkpoint/" + ckpts[0])
    checkpoint_2 = torch.load("checkpoint/" + ckpts[1])
    
    # g_ema_1.load_state_dict(checkpoint_1["g_ema"] if 'g_ema' in checkpoint_1.keys() else checkpoint_1)
    # g_ema_2.load_state_dict(checkpoint_2["g_ema"] if 'g_ema' in checkpoint_2.keys() else checkpoint_2)

    g_ema_1.load_state_dict(checkpoint_1["g_ema"])
    g_ema_2.load_state_dict(checkpoint_2["g_ema"])
    # g_ema_1 = torch.load("checkpoint/" + dirs[0] + "/g_ema.pt")

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = [g_ema_1.mean_latent(args.truncation_mean), g_ema_2.mean_latent(args.truncation_mean)]
    else:
        mean_latent = None

    generate(args, [g_ema_1, g_ema_2], device, mean_latent)

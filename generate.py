import argparse
import os
from collections import defaultdict

import torch

from erasing.utils.utils import *

class_list = [
    "cassette player",
    "chain saw",
    "church",
    "gas pump",
    "tench",
    "garbage truck",
    "English springer",
    "golf ball",
    "parachute",
    "French horn",
]


def main(args):

    diffuser = StableDiffuser(scheduler="DDIM").to("cuda")
    erase_concept = args.erase_concept
    esd_path = f"models/esd-{erase_concept.lower().replace(' ','')}_from_{erase_concept.lower().replace(' ','')}-{args.train_method}_1-epochs_1000.pt"
    train_method = args.train_method

    finetuner = FineTunedModel(diffuser, train_method=train_method)
    finetuner.load_state_dict(torch.load(esd_path))
    seed = 1234
    generated_images = defaultdict(list)
    for cls in class_list:
        if args.finetuner:
            with finetuner:
                for _ in range(10):
                    images = diffuser(
                        f"an image of a {cls}",
                        img_size=512,
                        n_steps=50,
                        n_imgs=50,
                        generator=torch.Generator().manual_seed(seed),
                        guidance_scale=7.5,
                    )
                    generated_images[cls] += [image[0] for image in images]

        else:
            for _ in range(10):
                images = diffuser(
                    f"an image of a {cls}",
                    img_size=512,
                    n_steps=50,
                    n_imgs=50,
                    generator=torch.Generator().manual_seed(seed),
                    guidance_scale=7.5,
                )
                generated_images[cls] += [image[0] for image in images]

    for cls in generated_images:
        for idx, img in enumerate(generated_images[cls]):
            save_path = f"{args.output_dir}/{'finetuned_'+args.erase_concept if args.finetuner else 'sd'}/{cls.lower().replace(' ','')}_{idx}.jpg"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TrainESD", description="Finetuning stable diffusion to erase the concepts"
    )
    parser.add_argument(
        "--erase_concept", help="concept to erase", type=str, required=True
    )
    parser.add_argument(
        "--train_method",
        help="Type of method (xattn, noxattn, full, xattn-strict",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to directory to store images",
        type=str,
        default="output",
    )
    parser.add_argument(
        "--finetuner",
        help="Whether to use stable diffusion or finetuned model",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    main(args)

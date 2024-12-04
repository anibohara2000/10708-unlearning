import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from tqdm.auto import tqdm

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ImageClassification",
        description="Takes the path of images and generates classification scores",
    )
    parser.add_argument("--folder_path", help="path to images", type=str, required=True)
    parser.add_argument(
        "--save_path",
        help="path to save results",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--topk", type=int, required=False, default=5)
    parser.add_argument("--batch_size", type=int, required=False, default=250)
    args = parser.parse_args()

    folder = args.folder_path
    topk = args.topk
    device = args.device
    batch_size = args.batch_size
    save_path = args.save_path

    # Set default save path if not provided
    if save_path is None:
        name_ = folder.split("/")[-1]
        save_path = f"{folder}/{name_}_classification.csv"

    # Load ResNet50 model with pre-trained weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.to(device)
    model.eval()

    # Initialize dictionaries to store results
    def init_results_dict():
        scores = {}
        categories = {}
        indexes = {}
        for k in range(1, topk + 1):
            scores[f"top{k}"] = []
            indexes[f"top{k}"] = []
            categories[f"top{k}"] = []
        return scores, categories, indexes

    # Preprocess images using model's transforms
    preprocess = weights.transforms()

    # Function to load and preprocess images for classification
    def load_images(image_names, folder):
        images = []
        for name in image_names:
            img = Image.open(os.path.join(folder, name))
            batch = preprocess(img)
            images.append(batch)
        return torch.stack(images)

    # Perform classification in batches
    def classify_images(images):
        results_scores, results_indexes, results_categories = init_results_dict()

        for i in tqdm(range(((len(images) - 1) // batch_size) + 1)):
            batch_images = images[
                i * batch_size : min(len(images), (i + 1) * batch_size)
            ].to(device)
            with torch.no_grad():
                prediction = model(batch_images).softmax(1)
            probs, class_ids = torch.topk(prediction, topk, dim=1)

            for k in range(1, topk + 1):
                results_scores[f"top{k}"].extend(probs[:, k - 1].detach().cpu().numpy())
                results_indexes[f"top{k}"].extend(
                    class_ids[:, k - 1].detach().cpu().numpy()
                )
                results_categories[f"top{k}"].extend(
                    [
                        weights.meta["categories"][idx]
                        for idx in class_ids[:, k - 1].detach().cpu().numpy()
                    ]
                )

        return results_scores, results_indexes, results_categories

    # Classify and save results for a given directory of images
    def process_directory(image_names, folder_name, output_file):
        print(f"Classifying images from {folder_name}...")

        # Load and classify images from the given directory
        images = load_images(image_names, folder_name)
        scores, indexes, categories = classify_images(images)

        # Combine results into a final dictionary for saving as CSV
        dict_final = {"image_name": image_names}

        for k in range(1, topk + 1):
            dict_final[f"category_top{k}"] = categories[f"top{k}"]
            dict_final[f"index_top{k}"] = indexes[f"top{k}"]
            dict_final[f"scores_top{k}"] = scores[f"top{k}"]

        # Save classification result to CSV file
        df_results = pd.DataFrame(dict_final)
        df_results.to_csv(output_file, index=False)

    # Process non-finetuned images from output/sd/
    sd_folder_path = os.path.join(folder, "sd")

    if os.path.exists(sd_folder_path):
        names_sd = [
            name
            for name in os.listdir(sd_folder_path)
            if ".png" in name or ".jpg" in name
        ]

        if names_sd:
            sd_save_path = os.path.join(folder, "sd_classification.csv")
            process_directory(names_sd, sd_folder_path, sd_save_path)

    # Process finetuned images from output/{removed_class}/ directories (subdirectories within output/)
    removed_class_dirs = [
        d
        for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d)) and d != "sd"
    ]

    for removed_class_dir in removed_class_dirs:
        removed_class_folder_path = os.path.join(folder, removed_class_dir)

        names_removed_class = [
            name
            for name in os.listdir(removed_class_folder_path)
            if ".png" in name or ".jpg" in name
        ]

        if names_removed_class:
            removed_class_save_path = os.path.join(
                folder, f"{removed_class_dir}_classification.csv"
            )
            process_directory(
                names_removed_class, removed_class_folder_path, removed_class_save_path
            )

import argparse
import os
from collections import defaultdict

import pandas as pd

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
        description="Takes the path of classification csv files and outputs accuracy",
    )
    parser.add_argument(
        "--folder_path",
        help="path to classification csv files",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    folder = args.folder_path
    # Loop over all csv files in the folder
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".csv")]
    for file in files:
        # Load the csv file
        df = pd.read_csv(os.path.join(folder, file))
        total = defaultdict(int)
        correct = defaultdict(int)
        for idx, row in df.iterrows():
            # Check if the correct class is in the top 5 predictions
            filename = row["image_name"]
            true_label = filename.split("_")[0]
            pred_label = row["category_top1"].lower().replace(" ", "")

            total[true_label] += 1
            if pred_label == true_label:
                correct[true_label] += 1

        print()
        print(f"Results for {file}")

        if file.startswith("sd"):

            for cls in total:
                accuracy = correct[cls] / total[cls]
                other_accuracy = sum(v for k, v in correct.items() if k != cls) / sum(
                    v for k, v in total.items() if k != cls
                )
                print(
                    f"Accuracy for {cls}: {(accuracy*100):.2f}, other: {(other_accuracy*100):.2f}"
                )
        else:
            cls = [x for x in class_list if x in file][0].lower().replace(" ", "")
            accuracy = correct[cls] / total[cls]
            other_accuracy = sum(v for k, v in correct.items() if k != cls) / sum(
                v for k, v in total.items() if k != cls
            )
            print(
                f"Accuracy for {cls}: {(accuracy*100):.2f}, other: {(other_accuracy*100):.2f}"
            )

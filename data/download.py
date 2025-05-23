

import sys
import os
import gdown
from modular_legs import LEG_ROOT_DIR

def main():
    if len(sys.argv) < 2:
        print("Usage: python download.py <dataset_name>")
        print("Available datasets: designs_filtered")
        sys.exit(1)

    dataset_name = sys.argv[1]
    filename = ""
    data_type = None

    if dataset_name == "designs_filtered":
        url = "https://drive.google.com/uc?id=1pmF44Rnn9YBcjiC-Y8DJ5ToX4ZloWuth"
        filename = "designs_asym_filtered.pkl"
        data_type = 0
    else:
        print(f"Error: Unknown dataset '{dataset_name}'")
        sys.exit(1)

    if data_type == 0:
        target_dir = os.path.join(LEG_ROOT_DIR, "data/designs")
    elif data_type == 1:
        target_dir = os.path.join(LEG_ROOT_DIR, "data/rollouts")
    else:
        print(f"Error: Unknown data type '{data_type}'")
        sys.exit(1)

    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, filename)

    print(f"Downloading {dataset_name} using gdown API...")
    gdown.download(url, output_path, quiet=False)

    print(f"{dataset_name} downloaded to {output_path}")

if __name__ == "__main__":
    main()


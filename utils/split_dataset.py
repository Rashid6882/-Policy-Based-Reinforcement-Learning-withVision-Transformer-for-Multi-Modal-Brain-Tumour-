import os
import json
import random

def create_split(data_dir, output_path, sample_ratio=0.5, seed=42):
    random.seed(seed)
    all_cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("BraTS-GLI")]
    all_cases.sort()
    
    total_cases = len(all_cases)
    sample_size = int(total_cases * sample_ratio)
    sampled_cases = random.sample(all_cases, sample_size)
    sampled_cases.sort()
    
    val_size = int(len(sampled_cases) * 0.2)
    val_cases = sampled_cases[:val_size]
    train_cases = sampled_cases[val_size:]
    
    split_data = {
        "seed": seed,
        "total_available": total_cases,
        "sampled_count": len(sampled_cases),
        "train_count": len(train_cases),
        "val_count": len(val_cases),
        "train": train_cases,
        "val": val_cases
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=4)
        
    print(f"Dataset split saved to {output_path}: {len(train_cases)} train, {len(val_cases)} val cases.")
    return split_data

if __name__ == "__main__":
    data_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    output_path = "data/splits/train_val_split.json"
    create_split(data_dir, output_path, sample_ratio=0.5, seed=42)

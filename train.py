"""Regenerate hybrid_laptop_model.pth and model_info.pkl from laptops.csv.

This is not required to run the Streamlit app — the repo ships a
pre-trained hybrid_laptop_model.pth for that. Run this script if you
update laptops.csv and need to retrain, or if you want to reproduce the
shipped model from scratch.

NOTE: this reconstructs a reasonable training loop for
HybridLaptopRecommender. If your original training notebook used
different hyperparameters, a train/test split, or a different target
signal, adjust the constants and loss below to match what you actually
ran — this is meant as a working starting point, not a guaranteed replica
of the original training run.
"""
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from data import load_data
from model import HybridLaptopRecommender

FEATURE_COLS = ["ram", "storage", "display(in inch)", "rating", "price_usd"]
EMBEDDING_DIM = 30
EPOCHS = 20
LEARNING_RATE = 1e-3


def main():
    data, _ = load_data("laptops.csv")
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]

    num_items = len(data)
    num_users = num_items  # one synthetic "user" per row for this demo setup
    data["laptop_id"] = range(num_items)
    data["user_id"] = range(num_users)

    model = HybridLaptopRecommender(num_users, num_items, len(feature_cols), EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    user_ids = torch.tensor(data["user_id"].values, dtype=torch.long)
    item_ids = torch.tensor(data["laptop_id"].values, dtype=torch.long)
    features = torch.tensor(data[feature_cols].values.astype(float), dtype=torch.float)
    # Using rating as the training signal: the model learns to predict how
    # well-regarded a laptop is, and that signal drives the embeddings used
    # for similarity at inference time.
    targets = torch.tensor(data["rating"].values.astype(float), dtype=torch.float)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        preds = model(user_ids, item_ids, features)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{EPOCHS} — loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "hybrid_laptop_model.pth")
    with open("model_info.pkl", "wb") as f:
        pickle.dump(
            {
                "feature_cols": feature_cols,
                "num_users": num_users,
                "num_items": num_items,
                "embedding_dim": EMBEDDING_DIM,
            },
            f,
        )
    print("Saved hybrid_laptop_model.pth and model_info.pkl")


if __name__ == "__main__":
    main()

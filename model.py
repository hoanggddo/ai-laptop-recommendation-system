"""The hybrid recommendation model: learned embeddings + raw spec features.

Kept separate from app.py and data.py so the model architecture and
inference logic have no dependency on Streamlit or pandas parsing details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLaptopRecommender(nn.Module):
    """Combines learned user/item embeddings with raw spec features.

    Each laptop gets both a learned "item embedding" (what the model infers
    from interaction patterns) and a feature embedding (derived directly
    from its specs: RAM, storage, display, etc.). The two are combined so
    recommendations reflect both learned similarity and stated preferences.
    """

    def __init__(self, num_users: int, num_items: int, num_features: int, embedding_dim: int = 30):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_features = nn.Linear(num_features, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, item_ids, features):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        feature_embeds = self.fc_features(features)
        interaction = user_embeds * (item_embeds + feature_embeds)
        return self.fc(interaction).squeeze()


def load_model(model_path: str, num_users: int, num_items: int, num_features: int, embedding_dim: int):
    """Load a trained model's weights from disk and set it to eval mode."""
    model = HybridLaptopRecommender(num_users, num_items, num_features, embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def recommend_laptops(model, data, feature_cols, num_items, target_name=None, top_n=5, custom_features=None):
    """Return the top-N most similar laptops by embedding + feature similarity.

    Pass either target_name (recommend laptops similar to a specific one)
    or custom_features (recommend laptops matching a user's desired specs).
    """
    all_features = torch.tensor(
        data[feature_cols].iloc[:num_items].values.astype(float), dtype=torch.float
    )
    all_item_embeds = model.item_embedding.weight
    all_feature_embeds = model.fc_features(all_features)

    if target_name:
        target_row = data[data["name"] == target_name].iloc[0]
        target_idx = target_row.name
        target_embed = all_item_embeds[target_idx] + 0.5 * all_feature_embeds[target_idx]
    elif custom_features is not None:
        custom_features_tensor = torch.tensor(custom_features, dtype=torch.float)
        target_embed = 0.5 * model.fc_features(custom_features_tensor)
    else:
        return data.iloc[0:0]  # empty frame, same columns, nothing to recommend from

    similarities = F.cosine_similarity(
        target_embed.unsqueeze(0), all_item_embeds + 0.5 * all_feature_embeds
    )
    top_candidates = torch.topk(similarities, min(top_n * 3, len(data))).indices.tolist()
    return data.iloc[top_candidates]

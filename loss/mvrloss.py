import torch.nn as nn
import torch
import torch
# CONSTRUCTOR => REGULARIZER CONSTAN
# INPUT => BATCH_of_Embeddings, TARGET
# GENERATE ALL TRIPLET IN BATCH


def get_all_triplets_indices(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)

class MVR(nn.Module):

    def __init__(self, margin, reg):
        super(MVR, self).__init__()
        self.margin = margin
        self.reg = reg
        self.pairwise_euc = nn.PairwiseDistance(2)
        self.pairwise_cos = nn.CosineSimilarity(dim=1)
        self.relu = nn.ReLU()

    def forward(self, embeddings, labels):
        indices_tuple = get_all_triplets_indices(labels)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        ap_vec = embeddings[positive_idx] - embeddings[anchor_idx]
        an_vec = embeddings[negative_idx] - embeddings[anchor_idx]
        cos_dist = self.pairwise_cos(ap_vec, an_vec)
        ap_dist = self.pairwise_euc(embeddings[anchor_idx], embeddings[positive_idx])
        an_dist = self.pairwise_euc(embeddings[anchor_idx], embeddings[negative_idx])
        penalties = self.relu(ap_dist - an_dist + self.margin - self.reg*cos_dist)
        return torch.mean(penalties)

if __name__ == '__main__':
    loss = MVR(0.1, 0.4)
    embeddings = torch.randn(32, 128)
    target = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    target = target.repeat(4)
    net_loss = loss(embeddings, target)
    print("{}".format(net_loss))






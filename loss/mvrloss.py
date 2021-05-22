import torch.nn as nn
import torch
import torch
import numpy as np


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


class MVR_Triplet(nn.Module):

    def __init__(self, margin, reg):
        super(MVR_Triplet, self).__init__()
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
        penalties = self.relu(ap_dist - an_dist + self.margin - self.reg * cos_dist)
        return torch.mean(penalties)


class MVR_Proxy(nn.Module):

    def __init__(self, reg, no_class, embedding_dimension):
        super(MVR_Proxy, self).__init__()
        self.reg = reg
        self.no_class = no_class
        self.emb_dim = embedding_dimension
        with torch.no_grad():
            vec = np.random.randn(self.no_class, self.emb_dim)
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
            self.proxies = nn.Parameter((torch.from_numpy(vec)).float())

    def forward(self, embeddings, labels):
        anchor_embeddings = embeddings
        anchor_pr_dist = torch.cdist(anchor_embeddings, self.proxies)

        ## Computation of euclidean distance term in not only nominator but also denominator
        euc_pproxy_anchor = anchor_pr_dist[range(len(labels)), labels]
        mask_euc = torch.ones(anchor_pr_dist.shape, dtype=torch.bool)
        mask_euc[range(len(labels)), labels] = torch.tensor([False])
        euc_nproxy_anchor = anchor_pr_dist[mask_euc]
        euc_nproxy_anchor = euc_nproxy_anchor.view(anchor_pr_dist.size()[0], -1)
        ## Computation of cosine term in denominator
        torch.utils.backcompat.broadcast_warning.enabled = True
        proxy_anc_dif = self.proxies.unsqueeze(0) - anchor_embeddings.unsqueeze(1)
        mask_cos = torch.ones(proxy_anc_dif.shape, dtype=torch.bool)
        mask_cos[range(len(labels)), labels, :] = torch.tensor([False])
        fn_fa = proxy_anc_dif[mask_cos].view(-1, self.no_class - 1, self.emb_dim)
        norm_fn_fa = torch.linalg.norm(fn_fa, dim=2, keepdims=True)
        normalized_fn_fa = torch.div(fn_fa, norm_fn_fa)
        normalized_fn_fa = normalized_fn_fa.transpose(2, 1)
        fp_fa = (self.proxies[labels] - anchor_embeddings)
        norm_fp_fa = torch.linalg.norm(fp_fa, dim=1, keepdims = True)
        normalized_fp_fa = torch.div(fp_fa, norm_fp_fa)
        normalized_fp_fa = normalized_fp_fa.unsqueeze(1)
        cosine_distances = torch.bmm(normalized_fp_fa, normalized_fn_fa)
        cosine_distances = cosine_distances.squeeze(1)
        ## General Cost Computation
        unexp_denominator = -1 * (euc_nproxy_anchor + self.reg * cosine_distances)  # TODO: stability of exp
        exp_denominator = torch.exp(unexp_denominator)
        tot_denominator = torch.sum(exp_denominator, dim=1)
        exp_nominator = torch.exp(-1*euc_pproxy_anchor)
        unlog_loss = torch.div(exp_nominator, tot_denominator)
        log_loss = -1 * torch.log(unlog_loss)

        return log_loss.mean()


if __name__ == '__main__':
    loss = MVR_Proxy(0.1, 8, 128)
    vec = np.random.randn(32, 128)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    embeddings = torch.from_numpy(vec).float()
    target = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    target = target.repeat(4)
    net_loss = loss(embeddings, target)
    print("{}".format(net_loss))

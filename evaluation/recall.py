import torch
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from dataloader.trsfrms import unnormalize
import cv2
# 1. EXTRACT EMBEDDINGS
# 2. CALCULATE RECALL

idx_to_recall_at = {1: 1, 2: 2, 3: 4, 4: 8}


def give_recall(model, data_loader, cuda = None, visualize = False, dataset = None):
    model.eval()
    model.to(cuda)
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch, label in data_loader:
            batch = batch.to(cuda)
            label = label.to(cuda)
            embeddings = model(batch)
            all_embeddings.append(embeddings)
            all_labels.append(label)
    tnsr_all = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels)
    no_test_exp = tnsr_all.size()[0]
    Recalls = np.zeros([4], dtype=np.float32)
    for cur_query in tqdm(range(tnsr_all.size()[0])):
        query = tnsr_all[cur_query, :]
        dist_btw_cur_query_and_ref = torch.cdist(query.unsqueeze(0), tnsr_all)
        _, indices = torch.topk(dist_btw_cur_query_and_ref, k=9, largest=False)
        indices = indices.cpu().numpy().ravel()
        query_label = all_labels[cur_query].cpu().numpy()
        retrieved_label = all_labels[indices[1:]].cpu().numpy()
        for k in [1, 2, 3, 4]:
            if query_label in retrieved_label[: idx_to_recall_at[k]]:
                Recalls[k - 1] = Recalls[k - 1] + 1
    Recalls = (Recalls / no_test_exp)
    if visualize == True:
        plt.figure(figsize=(15, 15))
        random_query_idx = random.sample(range(tnsr_all.size()[0]), 5)
        queries = tnsr_all[random_query_idx, :]
        dist_btw_cur_query_and_ref = torch.cdist(queries, tnsr_all)
        _, indices = torch.topk(dist_btw_cur_query_and_ref, k = 6, dim = 1, largest= False)
        for query_no, retrieved_indices in enumerate(indices):
            query_img = unnormalize(dataset[random_query_idx[query_no]][0])
            plt.subplot(5, 6, 6*query_no + 1)
            plt.imshow(query_img)
            cur_indicies = retrieved_indices[1:]
            for question_no, questioned in enumerate(cur_indicies):
                if all_labels[random_query_idx[query_no]] == all_labels[questioned]:
                    unbordered_img = unnormalize(dataset[questioned][0])
                    bordered_img = cv2.copyMakeBorder(unbordered_img,10,10,10,10,cv2.BORDER_CONSTANT, value=[0, 255,0])
                    plt.subplot(5, 6, 6*query_no + (question_no + 2))
                    plt.imshow(bordered_img)
                else:
                    unbordered_img = unnormalize(dataset[questioned][0])
                    bordered_img = cv2.copyMakeBorder(unbordered_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT,value=[255, 0, 0])
                    plt.subplot(5, 6, 6*query_no + (question_no+2))
                    plt.imshow(bordered_img)
        plt.show()

    return Recalls

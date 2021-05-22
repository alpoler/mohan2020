from dataloader.cub_dataset import CUB
from model.bn_inception import bn_inception
import numpy as np
import random
import torch
import torchvision.transforms as trsfrm
from dataloader.trsfrms import must_transform
from evaluation.recall import give_recall
from torch.utils.data import DataLoader
import logging
# seeds
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}/test.log".format('log', "proxy_exp20")),
        logging.StreamHandler()
    ]
)

cuda = 'cuda:1'
root_dir = "./dataloader/data/CUB_200_2011/images"
transforms_test = trsfrm.Compose([must_transform(), trsfrm.Resize(256), trsfrm.CenterCrop(224)])
cub_test = CUB(root_dir, 'test', transforms_test)
test_loader = DataLoader(cub_test, batch_size=64, num_workers=8, shuffle=False, pin_memory=True)
saved_model_dir = "./MVR_proxy/exp20/best.pth"
net = bn_inception(embedding_size=64, pretrained=True, is_norm=True, bn_freeze=False)
net.load_state_dict(torch.load(saved_model_dir))
net.to(cuda)
net.eval()
with torch.no_grad():
    Recalls = give_recall(net, test_loader,cuda= cuda, visualize=True, dataset=cub_test)
    logging.info("Recall@1 = {}, Recall@2 = {}, Recall@4 = {}, Recall@8 = {}".format(Recalls[0], Recalls[1],Recalls[2], Recalls[3]))



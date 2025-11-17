import torch
import numpy as np
import random
from Microservices.RCA.CRFD.model.Trianer import CFTrainer
from Microservices.RCA.CRFD.model.CRFD import CRFD


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    dataset='TT'
    for i in range(5):
        num_node_type = 5
        num_edge_types = 7
        edge_type_emb_dim = 7
        node_dim = 303

        batch_size = 128
        n_epochs = 15
        encoder_layer = 1
        decoder_layer = 1

        out_channels = 16
        fpr = 0.5
        lr = 0.001
        lrstr = '0.001'

        model_name = "./Experiments/" + dataset + "/best_network.pth"
        model_path = "./Experiments/" + dataset + "/"

        setup_seed(42+i)
        CFTrainerI = CFTrainer(batch_size, lr, n_epochs,out_channels, model_path, dataset)
        net = CRFD(encoder_layer, decoder_layer, node_dim, out_channels, num_node_type,
                     num_edge_types, edge_type_emb_dim)

        net = CFTrainerI.train(net)

        net.load_state_dict(torch.load(model_name))
        normal_pattern, threshold = CFTrainerI.train_GMM(net,fpr)

        # CFTrainerI.validate(net,normal_pattern,threshold)
        CFTrainerI.test(net,normal_pattern,threshold)


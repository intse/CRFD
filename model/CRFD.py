import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from Microservices.RCA.CRFD.model.NTConv import NTConv
import copy
from collections import defaultdict


class CRFD(nn.Module):
    def __init__(self, encoder_layer,decoder_layer, node_dim, out_channels,num_node_type,
                     num_edge_types, edge_type_emb_dim):
        super(CRFD,self).__init__()

        self.encoder_convs = nn.ModuleList()
        for _ in range(encoder_layer):
            conv = NTConv(in_channels=node_dim,out_channels=2*out_channels,num_node_types=num_node_type,
                          num_edge_types=num_edge_types,edge_type_emb_dim=edge_type_emb_dim)
            self.encoder_convs.append(conv)

        self.decoder_convs = nn.ModuleList()
        for _ in range(decoder_layer):
            conv = NTConv(in_channels=out_channels,out_channels=node_dim,num_node_types=num_node_type,
                          num_edge_types=num_edge_types,edge_type_emb_dim=edge_type_emb_dim)
            self.decoder_convs.append(conv)

    def do_event(self, data, event_do_library, template_vector, do_idx):
        can_data_do = []
        if do_idx in data.edge_index[1]:
            mask_edges = (data.edge_index[1] == do_idx) & (
                        (data.node_type[data.edge_index[0]] == 0) | (data.node_type[data.edge_index[0]] == 1))
            source_nodes = data.edge_index[0, mask_edges.nonzero(as_tuple=True)[0]]

            parent_template = [data.node_event[i].item() for i in source_nodes]

            for tid in parent_template:
                if tid in event_do_library:
                    for (ct, _) in event_do_library[tid]:
                        data_do_ct = copy.deepcopy(data)
                        do_feature_ct = template_vector[ct] + [0] * 3
                        data_do_ct.x[do_idx] = torch.tensor(do_feature_ct, dtype=torch.float32, device=data.x.device)
                        can_data_do.append(data_do_ct)

            if len(can_data_do) == 0:
                data_do_ct = copy.deepcopy(data)
                data_do_ct.x[do_idx] = torch.zeros_like(data.x[do_idx])
                can_data_do.append(data_do_ct)
        else:
            data_do_ct = copy.deepcopy(data)
            data_do_ct.x[do_idx] = torch.zeros_like(data.x[do_idx])
            can_data_do.append(data_do_ct)

        do_nodes = set([do_idx])
        child_mask = data.edge_index[0] == do_idx
        children = data.edge_index[1, child_mask].unique().tolist()
        do_nodes.update(children)
        return can_data_do, list(do_nodes)

    def do_CPUMemory(self, data, servicelist, cpu_pattern, memory_pattern, do_idx):
        data_do = copy.deepcopy(data)
        mask_nodes = (data.pre_node_service == data.pre_node_service[do_idx]) & (
                    data.node_type == data.node_type[do_idx])
        mask_nodes_index = mask_nodes.nonzero(as_tuple=True)[0]

        mean_idx = data.node_mean[do_idx]
        std_idx = data.node_std[do_idx]
        do_feature = torch.zeros((1, data.x.shape[1]), dtype=torch.float32, device=data.x.device)
        ser_name = servicelist[data.pre_node_service[do_idx].item()]
        if data.node_type[do_idx] == 2:
            do_feature[:, data.x.shape[1] - 3] = (cpu_pattern[ser_name][0] - mean_idx) / std_idx
        else:
            do_feature[:, data.x.shape[1] - 2] = (memory_pattern[ser_name][0] - mean_idx) / std_idx

        data_do.x[mask_nodes_index] = do_feature
        do_nodes = set()
        for node in mask_nodes_index:
            child_mask = data.edge_index[0] == node
            node_children = data.edge_index[1, child_mask].unique().tolist() + [node.item()]
            do_nodes.update(node_children)

        return data_do, list(do_nodes)

    def do_network_scheme(self,data,pre_service_out_mask_nodes_indices,pre_service_in_mask_nodes_indices,
                                        post_service_out_mask_nodes_indices,post_service_in_mask_nodes_indices,
                                        pre_service,post_service,network_pattern,servicelist):
        pre_in_networks = data.x[pre_service_in_mask_nodes_indices, data.x.shape[1] - 1]
        pre_out_networks = data.x[pre_service_out_mask_nodes_indices, data.x.shape[1]-1]
        post_in_networks = data.x[post_service_in_mask_nodes_indices, data.x.shape[1]-1]
        post_out_networks = data.x[post_service_out_mask_nodes_indices, data.x.shape[1]-1]

        pre_out_nodes_network_max = torch.zeros_like(data.node_mean[pre_service_out_mask_nodes_indices])
        for i,idx in enumerate(pre_service_out_mask_nodes_indices):
            pre_service_name = servicelist[pre_service]
            post_service_name = servicelist[data.post_node_service[idx]]
            if (pre_service_name,post_service_name) in network_pattern:
                network_max = network_pattern[(pre_service_name, post_service_name)][2]
            else:
                network_max = network_pattern[(post_service_name, pre_service_name)][2]
            mean_idx = data.node_mean[idx]
            std_idx = data.node_std[idx]
            pre_out_nodes_network_max[i]=(network_max-mean_idx)/std_idx

        pre_in_nodes_network_max = torch.zeros_like(data.node_mean[pre_service_in_mask_nodes_indices])
        for i,idx in enumerate(pre_service_in_mask_nodes_indices):
            pre_service_name = servicelist[data.pre_node_service[idx]]
            post_service_name = servicelist[pre_service]
            if (pre_service_name, post_service_name) in network_pattern:
                network_max = network_pattern[(pre_service_name, post_service_name)][2]
            else:
                network_max = network_pattern[(post_service_name, pre_service_name)][2]
            mean_idx = data.node_mean[idx]
            std_idx = data.node_std[idx]
            pre_in_nodes_network_max[i]=(network_max-mean_idx)/std_idx

        post_out_nodes_network_max = torch.zeros_like(data.node_mean[post_service_out_mask_nodes_indices])
        for i,idx in enumerate(post_service_out_mask_nodes_indices):
            pre_service_name = servicelist[post_service]
            post_service_name = servicelist[data.post_node_service[idx]]
            if (pre_service_name, post_service_name) in network_pattern:
                network_max = network_pattern[(pre_service_name, post_service_name)][2]
            else:
                network_max = network_pattern[(post_service_name, pre_service_name)][2]
            mean_idx = data.node_mean[idx]
            std_idx = data.node_std[idx]
            post_out_nodes_network_max[i]=(network_max-mean_idx)/std_idx

        post_in_nodes_network_max = torch.zeros_like(data.node_mean[post_service_in_mask_nodes_indices])
        for i,idx in enumerate(post_service_in_mask_nodes_indices):
            pre_service_name = servicelist[data.pre_node_service[idx]]
            post_service_name = servicelist[post_service]
            if (pre_service_name, post_service_name) in network_pattern:
                network_max = network_pattern[(pre_service_name, post_service_name)][2]
            else:
                network_max = network_pattern[(post_service_name, pre_service_name)][2]
            mean_idx = data.node_mean[idx]
            std_idx = data.node_std[idx]
            post_in_nodes_network_max[i]=(network_max-mean_idx)/std_idx

        if (pre_in_networks < pre_in_nodes_network_max).any() or (pre_out_networks<pre_out_nodes_network_max).any():
            return 2
        elif (post_in_networks < post_in_nodes_network_max).any() or (post_out_networks < post_out_nodes_network_max).any():
            return 1
        else:
            pre_mask_nodes_indices = torch.unique(torch.cat((pre_service_out_mask_nodes_indices, pre_service_in_mask_nodes_indices)))
            post_mask_nodes_indices = torch.unique(torch.cat((post_service_out_mask_nodes_indices, post_service_in_mask_nodes_indices)))
            if len(pre_mask_nodes_indices) > len(post_mask_nodes_indices):
                return 1
            else:
                return 2

    def do_Network(self,data, servicelist,network_pattern, do_idx):
        data_do_pre= copy.deepcopy(data)
        data_do_post = copy.deepcopy(data)
        pre_service = data.pre_node_service[do_idx].item()
        post_service = data.post_node_service[do_idx].item()
        mean_idx = data.node_mean[do_idx]
        std_idx = data.node_std[do_idx]

        pre_serivce_out_mask_nodes = (data.pre_node_service == pre_service) & (data.node_type == data.node_type[do_idx])
        pre_service_in_mask_nodes = (data.post_node_service == pre_service) & (data.node_type == data.node_type[do_idx])
        pre_service_out_mask_nodes_indices = torch.nonzero(pre_serivce_out_mask_nodes).squeeze(dim=1)
        pre_service_in_mask_nodes_indices = torch.nonzero(pre_service_in_mask_nodes).squeeze(dim=1)
        pre_mask_nodes_indices = torch.unique(torch.cat((pre_service_out_mask_nodes_indices, pre_service_in_mask_nodes_indices)))

        post_serivce_out_mask_nodes = (data.pre_node_service == post_service) & (data.node_type == data.node_type[do_idx])
        post_serivce_in_mask_nodes = (data.post_node_service == post_service) & (data.node_type == data.node_type[do_idx])
        post_service_out_mask_nodes_indices = torch.nonzero(post_serivce_out_mask_nodes).squeeze(dim=1)
        post_service_in_mask_nodes_indices = torch.nonzero(post_serivce_in_mask_nodes).squeeze(dim=1)
        post_mask_nodes_indices = torch.unique(torch.cat((post_service_out_mask_nodes_indices, post_service_in_mask_nodes_indices)))

        scheme = self.do_network_scheme(data,pre_service_out_mask_nodes_indices,pre_service_in_mask_nodes_indices,
                                        post_service_out_mask_nodes_indices,post_service_in_mask_nodes_indices,
                                        pre_service,post_service,network_pattern,servicelist)

        if scheme==1:
            for i in pre_mask_nodes_indices:
                i_pre_service = servicelist[data.pre_node_service[i].item()]
                i_post_service = servicelist[data.post_node_service[i].item()]
                do_feature_i = torch.zeros((1, data.x.shape[1]), dtype=torch.float32, device="cuda:0")
                if (i_pre_service, i_post_service) in network_pattern:
                    do_feature_i[:, data.x.shape[1]-1] = (network_pattern[(i_pre_service, i_post_service)][0] - mean_idx) / std_idx
                else:
                    do_feature_i[:, data.x.shape[1]-1] = (network_pattern[(i_post_service, i_pre_service)][0] - mean_idx) / std_idx
                data_do_pre.x[i] = do_feature_i

            do_pre_nodes = set()
            for node in pre_mask_nodes_indices:
                child_mask = data.edge_index[0] == node
                node_children = data.edge_index[1, child_mask].unique().tolist() + [node.item()]
                do_pre_nodes.update(node_children)

            return data_do_pre, list(do_pre_nodes), None, None

        else:
            for i in post_mask_nodes_indices:
                i_pre_service = servicelist[data.pre_node_service[i].item()]
                i_post_service = servicelist[data.post_node_service[i].item()]
                do_feature_i = torch.zeros((1, data.x.shape[1]), dtype=torch.float32, device="cuda:0")
                if (i_pre_service, i_post_service) in network_pattern:
                    do_feature_i[:, data.x.shape[1]-1] = (network_pattern[(i_pre_service, i_post_service)][0] - mean_idx) / std_idx
                else:
                    do_feature_i[:, data.x.shape[1]-1] = (network_pattern[(i_post_service, i_pre_service)][0] - mean_idx) / std_idx
                data_do_post.x[i] = do_feature_i

            do_post_nodes = set()
            for node in post_mask_nodes_indices:
                child_mask = data.edge_index[0] == node
                node_children = data.edge_index[1, child_mask].unique().tolist() + [node.item()]
                do_post_nodes.update(node_children)

            return None, None, data_do_post, list(do_post_nodes)


    def sampling(self, mean, log_std):
        std = torch.exp(log_std)
        qz_x = torch.distributions.Normal(mean, std)
        z = qz_x.rsample()
        return z


    def encoder(self, x, edge_index, node_type, edge_type):
        for conv in self.encoder_convs:
            x = conv(x,edge_index,node_type,edge_type)
        mean, log_std = torch.chunk(x, 2, dim=1)
        return mean,log_std

    def decoder(self, z,edge_index,node_type,edge_type):
        for conv in self.decoder_convs:
            z = conv(z,edge_index,node_type,edge_type)
        return z


    def forward(self, x, node_type, edge_index, edge_type, batch):
        mean,log_std = self.encoder(x, edge_index, node_type, edge_type)
        mu_logstd=torch.cat((mean,log_std),dim=1)
        h_=global_mean_pool(mu_logstd,batch)
        z = self.sampling(mean, log_std)
        x_hat = self.decoder(z, edge_index, node_type, edge_type)
        return h_, x_hat, mean, log_std


    def root_cause_locate(self, data, normal_pattern, event_do_library, template_vector,
                          servicelist, nodeType, cpu_pattern, memory_pattern, network_pattern):
        x, edge_index, node_type, edge_type, batch = data.x, data.edge_index, data.node_type, data.edge_type, data.batch
        U_mean, U_logstd = self.encoder(x, edge_index, node_type, edge_type)

        result_list = []
        for idx in range(x.shape[0]):
            U_mean_i = copy.deepcopy(U_mean)
            U_logstd_i = copy.deepcopy(U_logstd)

            if node_type[idx].item() in [0, 1]:
                data_do_list, do_nodes = self.do_event(data, event_do_library, template_vector, idx)
                data_do_max_score = -float('inf')
                for data_do_i in data_do_list:
                    do_mean, do_logstd = self.encoder(data_do_i.x, data_do_i.edge_index, data_do_i.node_type,
                                                      data_do_i.edge_type)
                    U_mean_tmp = copy.deepcopy(U_mean_i)
                    U_logstd_tmp = copy.deepcopy(U_logstd_i)
                    U_mean_tmp[do_nodes, :] = do_mean[do_nodes, :]
                    U_logstd_tmp[do_nodes, :] = do_logstd[do_nodes, :]
                    cf_mean_logstd = torch.cat((U_mean_tmp, U_logstd_tmp), dim=1)
                    cf_h = global_mean_pool(cf_mean_logstd, batch).cpu().detach().numpy()
                    score = normal_pattern.score_samples(cf_h).item()
                    if score > data_do_max_score:
                        data_do_max_score = score
                root_cause = nodeType[node_type[idx].item()] + "_" + str(data.node_event[idx].item())
                result_list.append({"events": idx, "resource": root_cause, "score": data_do_max_score,
                                    "pod": servicelist[data.pre_node_service[idx].item()]})
            elif node_type[idx].item() in [2, 3]:
                data_do, do_nodes = self.do_CPUMemory(data, servicelist, cpu_pattern, memory_pattern, idx)
                do_mean, do_logstd = self.encoder(data_do.x, data_do.edge_index, data_do.node_type,
                                                  data_do.edge_type)
                U_mean_i[do_nodes, :] = do_mean[do_nodes, :]
                U_logstd_i[do_nodes, :] = do_logstd[do_nodes, :]
                cf_mean_logstd = torch.cat((U_mean_i, U_logstd_i), dim=1)
                cf_h = global_mean_pool(cf_mean_logstd, batch).cpu().detach().numpy()
                score = normal_pattern.score_samples(cf_h).item()
                result_list.append({"events": idx, "resource": nodeType[node_type[idx].item()], "score": score,
                                    "pod": servicelist[data.pre_node_service[idx].item()]})
            else:
                data_do_pre, do_nodes_pre, data_do_post, do_nodes_post = self.do_Network(data, servicelist,
                                                                                         network_pattern, idx)
                U_mean_i = U_mean.clone()
                U_logstd_i = U_logstd.clone()

                if data_do_pre is not None:
                    do_mean, do_logstd = self.encoder(
                        data_do_pre.x, data_do_pre.edge_index, data_do_pre.node_type, data_do_pre.edge_type
                    )
                    U_mean_i[do_nodes_pre, :] = do_mean[do_nodes_pre, :]
                    U_logstd_i[do_nodes_pre, :] = do_logstd[do_nodes_pre, :]

                    cf_mean_logstd = torch.cat((U_mean_i, U_logstd_i), dim=1)
                    cf_h = global_mean_pool(cf_mean_logstd, batch).cpu().detach().numpy()
                    score = normal_pattern.score_samples(cf_h).item()

                    result_list.append({
                        "events": idx,
                        "resource": nodeType[node_type[idx].item()],
                        "score": score,
                        "pod": servicelist[data.pre_node_service[idx].item()]
                    })

                else:
                    do_mean, do_logstd = self.encoder(
                        data_do_post.x, data_do_post.edge_index, data_do_post.node_type, data_do_post.edge_type
                    )
                    U_mean_i[do_nodes_post, :] = do_mean[do_nodes_post, :]
                    U_logstd_i[do_nodes_post, :] = do_logstd[do_nodes_post, :]

                    cf_mean_logstd = torch.cat((U_mean_i, U_logstd_i), dim=1)
                    cf_h = global_mean_pool(cf_mean_logstd, batch).cpu().detach().numpy()
                    score = normal_pattern.score_samples(cf_h).item()

                    result_list.append({
                        "events": idx,
                        "resource": nodeType[node_type[idx].item()],
                        "score": score,
                        "pod": servicelist[data.post_node_service[idx].item()]
                    })

        max_score_events = defaultdict(lambda: {"score": float('-inf')})
        for event in result_list:
            key = (event["resource"], event["pod"])
            if event["resource"] in ["CPU", "Memory", "Network"]:
                if event["score"] > max_score_events[key]["score"]:
                    max_score_events[key] = event
            else:
                max_score_events[(event["resource"], event["pod"], event["events"])] = event

        result_list = list(max_score_events.values())
        result_list = sorted(result_list, key=lambda i: i['score'], reverse=True)
        return result_list
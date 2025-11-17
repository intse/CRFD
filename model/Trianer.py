import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from Microservices.RCA.CRFD.dataset.TT.TTDataset import TTDataset
from Microservices.RCA.CRFD.dataset.SN.SNDataset import SNDataset
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score
import random
from Microservices.observability.utils.log import Logger
import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from collections import defaultdict


class CFTrainer():
    def __init__(self,batch_size,lr,n_epochs,out_channels,model_path,dataName):
        super().__init__()
        self.device="cuda:0"
        self.dataName=dataName
        self.model_path=model_path
        self.log_path = self.model_path+"trainlog"
        self.logger = Logger(self.log_path, logging.INFO, __name__).getlog()

        if self.dataName=='TT':
            self.dataset=TTDataset("./OBD/TT/")
            self.train_index, self.val_index, self.test_index, self.val_anomaly_index, self.test_anomaly_index = self.TT_data_index_generate()
            self.event_do_library = self.read_event_do_library("./OBD/TT/raw/TT_normal_pattern.txt")
            self.template_vector = self.read_log_template_vector("./OBD/TT/raw/log_template_vector300_depth10.txt")
            self.cpu_pattern, self.memory_pattern, self.network_pattern = self.read_memory_pattern("./OBD/TT/raw/")
            self.servicelist = ['ts-cancel-service', 'ts-order-service', 'ts-rebook-service', 'ts-seat-service', 'ts-food-service', 'ts-user-service', 'ts-assurance-service', 'ts-travel-service', 'ts-price-service', 'ts-execute-service', 'ts-train-service', 'ts-inside-payment-service', 'ts-config-service', 'ts-basic-service', 'ts-travel2-service', 'ts-security-service', 'ts-consign-service', 'ts-order-other-service', 'ts-preserve-service', 'ts-delivery-service', 'ts-payment-service', 'ts-contacts-service', 'ts-preserve-other-service', 'ts-route-service', 'ts-station-service', 'ts-consign-price-service']

        else:
            self.dataset = SNDataset("./OBD/SN/")
            self.train_index, self.val_index, self.test_index, self.val_anomaly_index, self.test_anomaly_index = self.Other_data_index_generate()
            self.event_do_library = self.read_event_do_library("./OBD/SN/raw/SN_normal_pattern.txt")
            self.template_vector = self.read_log_template_vector("./OBD/SN/raw/log_template_vector300_depth10.txt")
            self.cpu_pattern,self.memory_pattern,self.network_pattern = self.read_memory_pattern("./OBD/SN/raw/")
            self.servicelist = ['compose-post-service', 'home-timeline-service', 'url-shorten-service', 'user-service',
                                'user-timeline-service', 'post-storage-service', 'social-graph-service', 'nginx-web-server',
                                'user-mention-service', 'media-service', 'unique-id-service', 'text-service']

        self.batch_size=batch_size
        self.lr=lr
        self.n_epochs=n_epochs
        self.nodeType=["LogEvent","SpanEvent","CPU","Memory","Network"]
        self.normal_pattern = None
        self.threshold = None

    def read_memory_pattern(self,infilepath):
        memoryfile = infilepath+self.dataName+"_memory_pattern.csv"
        memorydf = pd.read_csv(memoryfile)
        memorydf.set_index('Service', inplace=True)
        memory_dict = dict(zip(memorydf.index,zip(memorydf['Common_Memory'], memorydf['Min_Memory'], memorydf['Max_Memory'])))

        cpufile = infilepath+self.dataName+"_cpu_pattern.csv"
        cpudf = pd.read_csv(cpufile)
        cpudf.set_index('Service', inplace=True)
        cpu_dict = dict(zip(cpudf.index,zip(cpudf['Common_CPU'], cpudf['Min_CPU'], cpudf['Max_CPU'])))

        networkfile = infilepath+self.dataName+"_network_pattern.csv"
        networkdf = pd.read_csv(networkfile)
        networkdf.set_index(['Service1', 'Service2'], inplace=True)
        network_dict = dict(zip(networkdf.index, zip(networkdf['Common_Network'], networkdf['Min_Network'], networkdf['Max_Network'])))

        return cpu_dict, memory_dict, network_dict


    def read_event_do_library(self,infile):
        events_dict = {}
        all_num = 0
        with open(infile, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for l in lines:
                l = l.split(" -> ")
                event_1 = l[0]
                event_2 = l[1].split("        : ")[0]
                fre = float(l[1].split("        : ")[1])
                events_dict[(event_1, event_2)] = fre
                all_num += fre
        f.close()

        event_do_library = {}
        for events, fre in events_dict.items():
            event_1 = int(events[0])
            event_2 = int(events[1])
            prob = fre / all_num

            if event_1 not in event_do_library:
                event_do_library[event_1] = []
            event_do_library[event_1].append((event_2, prob))

        return event_do_library

    def read_log_template_vector(self,infile):
        template_vector_dict={}
        with open(infile, encoding='utf-8-sig') as f:
            for line in f:
                line = line.replace("\n", "").strip().split()
                id = int(line[0])
                templateVector = line[1:]
                template_vector_dict[id] = list(map(float,templateVector))
        f.close()

        return template_vector_dict


    def TT_data_index_generate(self):
        normal_index = range(1736, 20136)
        anomaly_index = range(0,1736)

        train_normal_index = random.sample(normal_index, round(0.8 * len(normal_index)))
        normal_leave_index = set(normal_index) - set(train_normal_index)
        val_normal_index = random.sample(list(normal_leave_index), round(0.5 * len(normal_leave_index)))
        test_normal_index = list(set(normal_index) - set(train_normal_index) - set(val_normal_index))

        val_anomaly_index = random.sample(anomaly_index, round(0.3 * len(anomaly_index)))
        test_anomaly_index = list(set(anomaly_index)-set(val_anomaly_index))

        train_index = train_normal_index
        val_index = val_normal_index + val_anomaly_index
        test_index = test_normal_index + test_anomaly_index

        self.logger.info("Train Normal dataset: {};  Train Anomaly dataset: {}".format(len(train_index), 0))
        self.logger.info("Val Normal dataset: {};  Val Anomaly dataset: {}".format(len(val_normal_index), len(val_anomaly_index)))
        self.logger.info("Test Normal dataset: {};  Test Anomaly dataset: {}".format(len(test_normal_index), len(test_anomaly_index)))

        return train_index, val_index, test_index, val_anomaly_index, test_anomaly_index

    def Other_data_index_generate(self):
        normal_index = []
        anomaly_index = []
        with open("./OBD/SN/raw/normal_index.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n", "")
                normal_index.append(int(line))
        f.close()

        with open("./OBD/SN/raw/abnormal_index.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n", "")
                anomaly_index.append(int(line))
        f.close()

        train_normal_index = random.sample(normal_index, round(0.8 * len(normal_index)))
        normal_leave_index = set(normal_index) - set(train_normal_index)
        val_normal_index = random.sample(list(normal_leave_index), round(0.5 * len(normal_leave_index)))
        test_normal_index = list(set(normal_index) - set(train_normal_index) - set(val_normal_index))

        val_anomaly_index = random.sample(anomaly_index, round(0.3 * len(anomaly_index)))
        test_anomaly_index = list(set(anomaly_index)-set(val_anomaly_index))

        train_index = train_normal_index
        val_index = val_normal_index + val_anomaly_index
        test_index = test_normal_index + test_anomaly_index

        self.logger.info("Train Normal dataset: {};  Train Anomaly dataset: {}".format(len(train_index), 0))
        self.logger.info("Val Normal dataset: {};  Val Anomaly dataset: {}".format(len(val_normal_index), len(val_anomaly_index)))
        self.logger.info("Test Normal dataset: {};  Test Anomaly dataset: {}".format(len(test_normal_index), len(test_anomaly_index)))

        return train_index, val_index, test_index, val_anomaly_index, test_anomaly_index


    def compute_batch_avg_service_event(self, data):
        batch = data.batch
        pre_service = data.pre_node_service
        post_service = data.post_node_service
        node_service = torch.cat([pre_service.unsqueeze(1), post_service.unsqueeze(1)], dim=1)

        batch_ids = torch.unique(batch)
        n_samples = len(batch_ids)

        batch_matrix = batch.unsqueeze(1) == batch_ids.unsqueeze(0)
        nodes_per_sample = batch_matrix.sum(dim=0)
        avg_events = nodes_per_sample.float().mean().item()

        services_per_sample = []
        for i, b in enumerate(batch_ids):
            mask = batch == b
            unique_services = torch.unique(node_service[mask]).numel()
            services_per_sample.append(unique_services)

        avg_services = sum(services_per_sample) / n_samples

        return avg_services, avg_events

    def train(self, net):
        net = net.to(self.device)
        self.logger.info('Initializing train dataset and eval dataset...')
        train_dataset= self.dataset[self.train_index]
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8)
        self.logger.info('Train dataset initialized.')

        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-5)

        self.logger.info('Starting training...')
        train_loss_value = []
        avgservices_training_time = defaultdict(list)
        avgevents_training_time = defaultdict(list)
        start_time = time.time()
        for epoch in range(self.n_epochs):
            loss_epoch = 0.0
            n_step=0
            epoch_start_time = time.time()
            net.train()
            for data in train_loader:
                batch_start_time = time.time()
                data=data.to(self.device)
                x, node_type, edge_index, edge_type, batch = \
                    data.x, data.node_type, data.edge_index, data.edge_type, data.batch

                optimizer.zero_grad()
                h_, x_hat_g, mean_g, log_std_g = net(x, node_type, edge_index, edge_type, batch)
                loss = self.gvae_loss_cal(x, x_hat_g, mean_g, log_std_g)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_step+=1

                batch_end_time = time.time()
                batch_training_time = batch_end_time - batch_start_time
                avg_services, avg_events = self.compute_batch_avg_service_event(data)
                avgservices_training_time[round(avg_services, 1)].append(batch_training_time)
                avgevents_training_time[round(avg_events, 1)].append(batch_training_time)

            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            self.logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.10f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_step))
            train_loss_value.append(loss_epoch / n_step)
            torch.save(net.state_dict(), self.model_path + "/best_network.pth")

        train_time = time.time() - start_time
        self.logger.info('Training time: %.3f' % train_time)
        self.logger.info('Finished training.')

        service_train_timedf = pd.DataFrame(
            [{'Avg_Services': k, 'Batch_Training_Time': sum(v) / len(v)}
             for k, v in avgservices_training_time.items()]
        )
        service_train_timedf.to_csv(self.model_path+'avg_services_batch_training_time.csv', index=False)
        event_train_timedf = pd.DataFrame(
            [{'Avg_Events': k, 'Batch_Training_Time': sum(v) / len(v)}
             for k, v in avgevents_training_time.items()]
        )
        event_train_timedf.to_csv(self.model_path+'avg_events_batch_training_time.csv', index=False)

        x = [i for i in range(0, epoch+1)]
        plt.title('Loss vs. epoches')
        plt.ylabel('Loss')
        plt.xlabel("Epoches")
        plt.plot(x, train_loss_value, marker='o', color='y', markersize=5)
        plt.savefig(self.model_path+'Train.png', dpi=120)
        plt.show()

        return net


    def train_GMM(self,net,FPR):
        net = net.to(self.device)
        train_dataset = self.dataset[self.train_index]
        train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=False,num_workers=8)
        train_normal_embeddings = []
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                data = data.to(self.device)
                x, node_type, edge_index, edge_type, batch = \
                    data.x, data.node_type, data.edge_index, data.edge_type, data.batch

                h_,x_hat_g, mean_g, log_std_g = net(x, node_type, edge_index, edge_type, batch)
                train_normal_embeddings.append(h_)

        train_normal_embeddings = torch.cat(train_normal_embeddings, dim=0).cpu().detach().numpy()
        self.normal_pattern = self.GaussianMixture_model(train_normal_embeddings)
        log_probs = self.normal_pattern.score_samples(train_normal_embeddings)
        self.threshold = np.percentile(log_probs, FPR)
        pattern_train_time = time.time() - start_time
        self.logger.info('Pattern Train Time: %.3f' % pattern_train_time)
        return self.normal_pattern,self.threshold


    def validate(self, net, normal_pattern, threshold):
        net = net.to(self.device)
        self.normal_pattern = normal_pattern
        self.threshold = threshold
        val_dataset = self.dataset[self.val_index]
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        val_locate_dataset = self.dataset[self.val_anomaly_index]
        val_locate_loader = DataLoader(val_locate_dataset,batch_size=1,shuffle=False,num_workers=8)

        self.logger.info('Starting validating...')
        val_embeddings = []
        val_labels = []
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                x, node_type, edge_index, edge_type, batch = \
                    data.x, data.node_type, data.edge_index, data.edge_type, data.batch
                h_, x_hat_g, mean_g, log_std_g = net(x, node_type, edge_index, edge_type, batch)
                val_embeddings.append(h_)
                val_labels.append(data.y)

        val_embeddings = torch.cat(val_embeddings, dim=0).cpu().detach().numpy()
        normal_log_likelihoods = self.normal_pattern.score_samples(val_embeddings)
        preds = np.where(normal_log_likelihoods > self.threshold, 0, 1)
        val_labels = torch.cat(val_labels, dim=0)
        val_labels_NF = val_labels[:, 0].squeeze().cpu().detach().numpy()

        combined_array = np.column_stack((val_labels_NF, normal_log_likelihoods))
        df = pd.DataFrame(combined_array, columns=['isAnomaly', 'LogLike'])
        df.to_csv(self.model_path+'validate_output.csv', index=False)

        f1 = f1_score(val_labels_NF, preds, average='binary')
        accuracy = accuracy_score(val_labels_NF, preds)
        precision = precision_score(val_labels_NF, preds, average='binary')
        recall = recall_score(val_labels_NF, preds, average='binary')
        self.logger.info('Anomaly Detection Accuracy: ' + str(accuracy))
        self.logger.info('Anomaly Detection Precision: ' + str(precision))
        self.logger.info('Anomaly Detection Recall: ' + str(recall))
        self.logger.info('Anomaly Detection F1: ' + str(f1))


        self.logger.info('Starting validate anomaly locating...')
        fault_number = 0
        service_top_list = []
        inner_service_top_list = []
        with torch.no_grad():
            for data in val_locate_loader:
                data = data.to(self.device)
                anomaly_node = data.node_is_root_cause != -1
                inject_service_list = []
                root_cause_list = []
                for i in range(anomaly_node.shape[0]):
                    if anomaly_node[i].item() == True:
                        if self.dataName == 'SN':
                            root_cause = self.nodeType[data.node_type[i].item()]
                            inject_service = self.servicelist[data.node_is_root_cause[i].item()]
                            if root_cause not in root_cause_list:
                                inject_service_list.append(inject_service)
                                root_cause_list.append(root_cause)
                        else:
                            if data.node_type[i].item() in [2, 3, 4]:
                                root_cause = self.nodeType[data.node_type[i].item()]
                                inject_service = self.servicelist[data.node_is_root_cause[i].item()]
                                if root_cause not in root_cause_list:
                                    inject_service_list.append(inject_service)
                                    root_cause_list.append(root_cause)
                            else:
                                root_cause = self.nodeType[data.node_type[i].item()]+"_"+str(data.node_event[i].item())
                                inject_service = self.servicelist[data.node_is_root_cause[i].item()]
                                inject_service_list.append(inject_service)
                                root_cause_list.append(root_cause)

                self.logger.info("Inject Ground Truth: %s %s", inject_service_list[0], root_cause_list[0])

                result_list = net.root_cause_locate(data, self.normal_pattern, self.event_do_library,
                                                    self.template_vector,
                                                    self.servicelist, self.nodeType, self.cpu_pattern,
                                                    self.memory_pattern,
                                                    self.network_pattern)


                topk = 1
                for i in range(len(result_list)):
                    if inject_service_list[0] in result_list[i]["pod"]:
                        service_top_list.append(topk)
                        self.logger.info("Locating Results: %s", result_list[i]["pod"])
                        break
                    else:
                        topk = topk + 1
                self.logger.info("Service Position: %s", topk)

                topk = 1
                for i in range(len(result_list)):
                    if inject_service_list[0] in result_list[i]["pod"] and root_cause_list[0] in result_list[i][
                        "resource"]:
                        inner_service_top_list.append(topk)
                        self.logger.info("Locating Results: %s, %s", result_list[i]["pod"],
                                         result_list[i]["resource"])
                        break
                    else:
                        topk = topk + 1
                self.logger.info("Inner-Service Position: %s", topk)
                self.logger.info("----------------------------------------------------")

                fault_number += 1


        top5 = 0
        top4 = 0
        top3 = 0
        top2 = 0
        top1 = 0
        for num in service_top_list:
            if num <= 5:
                top5 += 1
            if num <= 4:
                top4 += 1
            if num <= 3:
                top3 += 1
            if num <= 2:
                top2 += 1
            if num == 1:
                top1 += 1
        AS1 = top1 / fault_number * 100
        AS2 = top2 / fault_number * 100
        AS3 = top3 / fault_number * 100
        AS4 = top4 / fault_number * 100
        AS5 = top5 / fault_number * 100
        AVGS5 = (AS1 + AS2 + AS3 + AS4 + AS5) / 5
        self.logger.info('-------- Service Fault numbuer : %s-------', fault_number)
        self.logger.info('--------AS@1 Result-------')
        self.logger.info("%f %%" % AS1)
        self.logger.info('--------AS@2 Result-------')
        self.logger.info("%f %%" % AS2)
        self.logger.info('--------AS@3 Result-------')
        self.logger.info("%f %%" % AS3)
        self.logger.info('--------AS@4 Result-------')
        self.logger.info("%f %%" % AS4)
        self.logger.info('--------AS@5 Result-------')
        self.logger.info("%f %%" % AS5)
        self.logger.info('--------AVGS@5 Result-------')
        self.logger.info("%f %%" % AVGS5)

        top5 = 0
        top4 = 0
        top3 = 0
        top2 = 0
        top1 = 0
        for num in inner_service_top_list:
            if num <= 5:
                top5 += 1
            if num <= 4:
                top4 += 1
            if num <= 3:
                top3 += 1
            if num <= 2:
                top2 += 1
            if num == 1:
                top1 += 1
        AIS1 = top1 / fault_number * 100
        AIS2 = top2 / fault_number * 100
        AIS3 = top3 / fault_number * 100
        AIS4 = top4 / fault_number * 100
        AIS5 = top5 / fault_number * 100
        AVGIS5 = (AIS1 + AIS2 + AIS3 + AIS4 + AIS5) / 5
        self.logger.info('-------- Inner-Service Fault numbuer : %s-------', fault_number)
        self.logger.info('--------AIS@1 Result-------')
        self.logger.info("%f %%" % AIS1)
        self.logger.info('--------AIS@2 Result-------')
        self.logger.info("%f %%" % AIS2)
        self.logger.info('--------AIS@3 Result-------')
        self.logger.info("%f %%" % AIS3)
        self.logger.info('--------AIS@4 Result-------')
        self.logger.info("%f %%" % AIS4)
        self.logger.info('--------AIS@5 Result-------')
        self.logger.info("%f %%" % AIS5)
        self.logger.info('--------AVGIS@5 Result-------')
        self.logger.info("%f %%" % AVGIS5)
        self.logger.info('Finished validating.')


    def test(self, net, normal_pattern, threshold):
        net = net.to(self.device)
        self.normal_pattern = normal_pattern
        self.threshold = threshold
        test_dataset = self.dataset[self.test_index]
        test_loader = DataLoader(test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=8)
        locate_dataset = self.dataset[self.test_anomaly_index]
        locate_loader = DataLoader(locate_dataset,batch_size=1,shuffle=False,num_workers=8)

        self.logger.info('Starting testing...')
        test_embeddings = []
        test_labels = []
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                x, node_type, edge_index, edge_type, batch = \
                    data.x, data.node_type, data.edge_index, data.edge_type, data.batch

                h_,x_hat_g, mean_g, log_std_g = net(x, node_type, edge_index, edge_type, batch)

                test_embeddings.append(h_)
                test_labels.append(data.y)

        test_embeddings = torch.cat(test_embeddings, dim=0).cpu().detach().numpy()
        normal_log_likelihoods = self.normal_pattern.score_samples(test_embeddings)
        preds = np.where(normal_log_likelihoods > self.threshold, 0, 1)

        self.test_time = time.time() - start_time
        self.logger.info('Anomaly Detecting time: %.3f' % self.test_time)

        test_labels = torch.cat(test_labels, dim=0)
        test_labels_NF = test_labels[:, 0].squeeze().cpu().detach().numpy()


        combined_array = np.column_stack((test_labels_NF, normal_log_likelihoods))
        df = pd.DataFrame(combined_array, columns=['isAnomaly', 'LogLike'])
        df.to_csv(self.model_path+'test_output.csv', index=False)


        f1 = f1_score(test_labels_NF, preds, average='binary')
        accuracy = accuracy_score(test_labels_NF, preds)
        precision = precision_score(test_labels_NF, preds, average='binary')
        recall = recall_score(test_labels_NF, preds, average='binary')
        self.logger.info('Anomaly Detection Accuracy: ' + str(accuracy))
        self.logger.info('Anomaly Detection Precision: ' + str(precision))
        self.logger.info('Anomaly Detection Recall: ' + str(recall))
        self.logger.info('Anomaly Detection F1: ' + str(f1))

        self.logger.info('Starting test anomaly locating...')
        fault_number = 0
        service_top_list = []
        inner_service_top_list = []
        avgservices_inference_time = defaultdict(list)
        avgevents_inference_time = defaultdict(list)
        each_inference_time =[]
        with torch.no_grad():
            for data in locate_loader:
                data = data.to(self.device)
                anomaly_node = data.node_is_root_cause != -1
                inject_service_list = []
                root_cause_list = []
                for i in range(anomaly_node.shape[0]):
                    if anomaly_node[i].item() == True:
                        if self.dataName == 'SN':
                            root_cause = self.nodeType[data.node_type[i].item()]
                            inject_service = self.servicelist[data.node_is_root_cause[i].item()]
                            if root_cause not in root_cause_list:
                                inject_service_list.append(inject_service)
                                root_cause_list.append(root_cause)
                        else:
                            if data.node_type[i].item() in [2, 3, 4]:
                                root_cause = self.nodeType[data.node_type[i].item()]
                                inject_service = self.servicelist[data.node_is_root_cause[i].item()]
                                if root_cause not in root_cause_list:
                                    inject_service_list.append(inject_service)
                                    root_cause_list.append(root_cause)
                            else:
                                root_cause = self.nodeType[data.node_type[i].item()]+"_"+str(data.node_event[i].item())
                                inject_service = self.servicelist[data.node_is_root_cause[i].item()]
                                inject_service_list.append(inject_service)
                                root_cause_list.append(root_cause)

                self.logger.info("Inject Ground Truth: %s %s", inject_service_list[0], root_cause_list[0])

                avg_services, avg_events = self.compute_batch_avg_service_event(data)
                start_time = time.time()
                result_list = net.root_cause_locate(data, self.normal_pattern, self.event_do_library,
                                                    self.template_vector,
                                                    self.servicelist, self.nodeType, self.cpu_pattern,
                                                    self.memory_pattern,
                                                    self.network_pattern)
                end_time = time.time()
                inference_time = end_time-start_time
                avgservices_inference_time[avg_services].append(inference_time)
                avgevents_inference_time[avg_events].append(inference_time)
                each_inference_time.append(inference_time)

                topk = 1
                for i in range(len(result_list)):
                    if inject_service_list[0] in result_list[i]["pod"]:
                        service_top_list.append(topk)
                        self.logger.info("Locating Results: %s", result_list[i]["pod"])
                        break
                    else:
                        topk = topk + 1
                self.logger.info("Service Position: %s", topk)

                topk = 1
                for i in range(len(result_list)):
                    if inject_service_list[0] in result_list[i]["pod"] and root_cause_list[0] in result_list[i][
                        "resource"]:
                        inner_service_top_list.append(topk)
                        self.logger.info("Locating Results: %s, %s", result_list[i]["pod"],
                                         result_list[i]["resource"])
                        break
                    else:
                        topk = topk + 1
                self.logger.info("Inner-Service Position: %s", topk)
                self.logger.info("----------------------------------------------------")

                fault_number += 1


        service_inference_timedf = pd.DataFrame(
            [{'Avg_Services': k, 'Batch_Inference_Time': sum(v) / len(v)}
             for k, v in avgservices_inference_time.items()]
        )
        service_inference_timedf.to_csv(self.model_path+'avg_services_inference_time.csv', index=False)
        event_inference_timedf = pd.DataFrame(
            [{'Avg_Events': k, 'Batch_Inference_Time': sum(v) / len(v)}
             for k, v in avgevents_inference_time.items()]
        )
        event_inference_timedf.to_csv(self.model_path+'avg_events_inference_time.csv', index=False)


        top5 = 0
        top4 = 0
        top3 = 0
        top2 = 0
        top1 = 0
        for num in service_top_list:
            if num <= 5:
                top5 += 1
            if num <= 4:
                top4 += 1
            if num <= 3:
                top3 += 1
            if num <= 2:
                top2 += 1
            if num == 1:
                top1 += 1
        AS1 = top1 / fault_number * 100
        AS2 = top2 / fault_number * 100
        AS3 = top3 / fault_number * 100
        AS4 = top4 / fault_number * 100
        AS5 = top5 / fault_number * 100
        AVGS5 = (AS1 + AS2 + AS3 + AS4 + AS5) / 5
        self.logger.info('-------- Service Fault numbuer : %s-------', fault_number)
        self.logger.info('--------AS@1 Result-------')
        self.logger.info("%f %%" % AS1)
        self.logger.info('--------AS@2 Result-------')
        self.logger.info("%f %%" % AS2)
        self.logger.info('--------AS@3 Result-------')
        self.logger.info("%f %%" % AS3)
        self.logger.info('--------AS@4 Result-------')
        self.logger.info("%f %%" % AS4)
        self.logger.info('--------AS@5 Result-------')
        self.logger.info("%f %%" % AS5)
        self.logger.info('--------AVGS@5 Result-------')
        self.logger.info("%f %%" % AVGS5)

        top5 = 0
        top4 = 0
        top3 = 0
        top2 = 0
        top1 = 0
        for num in inner_service_top_list:
            if num <= 5:
                top5 += 1
            if num <= 4:
                top4 += 1
            if num <= 3:
                top3 += 1
            if num <= 2:
                top2 += 1
            if num == 1:
                top1 += 1
        AIS1 = top1 / fault_number * 100
        AIS2 = top2 / fault_number * 100
        AIS3 = top3 / fault_number * 100
        AIS4 = top4 / fault_number * 100
        AIS5 = top5 / fault_number * 100
        AVGIS5 = (AIS1 + AIS2 + AIS3 + AIS4 + AIS5) / 5
        self.logger.info('-------- Inner-Service Fault numbuer : %s-------', fault_number)
        self.logger.info('--------AIS@1 Result-------')
        self.logger.info("%f %%" % AIS1)
        self.logger.info('--------AIS@2 Result-------')
        self.logger.info("%f %%" % AIS2)
        self.logger.info('--------AIS@3 Result-------')
        self.logger.info("%f %%" % AIS3)
        self.logger.info('--------AIS@4 Result-------')
        self.logger.info("%f %%" % AIS4)
        self.logger.info('--------AIS@5 Result-------')
        self.logger.info("%f %%" % AIS5)
        self.logger.info('--------AVGIS@5 Result-------')
        self.logger.info("%f %%" % AVGIS5)
        self.logger.info('--------Total Location Time-------')
        self.logger.info("%.3f" % sum(each_inference_time))
        self.logger.info('--------Average Location Time-------')
        self.logger.info("%.3f" % (sum(each_inference_time)/fault_number))
        self.logger.info('Finished testing.')


    def gvae_loss_cal(self, x, x_hat, mean, log_std):
        diff = x-x_hat
        sq_diff=torch.square(diff)
        reconstruction_loss = sq_diff.mean()
        kl_divergence = -0.5 * (1 + 2 * log_std - mean ** 2 - torch.exp(2*log_std)).sum(dim=1).mean()
        loss = reconstruction_loss + kl_divergence
        return loss


    def GaussianMixture_model(self,train_embeddings):
        K = train_embeddings.shape[1]
        normal_gmm = GaussianMixture(n_components=K, covariance_type='full', max_iter=100, random_state=42)
        normal_gmm.fit(train_embeddings)
        return normal_gmm




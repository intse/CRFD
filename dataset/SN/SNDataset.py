from torch_geometric.data import Dataset,Data
import numpy as np
import torch
import os
import random
from tqdm import tqdm
import time
from datetime import datetime
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def min_date_to_timestamp(timestr):
    time_str = timestr[0:4] + "-" + timestr[4:6] + "-" + timestr[6:8] + " " + timestr[8:10] + ":" + timestr[10:12] + ":00.00"
    datetime_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return obj_stamp

def second_date_to_timestamp(timestr):
    datetime_obj = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")
    obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return obj_stamp

class SNDataset(Dataset):
    def __init__(self,root,transform=None,pre_transform=None):
        super(SNDataset, self).__init__(root,transform,pre_transform)

    @property
    def raw_file_names(self):
        tracelabelfile='traceLabel.csv'
        graphfile="trace_graph_depth10.txt"
        logfile="log_template_depth10.txt"
        vectortemplatefile="log_template_vector300_depth10.txt"
        SpanTemplateIdfile="SpanTemplateId_depth10.txt"
        service_metricfile="SN_pod_metric_with_servicelatency_normalization.csv"
        service_relation_metricfile="relation_service_kpi_normalization.csv"

        pod_metric_statfile='SN_pod_metric_with_servicelatency_stat.csv'
        relation_statfile='relation_service_kpi_stat.csv'

        return [graphfile,logfile,vectortemplatefile,SpanTemplateIdfile,
                service_metricfile,service_relation_metricfile,tracelabelfile,
                pod_metric_statfile,relation_statfile]

    @property
    def processed_file_names(self):
        return ['data_0_depth10.pt']

    def download(self):
        pass

    def process(self):
        edge_type_dict = {'Event_to_Event': 0, 'Event_to_CPU': 1, 'Event_to_Memory': 2,
                          'CPU_to_Event': 3, 'Memory_to_Event': 4, 'Network_to_Event':5,'Event_to_Network':6}

        ServiceMetric_Means={}
        ServiceMetric_Stds={}

        network_df=pd.read_csv(self.raw_paths[8])
        network_means = network_df.set_index('Column')['Mean'].to_dict()
        network_stds = network_df.set_index('Column')['Std'].to_dict()
        ServiceMetric_Means['network']=network_means['NL']
        ServiceMetric_Stds['network']=network_stds['NL']

        cpu_memory_df = pd.read_csv(self.raw_paths[7])
        cpu_memory_means = cpu_memory_df.set_index('Column')['Mean'].to_dict()
        cpu_memory_stds = cpu_memory_df.set_index('Column')['Std'].to_dict()
        for metric,mean in cpu_memory_means.items():
            ServiceMetric_Means[metric]=cpu_memory_means[metric]
            ServiceMetric_Stds[metric]=cpu_memory_stds[metric]

        trace_anomaly={}
        with open(self.raw_paths[6], "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "").strip().split(",")
                if line[0] == "TraceID":
                    continue
                traceid = line[0]
                anomaly_type = line[2]
                trace_anomaly[traceid]=anomaly_type
        f.close()

        service_metric={}
        df = pd.read_csv(self.raw_paths[4], index_col=0)
        for column in df.columns:
            if column.find("_networklatency") >= 0:
                continue
            for index in df.index:
                value = df.at[index, column]
                service_metric[(column, index)] = value

        service_relation_metric={}
        with open(self.raw_paths[5], "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "").strip().split(",")
                if line[0] == "Service1":
                    continue
                service_name1 = line[0]
                service_name2 = line[1]
                timestamp = min_date_to_timestamp(line[2])
                networklatency = float(line[9])
                service_relation_metric[(service_name1, service_name2, timestamp)] = networklatency
        f.close()

        template_vector_dict = {}
        SpanTemplateId=set()
        with open(self.raw_paths[3], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip()
                SpanTemplateId.add(line)
        f.close()

        with open(self.raw_paths[2], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip().split()
                id = line[0]
                templateVector = line[1:]
                template_vector_dict[id] = list(map(float,templateVector))
        f.close()

        logVector={}
        logType={}
        logServiceName={}
        logTimeStamp = {}
        logSpanId={}
        logIsRootCause={}
        logTemplateId = {}
        ServiceSet = set()
        index = 0
        with open(self.raw_paths[1], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().replace("\n", "").split(" - ")
                content = line[0].split("[SW_CTX:[")
                timestamp_str = content[0][:23]
                timestamp=second_date_to_timestamp(timestamp_str)
                service_name = content[1].split(",")[0].strip()
                spanid = content[1].split(",")[3].split("]")[0].strip()
                template_id = line[1]

                logId = str(index)
                logVector[logId] = template_vector_dict[template_id]
                logTimeStamp[logId] = timestamp
                logServiceName[logId] = service_name
                logTemplateId[logId] = int(template_id)
                ServiceSet.add(service_name)
                logSpanId[logId] = spanid
                logIsRootCause[logId]= -1

                if template_id in SpanTemplateId:
                    logType[logId] = "SpanEvent"
                else:
                    logType[logId] = "LogEvent"
                index += 1
        f.close()

        ServiceList=list(ServiceSet)
        print(ServiceList)


        idx=0
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                SpanEventNode = set()
                LogEventNode = set()
                CPUNode = set()
                MemoryNode = set()
                NetworkNode = set()

                line = line.strip().replace("\n", "").replace("\'", "").split("           ")
                trace_id = line[0]
                anomaly=trace_anomaly[trace_id]
                if anomaly=='normal':
                    anomaly_service= "null"
                    anomaly_metric="null"
                else:
                    anomaly_service= anomaly.split("_")[0]
                    anomaly_metric=anomaly.split("_")[1]

                edges = line[1][2:-2].split("), (")
                classlabel = int(line[2])
                isAnomaly = int(line[3])

                for edge in edges:
                    edge = edge.split(", ")
                    node1 = edge[0]
                    node2 = edge[1]
                    if logType[node1] == 'SpanEvent':
                        SpanEventNode.add(node1)
                    else:
                        LogEventNode.add(node1)

                    if logType[node2] == 'SpanEvent':
                        SpanEventNode.add(node2)
                    else:
                        LogEventNode.add(node2)

                    node1_cpu = node1 + "_cpu"
                    node1_memory = node1 + "_memory"
                    node2_cpu = node2 + "_cpu"
                    node2_memory = node2 + "_memory"
                    CPUNode.add(node1_cpu)
                    CPUNode.add(node2_cpu)
                    MemoryNode.add(node1_memory)
                    MemoryNode.add(node2_memory)

                    if logType[node1] == 'SpanEvent' and logType[node2] == 'SpanEvent' and logServiceName[node1] != \
                            logServiceName[node2]:
                        node1_node2_network = node1 + "_" + node2 + "_network"
                        NetworkNode.add(node1_node2_network)

                NodeX = []
                NodeType = []
                Pre_NodeService = []
                Post_NodeService = []
                Node_isRootCause = []
                Node_Mean = []
                Node_Std = []
                Node_Template = []
                for logId in LogEventNode:
                    NodeX.append(logVector[logId]+[0]*3)
                    Pre_NodeService.append(ServiceList.index(logServiceName[logId]))
                    Post_NodeService.append(ServiceList.index(logServiceName[logId]))
                    Node_isRootCause.append(logIsRootCause[logId])
                    NodeType.append(0)
                    Node_Mean.append(-1)
                    Node_Std.append(-1)
                    Node_Template.append(logTemplateId[logId])

                for logId in SpanEventNode:
                    NodeX.append(logVector[logId]+[0]*3)
                    Pre_NodeService.append(ServiceList.index(logServiceName[logId]))
                    Post_NodeService.append(ServiceList.index(logServiceName[logId]))
                    Node_isRootCause.append(logIsRootCause[logId])
                    NodeType.append(1)
                    Node_Mean.append(-1)
                    Node_Std.append(-1)
                    Node_Template.append(logTemplateId[logId])

                for nodeId in CPUNode:
                    logId = nodeId.split("_cpu")[0]
                    logId_timestamp = 1000 * int(logTimeStamp[logId] / 1000)
                    if (logServiceName[logId] + "_cpu", logId_timestamp) in service_metric:
                        cpu_value = service_metric[(logServiceName[logId] + "_cpu", logId_timestamp)]
                    else:
                        cpu_value = -1

                    NodeX.append([0]*300+[cpu_value]+[0]*2)
                    Pre_NodeService.append(ServiceList.index(logServiceName[logId]))
                    Post_NodeService.append(ServiceList.index(logServiceName[logId]))
                    if logServiceName[logId] == anomaly_service and anomaly_metric=='cpu':
                        Node_isRootCause.append(ServiceList.index(logServiceName[logId]))
                    else:
                        Node_isRootCause.append(-1)
                    NodeType.append(2)
                    Node_Mean.append(ServiceMetric_Means[logServiceName[logId] + "_cpu"])
                    Node_Std.append(ServiceMetric_Stds[logServiceName[logId] + "_cpu"])
                    Node_Template.append(-1)

                for nodeId in MemoryNode:
                    logId = nodeId.split("_memory")[0]
                    logId_timestamp = 1000 * int(logTimeStamp[logId] / 1000)
                    if (logServiceName[logId] + "_memory", logId_timestamp) in service_metric:
                        memory_value = service_metric[(logServiceName[logId] + "_memory", logId_timestamp)]
                    else:
                        memory_value = -1
                    NodeX.append([0]*301+[memory_value] + [0])
                    Pre_NodeService.append(ServiceList.index(logServiceName[logId]))
                    Post_NodeService.append(ServiceList.index(logServiceName[logId]))
                    Node_isRootCause.append(-1)
                    NodeType.append(3)
                    Node_Mean.append(ServiceMetric_Means[logServiceName[logId] + "_memory"])
                    Node_Std.append(ServiceMetric_Stds[logServiceName[logId] + "_memory"])
                    Node_Template.append(-1)

                for nodeId in NetworkNode:
                    logId1 = nodeId.split("_")[0]
                    logId2 = nodeId.split("_")[1]
                    logId1_timestamp = 60000 * int(logTimeStamp[logId1] / 60000)
                    if (logServiceName[logId1], logServiceName[logId2], logId1_timestamp) in service_relation_metric:
                        network_value = service_relation_metric[
                            (logServiceName[logId1], logServiceName[logId2], logId1_timestamp)]
                        NodeX.append([0]*302 + [network_value])
                    elif (logServiceName[logId2], logServiceName[logId1], logId1_timestamp) in service_relation_metric:
                        network_value = service_relation_metric[
                            (logServiceName[logId2], logServiceName[logId1], logId1_timestamp)]
                        NodeX.append([0]*302 + [network_value])
                    else:
                        network_value = -1
                        NodeX.append([0]*302 + [network_value])
                    Pre_NodeService.append(ServiceList.index(logServiceName[logId1]))
                    Post_NodeService.append(ServiceList.index(logServiceName[logId2]))
                    if logServiceName[logId1] == anomaly_service and logServiceName[logId2] == anomaly_service \
                            and anomaly_metric=='network':
                        Node_isRootCause.append(ServiceList.index(logServiceName[logId1]))
                    elif logServiceName[logId1] == anomaly_service and anomaly_metric=='network':
                        Node_isRootCause.append(ServiceList.index(logServiceName[logId1]))
                    elif logServiceName[logId2] == anomaly_service and anomaly_metric=='network':
                        Node_isRootCause.append(ServiceList.index(logServiceName[logId2]))
                    else:
                        Node_isRootCause.append(-1)
                    NodeType.append(4)
                    Node_Mean.append(ServiceMetric_Means["network"])
                    Node_Std.append(ServiceMetric_Stds["network"])
                    Node_Template.append(-1)

                LogEventNode = list(LogEventNode)
                SpanEventNode = list(SpanEventNode)
                CPUNode = list(CPUNode)
                MemoryNode = list(MemoryNode)
                NetworkNode = list(NetworkNode)
                AllNode = LogEventNode + SpanEventNode + CPUNode + MemoryNode + NetworkNode

                Pre_edge_index = []
                Tar_edge_index = []
                EdgeType = []
                for edge in edges:
                    edge = edge.split(", ")
                    node1 = edge[0]
                    node2 = edge[1]

                    node1_cpu = node1 + "_cpu"
                    node1_memory = node1 + "_memory"

                    node1_Id = AllNode.index(node1)
                    node2_Id = AllNode.index(node2)
                    node1_cpu_Id = AllNode.index(node1_cpu)
                    node1_memory_Id = AllNode.index(node1_memory)

                    Pre_edge_index.append(node1_Id)
                    Tar_edge_index.append(node2_Id)
                    EdgeType.append(edge_type_dict['Event_to_Event'])

                    Pre_edge_index.append(node1_Id)
                    Tar_edge_index.append(node1_cpu_Id)
                    EdgeType.append(edge_type_dict['Event_to_CPU'])
                    Pre_edge_index.append(node1_cpu_Id)
                    Tar_edge_index.append(node2_Id)
                    EdgeType.append(edge_type_dict['CPU_to_Event'])

                    Pre_edge_index.append(node1_Id)
                    Tar_edge_index.append(node1_memory_Id)
                    EdgeType.append(edge_type_dict['Event_to_Memory'])
                    Pre_edge_index.append(node1_memory_Id)
                    Tar_edge_index.append(node2_Id)
                    EdgeType.append(edge_type_dict['Memory_to_Event'])

                    if logType[node1] == 'SpanEvent' and logType[node2] == 'SpanEvent' and logServiceName[node1] != \
                            logServiceName[node2]:
                        node1_node2_network = node1 + "_" + node2 + "_network"
                        network_node_Id = AllNode.index(node1_node2_network)
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(network_node_Id)
                        EdgeType.append(edge_type_dict['Event_to_Network'])
                        Pre_edge_index.append(network_node_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['Event_to_Network'])

                labels = []
                labels.append([isAnomaly, classlabel])

                edge_index = torch.as_tensor([Pre_edge_index, Tar_edge_index], dtype=torch.long)
                X = torch.as_tensor(NodeX, dtype=torch.float32)
                Pre_NodeServiceX = torch.as_tensor(Pre_NodeService, dtype=torch.long)
                Post_NodeServiceX = torch.as_tensor(Post_NodeService, dtype=torch.long)
                Node_isRootCauseX = torch.as_tensor(Node_isRootCause, dtype=torch.long)
                NodeType = torch.as_tensor(NodeType, dtype=torch.long)
                EdgeType = torch.as_tensor(EdgeType, dtype=torch.long)
                NodeEvent = torch.as_tensor(Node_Template, dtype=torch.long)
                NodeMean = torch.as_tensor(Node_Mean, dtype=torch.float32)
                NodeStd = torch.as_tensor(Node_Std, dtype=torch.float32)
                y = torch.as_tensor(labels, dtype=torch.long)

                data = Data(x=X, pre_node_service=Pre_NodeServiceX, post_node_service=Post_NodeServiceX,
                            node_is_root_cause=Node_isRootCauseX, node_mean=NodeMean,node_std=NodeStd,
                            node_event=NodeEvent, node_type=NodeType, edge_type=EdgeType,
                            edge_index=edge_index, y=y)
                torch.save(data, os.path.join(self.processed_dir, 'data_{}_depth10.pt'.format(idx)))
                idx += 1
        f.close()

    def len(self) -> int:
        datalen=0
        basedir="./OBD/SN/processed"
        for file in os.listdir(basedir):
            datalen+=1
        return datalen-2

    def get(self,idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}_depth10.pt'))
        return data






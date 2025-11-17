from torch_geometric.data import Dataset,Data
import numpy as np
import torch
import os
import random
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def date_to_timestamp(timestr):
    datetime_obj = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")
    obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return obj_stamp



class TTDataset(Dataset):
    def __init__(self,root,transform=None,pre_transform=None):
        super(TTDataset, self).__init__(root,transform,pre_transform)

    @property
    def raw_file_names(self):
        graphfile1="F1_trace_graph_depth10.txt"
        graphfile2 = "F2_trace_graph_depth10.txt"
        graphfile3 = "F3_trace_graph_depth10.txt"
        graphfilebig = "Normalbig_trace_graph_depth10.txt"

        logfile1="F1_log_template_depth10.txt"
        logfile2 = "F2_log_template_depth10.txt"
        logfile3 = "F3_log_template_depth10.txt"
        logfilebig = "Normalbig_log_template_depth10.txt"

        F1_service_metricfile="F1_pod_metric_with_servicelatency_normalization.csv"
        F1_service_relation_metricfile="F1_relation_service_kpi_normalization.csv"

        F2_service_metricfile="F2_pod_metric_with_servicelatency_normalization.csv"
        F2_service_relation_metricfile="F2_relation_service_kpi_normalization.csv"

        F3_service_metricfile="F3_pod_metric_with_servicelatency_normalization.csv"
        F3_service_relation_metricfile="F3_relation_service_kpi_normalization.csv"

        Normalbig_service_metricfile="normal_pod_metric_with_servicelatency_normalization.csv"
        Normalbig_service_relation_metricfile="normal_relation_service_kpi_normalization.csv"

        vectortemplatefile="log_template_vector300_depth10.txt"
        SpanTemplateIdfile="SpanTemplateId_depth10.txt"
        pod_metric_statfile='pod_metric_with_servicelatency_stat.csv'
        relation_statfile='relation_service_kpi_stat.csv'

        return [graphfile1,graphfile2,graphfile3,graphfilebig,
                logfile1,logfile2,logfile3,logfilebig,
                F1_service_metricfile,F1_service_relation_metricfile,
                F2_service_metricfile,F2_service_relation_metricfile,
                F3_service_metricfile, F3_service_relation_metricfile,
                Normalbig_service_metricfile, Normalbig_service_relation_metricfile,
                vectortemplatefile,SpanTemplateIdfile,pod_metric_statfile,relation_statfile]

    @property
    def processed_file_names(self):
        return ['data_0_depth10.pt']

    def download(self):
        pass

    def process(self):
        edge_type_dict = {'Event_to_Event': 0, 'Event_to_CPU': 1, 'Event_to_Memory': 2,
                          'CPU_to_Event': 3, 'Memory_to_Event': 4, 'Network_to_Event':5,'Event_to_Network':6}

        graphfile_dict ={0:'F1',1:"F2",2:'F3',3:'F4',5:'normal'}
        event_dim = 300

        template_vector_dict = {}
        SpanTemplateId=set()
        ServiceSet=set()
        ServiceMetric_Means={}
        ServiceMetric_Stds={}

        network_df=pd.read_csv(self.raw_paths[19])
        network_means = network_df.set_index('Column')['Mean'].to_dict()
        network_stds = network_df.set_index('Column')['Std'].to_dict()
        ServiceMetric_Means['network']=network_means['NetworkLatency']
        ServiceMetric_Stds['network']=network_stds['NetworkLatency']

        cpu_memory_df = pd.read_csv(self.raw_paths[18])
        cpu_memory_means = cpu_memory_df.set_index('Column')['Mean'].to_dict()
        cpu_memory_stds = cpu_memory_df.set_index('Column')['Std'].to_dict()
        for metric,mean in cpu_memory_means.items():
            ServiceMetric_Means[metric]=cpu_memory_means[metric]
            ServiceMetric_Stds[metric]=cpu_memory_stds[metric]

        with open(self.raw_paths[17], encoding='utf-8-sig') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip()
                SpanTemplateId.add(line)
        f.close()

        with open(self.raw_paths[16], encoding='utf-8-sig') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip().split()
                id = line[0]
                templateVector = line[1:]
                template_vector_dict[id] = list(map(float,templateVector))
        f.close()

        service_metric={}
        service_relation_metric={}
        for i in tqdm(range(4)):
            targetIndex=8+2*i
            df = pd.read_csv(self.raw_paths[targetIndex], index_col=0)
            for column in df.columns:
                if column.find("_networklatency")>=0 or column.find("_servicelatency")>=0 :
                    continue
                for index in df.index:
                    value=df.at[index,column]
                    service_metric[(column,index)]=value

            with open(self.raw_paths[targetIndex+1],"r",encoding='utf-8') as f:
                lines=f.readlines()
                for line in lines:
                    line = line.replace("\n","").strip().split(",")
                    if line[0]=="Service1_ID":
                        continue
                    service_name1=line[0]
                    service_name2=line[1]
                    timestamp=int(line[2])
                    networklatency=float(line[-1])
                    service_relation_metric[(service_name1,service_name2,timestamp)]=networklatency
            f.close()

        logfilelist=self.raw_paths[4:8]
        logVector={}
        logType={}
        logServiceName={}
        logTimeStamp={}
        logSpanId={}
        logIsRootCause={}
        logTemplateId={}
        for i in tqdm(range(len(logfilelist))):
            index=0
            with open(logfilelist[i], encoding='utf-8') as f:
                for line in f:
                    line = line.strip().replace("\n", "").split(" - ")
                    content=line[0].split("       ")[2].split("[SW_CTX:[")
                    timestamp=date_to_timestamp(content[0][:23])
                    service_name=content[1].split(",")[0].strip()
                    spanid = content[1].split(",")[3].strip()+"S"+content[1].split(",")[4].split("]")[0].strip()

                    template_id = line[-2]
                    isRootCause = line[-1]

                    logId= str(i)+":"+str(index)
                    logVector[logId]=template_vector_dict[template_id]
                    logTimeStamp[logId]=timestamp
                    logServiceName[logId]=service_name
                    ServiceSet.add(service_name)
                    logSpanId[logId]=spanid
                    logIsRootCause[logId]=int(isRootCause)
                    logTemplateId[logId]=int(template_id)
                    if template_id in SpanTemplateId:
                        logType[logId]="SpanEvent"
                    else:
                        logType[logId]="LogEvent"
                    index+=1
            f.close()

        ServiceList=list(ServiceSet)
        print(ServiceList)

        graphfilelist=self.raw_paths[:4]
        idx=0
        for i in tqdm(range(len(graphfilelist))):
            with open(graphfilelist[i], 'r', encoding='utf-8') as f:
                for line in f:
                    SpanEventNode=set()
                    LogEventNode=set()
                    CPUNode=set()
                    MemoryNode=set()
                    NetworkNode=set()

                    line = line.strip().replace("\n", "").replace("\'", "").split("           ")
                    edges = line[1][2:-2].split("), (")
                    classlabel = int(line[2])

                    for edge in edges:
                        edge = edge.split(", ")
                        node1 = str(i) + ":" + edge[0]
                        node2 = str(i) + ":" + edge[1]
                        if logType[node1]=='SpanEvent':
                            SpanEventNode.add(node1)
                        else:
                            LogEventNode.add(node1)

                        if logType[node2]=='SpanEvent':
                            SpanEventNode.add(node2)
                        else:
                            LogEventNode.add(node2)

                        node1_cpu=node1+"_cpu"
                        node1_memory = node1 + "_memory"
                        node2_cpu = node2 + "_cpu"
                        node2_memory = node2 + "_memory"
                        CPUNode.add(node1_cpu)
                        CPUNode.add(node2_cpu)
                        MemoryNode.add(node1_memory)
                        MemoryNode.add(node2_memory)

                        if logType[node1]=='SpanEvent' and logType[node2]=='SpanEvent' and logServiceName[node1]!=logServiceName[node2]:
                            node1_node2_network=node1+"_"+ node2+"_network"
                            NetworkNode.add(node1_node2_network)

                    NodeX = []
                    NodeType=[]
                    Pre_NodeService = []
                    Post_NodeService = []
                    Node_isRootCause = []
                    Node_Mean=[]
                    Node_Std=[]
                    Node_Template=[]
                    for logId in LogEventNode:
                        NodeX.append(logVector[logId]+[0]*3)
                        Pre_NodeService.append(ServiceList.index(logServiceName[logId]))
                        Post_NodeService.append(ServiceList.index(logServiceName[logId]))
                        if logIsRootCause[logId]==1:
                            Node_isRootCause.append(ServiceList.index(logServiceName[logId]))
                        else:
                            Node_isRootCause.append(-1)
                        NodeType.append(0)
                        Node_Mean.append(-1)
                        Node_Std.append(-1)
                        Node_Template.append(logTemplateId[logId])

                    for logId in SpanEventNode:
                        NodeX.append(logVector[logId]+[0]*3)
                        Pre_NodeService.append(ServiceList.index(logServiceName[logId]))
                        Post_NodeService.append(ServiceList.index(logServiceName[logId]))
                        if logIsRootCause[logId]==1:
                            Node_isRootCause.append(ServiceList.index(logServiceName[logId]))
                        else:
                            Node_isRootCause.append(-1)
                        NodeType.append(1)
                        Node_Mean.append(-1)
                        Node_Std.append(-1)
                        Node_Template.append(logTemplateId[logId])

                    for nodeId in CPUNode:
                        logId = nodeId.split("_cpu")[0]
                        logId_timestamp = 60000 * int(logTimeStamp[logId] / 60000)
                        if (logServiceName[logId] + "_cpu",logId_timestamp) in service_metric:
                            cpu_value = service_metric[(logServiceName[logId] + "_cpu",logId_timestamp)]
                        else:
                            cpu_value = -1
                        NodeX.append([0]*event_dim+[cpu_value]+[0]*2)
                        Pre_NodeService.append(ServiceList.index(logServiceName[logId]))
                        Post_NodeService.append(ServiceList.index(logServiceName[logId]))
                        if i==2 and logServiceName[logId]=='ts-order-service':
                            Node_isRootCause.append(ServiceList.index(logServiceName[logId]))
                        else:
                            Node_isRootCause.append(-1)
                        NodeType.append(2)
                        Node_Mean.append(ServiceMetric_Means[logServiceName[logId] + "_cpu"])
                        Node_Std.append(ServiceMetric_Stds[logServiceName[logId] + "_cpu"])
                        Node_Template.append(-1)

                    for nodeId in MemoryNode:
                        logId = nodeId.split("_memory")[0]
                        logId_timestamp = 60000 * int(logTimeStamp[logId] / 60000)
                        if (logServiceName[logId] + "_memory",logId_timestamp) in service_metric:
                            memory_value = service_metric[(logServiceName[logId] + "_memory",logId_timestamp)]
                        else:
                            memory_value = -1
                        NodeX.append([0]*(event_dim+1)+[memory_value] + [0])
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
                        logId1_timestamp=60000*int(logTimeStamp[logId1]/60000)
                        if (logServiceName[logId1],logServiceName[logId2],logId1_timestamp) in service_relation_metric:
                            network_value = service_relation_metric[(logServiceName[logId1],logServiceName[logId2],logId1_timestamp)]
                            NodeX.append([0]*(event_dim+2) + [network_value])
                        elif (logServiceName[logId2],logServiceName[logId1],logId1_timestamp) in service_relation_metric:
                            network_value = service_relation_metric[(logServiceName[logId2], logServiceName[logId1], logId1_timestamp)]
                            NodeX.append([0]*(event_dim+2) + [network_value])
                        else:
                            network_value = -1
                            NodeX.append([0]*(event_dim+2) + [network_value])

                        Pre_NodeService.append(ServiceList.index(logServiceName[logId1]))
                        Post_NodeService.append(ServiceList.index(logServiceName[logId2]))
                        if i==1 and (logServiceName[logId1]=='ts-seat-service' or logServiceName[logId2]=='ts-seat-service'):
                            Node_isRootCause.append(ServiceList.index('ts-seat-service'))
                        else:
                            Node_isRootCause.append(-1)

                        NodeType.append(4)
                        Node_Mean.append(ServiceMetric_Means["network"])
                        Node_Std.append(ServiceMetric_Stds["network"])
                        Node_Template.append(-1)

                    LogEventNode = list(LogEventNode)
                    SpanEventNode=list(SpanEventNode)
                    CPUNode=list(CPUNode)
                    MemoryNode=list(MemoryNode)
                    NetworkNode=list(NetworkNode)
                    AllNode = LogEventNode + SpanEventNode + CPUNode + MemoryNode + NetworkNode

                    Pre_edge_index=[]
                    Tar_edge_index = []
                    EdgeType=[]

                    for edge in edges:
                        edge = edge.split(", ")
                        node1 = str(i) + ":" + edge[0]
                        node2 = str(i) + ":" + edge[1]

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

                        if logType[node1]=='SpanEvent' and logType[node2]=='SpanEvent' and logServiceName[node1]!=logServiceName[node2]:
                            node1_node2_network = node1 + "_" + node2 + "_network"
                            network_node_Id = AllNode.index(node1_node2_network)
                            Pre_edge_index.append(node1_Id)
                            Tar_edge_index.append(network_node_Id)
                            EdgeType.append(edge_type_dict['Event_to_Network'])
                            Pre_edge_index.append(network_node_Id)
                            Tar_edge_index.append(node2_Id)
                            EdgeType.append(edge_type_dict['Event_to_Network'])


                    labels=[]
                    if i==len(graphfilelist)-1:
                        NF_label = 0
                    else:
                        NF_label = 1
                    labels.append([NF_label, i, classlabel])

                    edge_index=torch.as_tensor([Pre_edge_index,Tar_edge_index],dtype=torch.long)
                    X=torch.as_tensor(NodeX,dtype=torch.float32)
                    Pre_NodeServiceX=torch.as_tensor(Pre_NodeService,dtype=torch.long)
                    Post_NodeServiceX = torch.as_tensor(Post_NodeService, dtype=torch.long)
                    Node_isRootCauseX=torch.as_tensor(Node_isRootCause,dtype=torch.long)
                    NodeType=torch.as_tensor(NodeType,dtype=torch.long)
                    EdgeType= torch.as_tensor(EdgeType,dtype=torch.long)
                    NodeEvent= torch.as_tensor(Node_Template,dtype=torch.long)
                    NodeMean=torch.as_tensor(Node_Mean,dtype=torch.float32)
                    NodeStd = torch.as_tensor(Node_Std, dtype=torch.float32)
                    y=torch.as_tensor(labels,dtype=torch.long)

                    data = Data(x=X,pre_node_service=Pre_NodeServiceX,post_node_service=Post_NodeServiceX,
                                node_is_root_cause=Node_isRootCauseX,node_mean=NodeMean,node_std=NodeStd,
                                node_event=NodeEvent,node_type=NodeType,edge_type=EdgeType,
                                edge_index=edge_index, y=y)
                    torch.save(data, os.path.join(self.processed_dir, 'data_{}_depth10.pt'.format(idx)))
                    idx+=1
            print(graphfilelist[i] + ": " + str(idx - 1))
            f.close()

    def len(self) -> int:
        datalen=0
        basedir="./OBD/TT/processed"
        for file in os.listdir(basedir):
            datalen+=1
        return datalen-2

    def get(self,idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}_depth10.pt'))
        return data
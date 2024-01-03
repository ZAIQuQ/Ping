import numpy as np 
import pandas as pd
import json, dgl, os, torch
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

embedding_size = 300
dataset_name = 'ifttt_dataset_dgl.bin'
data_source = '../ifttt_build_dataset/'


class IFTTTGraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='ifttt_graph')

    def process(self):
        edges = pd.read_csv(data_source + 'graph_edges.csv')
        properties = pd.read_csv(data_source + 'graph_properties.csv')
        with open(data_source + 'embedding.json','r',encoding='utf8') as fp:
            embedding_dict = json.load(fp)[0]

        self.graphs = []
        self.labels = []
        label_dict = {}
        num_nodes_dict = {}
        
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']
        # 把每个graph id下的边集合整理出来
        edges_group = edges.groupby('graph_id')

        for graph_id in edges_group.groups:
            # 遍历对应ID下边集合,得到邻接节点对(src,dst)
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # 按(src,dst节点对)建图
            g = dgl.graph((src, dst), num_nodes=num_nodes)

            g.ndata['embedding'] = torch.zeros(g.num_nodes(), embedding_size)
            g.ndata['embedding'][src] = torch.Tensor([embedding_dict[str(i)] for i in src])
            g.ndata['embedding'][dst] = torch.Tensor([embedding_dict[str(i)] for i in dst])
            
            self.graphs.append(g)
            self.labels.append(label)

        # 图标签，但是我们没有
        self.labels = torch.LongTensor(self.labels)  


    def __getitem__(self, i):
        # 取ID为i的图和标签
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        # 保存图和标签
        graph_path = os.path.join(data_source, dataset_name)
        save_graphs(graph_path, self.graphs, {'labels': self.labels})

    def load(self):
        # 加载图和标签
        graph_path = os.path.join(data_source, dataset_name)
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']

    def has_cache(self):
        # 检查图数据是否已缓存
        graph_path = os.path.join(data_source, dataset_name)
        return os.path.exists(graph_path)

if __name__ == "__main__":
    IFTTTGraphDataset=IFTTTGraphDataset()

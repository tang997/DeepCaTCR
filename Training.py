import re
import numpy as np
import torch
import math
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from MHSA import ModelB,ModelE,ModelEE,ModelG,ModelW,DeepCAT,DeepLION
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score,f1_score


AAidx_file='AAidx_PCA.txt'
gg=open(AAidx_file)
AAidx_Names=gg.readline().strip().split('\t')
AAidx_Dict={}
for ll in gg.readlines():
    ll=ll.strip().split('\t')
    AA=ll[0]
    tag=0
    aa=[]
    for xx in ll[1:]:
        aa.append(float(xx))
    if tag==1:
        continue
    AAidx_Dict[AA]=aa

#AAindex编码
def AAindexEncoding(Seq):
    Ns = len(Seq)
    max_len = 20  # 设置最大长度为20
    AAE = np.zeros([max_len, 15])  # 创建空的编码数组
    for kk in range(Ns):
        ss = Seq[kk]
        AAE[kk,] = AAidx_Dict[ss]
    if Ns < max_len:
        AAE[Ns:,] = 0  # 填充0行
    AAE = np.transpose(AAE.astype(np.float32))
    return AAE

def read_file(file):
    seq_list= []
    pat = re.compile('[\\*_~XB]')
    with open(file, 'r') as f:
        head_line = True
        for line in f:
            if head_line:
                head_line = False
                continue
            items = line.strip().split('\t')
            aa_seq = items[0]
            if aa_seq[0] != 'C' or aa_seq[-1] != 'F':
                continue
            if len(aa_seq) < 10 or len(aa_seq) > 20:
                continue
            if len(pat.findall(aa_seq)) > 0:
                continue
            seq_list.append(aa_seq)
    f.close()
    return seq_list


class TCRDataset(Dataset):
    def __init__(self, pos_file, neg_file):
        self.sequences = []
        self.labels = []

        # 加载正样本
        seqs_c= read_file(pos_file)
        self.sequences.extend(seqs_c)
        self.labels.extend([1] * len(seqs_c))

        # 加载负样本
        seqs_n= read_file(neg_file)
        self.sequences.extend(seqs_n)
        self.labels.extend([0] * len(seqs_n))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoded = AAindexEncoding(seq)
        return torch.Tensor(encoded), torch.tensor(label, dtype=torch.long)


def train_model():
    # 超参数设置
    BATCH_SIZE = 32
    EPOCHS = 1000
    LR = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = ModelE(0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 加载数据
    train_dataset = TCRDataset(
        'Trainingdata/file1_20000.txt',
        'Trainingdata/NormalCDR3.txt'
    )
    test_dataset = TCRDataset(
        'Trainingdata/file2_10000.txt',
        'Trainingdata/NormalCDR3_test.txt'
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    best_auc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Loss: {total_loss / len(train_loader):.4f} | Val AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}')

        # 保存最佳模型
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f'best_model.pth')
            print(f'New best model saved with AUC: {auc:.4f}')


if __name__ == '__main__':
    # 确保这些函数与您原始代码中的实现一致
    # read_file()
    # AAindexEncoding()
    train_model()










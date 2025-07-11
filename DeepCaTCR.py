import argparse
import os
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, matthews_corrcoef

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

class CAT(nn.Module):
    def __init__(self, aa_num, feature_num, filter_num, kernel_size):
        super(CAT, self).__init__()
        self.aa_num = aa_num
        self.feature_num = feature_num
        self.kernel_size = kernel_size
        self.filter_num = filter_num
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.feature_num,
                    out_channels=self.filter_num[idx],
                    kernel_size=h,
                    stride=1
                ),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(3)
            ) for idx, h in enumerate(self.kernel_size)
        ])

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        x = [conv(x) for conv in self.convs]
        return x

class MHSA(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(MHSA, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.transpose_for_scores(key)
        query_heads = self.transpose_for_scores(query)
        value_heads = self.transpose_for_scores(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)
        return context

class BiLSTMAtte(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTMAtte, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.weight_W = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.weight_W)
        nn.init.xavier_uniform_(self.weight_proj)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)  # [batch_size, seq_len, hidden_size*2]
        u = torch.tanh(torch.matmul(lstm_out, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = lstm_out * att_score
        feat = torch.sum(scored_x, dim=1)
        return feat

class ModelE(nn.Module):
    def __init__(self, drop_out=0.5):
        super(ModelE, self).__init__()
        self.conv_block = CAT(20, 15, [4, 4, 4, 4], [2, 3, 4, 5])
        self.attention_layer = MHSA(2, 4, 4)  
        self.bilstm_att = BiLSTMAtte(3, 3, 1) 
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)  # BiLSTM output: hidden_size*2 (3*2=6)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)  # 多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)  # [batch, seq, channels]
            mx = self.attention_layer(x[i])  # 多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)  # 连接 [batch, 3, 16]
        x = x.permute(0, 2, 1)  # [batch, 16, 3] for BiLSTM
        x = self.bilstm_att(x)  # BiLSTM+Att [batch, 6]
        x = self.dropout(self.fc(x))  # 全连接层
        return x
    
    def forward_motifs(self, x):
        x_conv = x.reshape(-1, 15, 20)  # [batch*tcr_num, 15, 20]
        kernel_sizes = self.conv_block.kernel_size

        motif_indices = []
        for conv_idx, conv in enumerate(self.conv_block.convs):
            conv_layer = conv[0]
            relu = conv[1]
        
            # 计算卷积输出
            conv_out = relu(conv_layer(x_conv))  # [batch*tcr_num, filters, length]
        
            # 全局最大池化 - 获取每个filter的最高激活值和位置
            max_vals, max_indices = torch.max(conv_out, dim=2)  # 形状: [batch*tcr_num, filters]
        
            # 保存位置和分数信息
            motif_indices.append((max_indices, max_vals, conv_out.shape[2], kernel_sizes[conv_idx]))

        with torch.no_grad():
            pred = self.forward(x)
            probs = F.softmax(pred, dim=1)[:, 1]  # 阳性概率

        batch_tcr_num = x_conv.shape[0]
        results = []

        for i in range(batch_tcr_num):
            seq_motifs = []
            for max_indices, scores, seq_len, kernel_size in motif_indices:
                for j in range(max_indices.shape[1]):  # 遍历每个filter
                    pos = max_indices[i, j].item()  # 获取位置
                    # 确保位置在有效范围内
                    if pos < seq_len - kernel_size + 1:
                        score = scores[i, j].item()  # 获取分数
                        seq_motifs.append((pos, kernel_size, score))
            results.append(seq_motifs)

        return probs.cpu().numpy(), results

def calculate_metrics(y_true, y_pred, y_probs):
    """计算所有性能指标"""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0  # 灵敏度
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度

    # 计算MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'ACC': acc,
        'AUC': auc,
        'SEN': sen,
        'SPE': spe,
        'F1': f1,
        'MCC': mcc
    }

def print_metrics(metrics, title="Performance Metrics"):
    """打印性能指标"""
    print(f"\n===== {title} =====")
    print(f"ACC: {metrics['ACC']:.4f}")
    print(f"AUC: {metrics['AUC']:.4f}")
    print(f"SEN: {metrics['SEN']:.4f}")
    print(f"SPE: {metrics['SPE']:.4f}")
    print(f"F1: {metrics['F1']:.4f}")
    print(f"MCC: {metrics['MCC']:.4f}")

def load_AAindex():
    AAidx_file = 'AAidx_PCA.txt'
    with open(AAidx_file) as gg:
        AAidx_Names = gg.readline().strip().split('\t')
        AAidx_Dict = {}
        for ll in gg.readlines():
            ll = ll.strip().split('\t')
            AA = ll[0]
            tag = 0
            aa = []
            for xx in ll[1:]:
                aa.append(float(xx))
            if tag == 1:
                continue
            AAidx_Dict[AA] = aa
    return AAidx_Dict

AAidx_Dict = load_AAindex()

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

def read_sequences(file):
    seq_list = []
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
    return seq_list

class TCRDataset(Dataset):
    def __init__(self, pos_file, neg_file):
        self.sequences = []
        self.labels = []

        # 加载正样本
        seqs_c = read_sequences(pos_file)
        self.sequences.extend(seqs_c)
        self.labels.extend([1] * len(seqs_c))

        # 加载负样本
        seqs_n = read_sequences(neg_file)
        self.sequences.extend(seqs_n)
        self.labels.extend([0] * len(seqs_n))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoded = AAindexEncoding(seq)
        return torch.Tensor(encoded), torch.tensor(label, dtype=torch.long)

def train_model(record_dir, sample_name, device):
    # 创建数据集
    train_dataset = TCRDataset('Trainingdata/caTCRsTrain.txt', 'Trainingdata/non-caTCRsTrain.txt')
    val_dataset = TCRDataset('Trainingdata/caTCRsTest.txt', 'Trainingdata/non-caTCRsTest.txt')
    
    # 创建模型
    model = ModelE(drop_out=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 确保记录目录存在
    os.makedirs(record_dir, exist_ok=True)
    model_save_path = os.path.join(record_dir, f"{sample_name}_DeepCaTCR_best_model.pth")

    # 早停参数
    best_val_loss = float('inf')
    PATIENCE = 20
    patience_counter = 0
    best_metrics = None

    # 训练循环
    for epoch in range(1000):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = (probs > 0.5).long()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算验证指标
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'\nEpoch {epoch + 1}/1000')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print_metrics(metrics, "Validation Metrics")

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = metrics
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path} with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # 打印最佳模型性能
    print("\n===== Best Model Performance =====")
    print(f'Validation Loss: {best_val_loss:.4f}')
    print_metrics(best_metrics)
    
    return model_save_path

def evaluate_model(model_path, device, pos_file='Trainingdata/caTCRsTest.txt', neg_file='Trainingdata/non-caTCRsTest.txt'):
    """评估训练好的模型性能"""
    # 加载测试数据
    test_dataset = TCRDataset(pos_file, neg_file)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 初始化模型
    model = ModelE(drop_out=0.5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for input_data, labels in test_loader:
            input_tensor = input_data.to(device)
            labels = labels.to(device)
            
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    print_metrics(metrics, "Model Evaluation Results")
    
    # 保存结果到文件
    result_file = f"{os.path.dirname(model_path)}/evaluation_results.txt"
    with open(result_file, 'w') as f:
        f.write("Model Evaluation Results:\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"ACC: {metrics['ACC']:.4f}\n")
        f.write(f"AUC: {metrics['AUC']:.4f}\n")
        f.write(f"SEN: {metrics['SEN']:.4f}\n")
        f.write(f"SPE: {metrics['SPE']:.4f}\n")
        f.write(f"F1: {metrics['F1']:.4f}\n")
        f.write(f"MCC: {metrics['MCC']:.4f}\n")
    
    print(f"Evaluation results saved to {result_file}")
    return metrics

def visualize_modelE_motifs(sequences, sample_name, model_path, device, record_dir,
                           score_thres=0.98, topk_motifs=5, max_per_plot=20):
    """可视化高置信度TCR及其motifs"""
    # 创建模型并加载权重
    model = ModelE(0.5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 生成输入数据
    input_data = []
    for seq in sequences:
        encoded = AAindexEncoding(seq)
        input_data.append(encoded)
    
    input_tensor = torch.tensor(np.array(input_data), dtype=torch.float32).to(device)
    
    # 获取motif信息
    with torch.no_grad():
        probs, all_motifs = model.forward_motifs(input_tensor)

    # 筛选高置信度TCR
    high_conf_tcrs = []
    
    for i, seq in enumerate(sequences):
        if probs[i] > score_thres:
            motifs = []
            for (pos, k_size, score) in all_motifs[i]:
                start = pos
                end = start + k_size
                if end <= len(seq):
                    motif_str = seq[start:end]
                    motifs.append((start, end, motif_str, score))
            # 只保留topk motifs
            motifs.sort(key=lambda x: x[3], reverse=True)
            high_conf_tcrs.append((seq, float(probs[i]), motifs[:topk_motifs]))

    if not high_conf_tcrs:
        print(f"No high-confidence TCRs found for {sample_name} with threshold {score_thres}")
        return

    # 创建输出目录
    os.makedirs(record_dir, exist_ok=True)
    safe_name = "".join(c for c in sample_name if c.isalnum() or c in "._- ")

    # 保存高置信度TCR到CSV文件
    csv_path = f"{record_dir}/{safe_name}_high_confidence_tcrs.csv"
    with open(csv_path, 'w') as f:
        f.write("TCR Sequence,Label,Score,Motif1,Motif1_Score,Motif2,Motif2_Score,Motif3,Motif3_Score,Motif4,Motif4_Score,Motif5,Motif5_Score\n")
        for tcr_data in high_conf_tcrs:
            seq, score, motifs = tcr_data
            # 准备motif数据 (最多5个)
            motif_fields = []
            for i in range(5):
                if i < len(motifs):
                    motif_str = motifs[i][2]
                    motif_score = motifs[i][3]
                    motif_fields.extend([motif_str, f"{motif_score:.4f}"])
                else:
                    motif_fields.extend(["", ""])
            # 写入行
            line = f"{seq},1,{score:.4f}," + ",".join(motif_fields) + "\n"
            f.write(line)
    print(f"Saved high-confidence TCRs to CSV: {csv_path}")

    # 可视化设置 
    min_font = 14
    max_font = 24
    row_height = 1.5
    
    # 计算需要多少张图 (每张图最多max_per_plot个TCR)
    num_tcrs = len(high_conf_tcrs)
    num_plots = (num_tcrs + max_per_plot - 1) // max_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_per_plot
        end_idx = min((plot_idx + 1) * max_per_plot, num_tcrs)
        plot_tcrs = high_conf_tcrs[start_idx:end_idx]
        
        fig_height = 3 + len(plot_tcrs) * row_height * 0.8
        fig, ax = plt.subplots(figsize=(15, fig_height))
        plt.title(f"Sample: {safe_name} | TCRs {start_idx+1}-{end_idx} of {num_tcrs}\nScore > {score_thres}", 
                fontsize=14, pad=15)
        
        y_position = 0
        max_seq_len = max(len(tcr[0]) for tcr in plot_tcrs)
        
        for tcr_idx, tcr_data in enumerate(plot_tcrs):
            seq, score, motifs = tcr_data
            seq_len = len(seq)
            
            weights = [0.0] * seq_len
            for motif in motifs:
                start, end, motif_str, motif_score = motif
                for pos in range(start, end):
                    if pos < seq_len:
                        weights[pos] = max(weights[pos], motif_score)
            
            max_weight = max(weights) if weights else 1
            norm_weights = [w / max_weight for w in weights] if max_weight > 0 else [0] * seq_len
            
            x_position = 0
            for pos, aa in enumerate(seq):
                font_size = min_font + norm_weights[pos] * (max_font - min_font)
    
                if norm_weights[pos] < 0.5:
                    # 蓝色到橙色过渡 (0-0.5)
                    interp = norm_weights[pos] * 2  # 映射到0-1范围
                    r = 0.121 + (1.0 - 0.121) * interp    # 1f(31) -> ff(255)
                    g = 0.466 + (0.498 - 0.466) * interp  # 77(119) -> 7f(127)
                    b = 0.705 + (0.055 - 0.705) * interp  # b4(180) -> 0e(14)
                else:
                    # 橙色到红色过渡 (0.5-1.0)
                    interp = (norm_weights[pos] - 0.5) * 2  # 映射到0-1范围
                    r = 1.0 + (0.839 - 1.0) * interp      # ff(255) -> d6(214)
                    g = 0.498 + (0.152 - 0.498) * interp  # 7f(127) -> 27(39)
                    b = 0.055 + (0.156 - 0.055) * interp  # 0e(14) -> 28(40)
    
                ax.text(x_position, y_position, aa, 
                        fontsize=font_size,
                        color=(r, g, b),  # 使用蓝-橙-红渐变
                        fontfamily='monospace',
                        ha='center', va='center',
                        fontweight='bold')
                x_position += 1
            
            ax.text(max_seq_len + 2, y_position, f"Score: {score:.3f}", 
                    ha='left', va='center', fontsize=12)
            
            if tcr_idx < len(plot_tcrs) - 1:
                ax.hlines(y_position - row_height/2, -1, max_seq_len + 10, 
                         colors='gray', linestyles='dashed', alpha=0.5)
            
            y_position -= row_height
        
        ax.set_xlim(-1, max_seq_len + 15)
        ax.set_ylim(y_position - row_height, 1)
        ax.set_aspect('auto')
        ax.set_axis_off()
        plt.tight_layout()
        
        img_path = f"{record_dir}/{safe_name}_motifs_plot{plot_idx+1}.svg"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved motif visualization: {img_path}")

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='Train ModelE and visualize motifs')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--visualize', action='store_true', help='Visualize motifs')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model performance')
    parser.add_argument('--record_dir', type=str, help='Output directory for results', default='results')
    parser.add_argument('--sample_name', type=str, help='Sample name for visualization', default='my_experiment')
    parser.add_argument('--model_path', type=str, help='Path to trained model for evaluation/visualization')
    parser.add_argument('--pos_file', type=str, help='Positive sequences file', default='Trainingdata/caTCRsTest.txt')
    parser.add_argument('--neg_file', type=str, help='Negative sequences file', default='Trainingdata/non-caTCRsTest.txt')

    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.record_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 训练模型
    if args.train:
        model_path = train_model(
            record_dir=args.record_dir,
            sample_name=args.sample_name,
            device=device
        )
        args.model_path = model_path  # 更新模型路径
    
    # 评估模型
    if args.evaluate:
        if not args.model_path:
            # 默认使用训练好的模型路径
            args.model_path = f'{args.record_dir}/{args.sample_name}_DeepCaTCR_best_model.pth'
        
        evaluate_model(
            model_path=args.model_path,
            device=device,
            pos_file=args.pos_file,
            neg_file=args.neg_file
        )
    
    # 可视化motifs
    if args.visualize:
        if not args.model_path:
            # 默认使用训练好的模型路径
            args.model_path = f'{args.record_dir}/{args.sample_name}_DeepCaTCR_best_model.pth'
        
        # 加载测试数据
        pos_seqs = read_sequences(args.pos_file)
        neg_seqs = read_sequences(args.neg_file)
        test_sequences = pos_seqs + neg_seqs
        
        # 可视化
        visualize_modelE_motifs(
            sequences=test_sequences,
            sample_name=args.sample_name,
            model_path=args.model_path,
            device=device,
            record_dir=args.record_dir
        )

if __name__ == '__main__':
    main()
import re
import math
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



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
        lstm_out, _ = self.bilstm(x)
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
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])
            m.append(mx)
        x = torch.cat(m, dim=2)
        x = x.permute(0, 2, 1)
        x = self.bilstm_att(x)
        x = self.dropout(self.fc(x))
        return x

    def forward_motifs(self, x):
        x_conv = x.reshape(-1, 15, 20)
        motif_indices = []
        
        for conv_idx, conv in enumerate(self.conv_block.convs):
            conv_out = conv[1](conv[0](x_conv))
            max_vals, max_indices = torch.max(conv_out, dim=2)
            motif_indices.append((
                max_indices, 
                max_vals, 
                conv_out.shape[2], 
                self.conv_block.kernel_size[conv_idx]
            ))

        with torch.no_grad():
            probs = F.softmax(self.forward(x), dim=1)[:, 1]

        results = []
        for i in range(x_conv.shape[0]):
            seq_motifs = []
            for max_indices, scores, seq_len, kernel_size in motif_indices:
                for j in range(max_indices.shape[1]):
                    pos = max_indices[i, j].item()
                    if pos < seq_len - kernel_size + 1:
                        seq_motifs.append((
                            pos, 
                            kernel_size, 
                            scores[i, j].item()
                        ))
            results.append(seq_motifs)

        return probs.cpu().numpy(), results


class AAIndexEncoder:
    def __init__(self, aaidx_file='AAidx_PCA.txt'):
        self.aaidx_dict = self._load_aaidx(aaidx_file)
        
    def _load_aaidx(self, file_path):
        with open(file_path) as f:
            f.readline()  # skip header
            return {
                line.strip().split('\t')[0]: 
                [float(x) for x in line.strip().split('\t')[1:]]
                for line in f
            }
    
    def encode(self, seq, max_len=20):
        encoding = np.zeros([max_len, 15])
        for i, aa in enumerate(seq[:max_len]):
            encoding[i] = self.aaidx_dict.get(aa, [0]*15)
        return np.transpose(encoding.astype(np.float32))


class TCRProcessor:
    def __init__(self):
        self.pattern = re.compile('[\\*_~XB]')
        
    def is_valid_sequence(self, seq):
        return (seq.startswith('C') and seq.endswith('F') and 
                10 <= len(seq) <= 20 and 
                not self.pattern.search(seq))
    
    def read_file(self, file_path):
        sequences = []
        with open(file_path, 'r') as f:
            next(f)  # skip header
            for line in f:
                seq = line.strip().split('\t')[0]
                if self.is_valid_sequence(seq):
                    sequences.append(seq)
        return sequences


class CancerScorePredictor:
    def __init__(self, model_path, device):
        self.device = device
        self.model = self._load_model(model_path)
        self.encoder = AAIndexEncoder()
        self.processor = TCRProcessor()
        
    def _load_model(self, model_path):
        model = ModelE(0.5).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.eval()
    
    def calculate_scores(self, sequences):
        encoded_seqs = np.array([self.encoder.encode(seq) for seq in sequences])
        input_tensor = torch.Tensor(encoded_seqs).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
            probs = [1 / (1 + math.exp(-(p[1] - p[0]))) for p in predictions]
            
        mean_score = np.mean(probs)
        variance = np.mean([(p - mean_score)**2 for p in probs])
        return mean_score, variance
    
    def process_directory(self, input_dir, output_file):
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)        
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Sample', 'meanscore', 'variance'])
            writer.writeheader()
            
            for filename in os.listdir(input_dir):
                file_path = os.path.join(input_dir, filename)
                if not os.path.isfile(file_path):
                    continue
                    
                sequences = self.processor.read_file(file_path)
                if not sequences:
                    print(f"Warning: No valid sequences in {filename}")
                    continue
                    
                mean_score, variance = self.calculate_scores(sequences)
                writer.writerow({
                    'Sample': filename,
                    'meanscore': mean_score,
                    'variance': variance
                })


def main():
    parser = argparse.ArgumentParser(description='Predict cancer scores and visualize TCR motifs')
    parser.add_argument('--input_dir', required=True, help='Directory containing TCR sequence files')
    parser.add_argument('--model_path', default='pre-trained_model/my_test_DeepCaTCR_best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--output_dir', default='repertoire_score', help='Output directory for results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    predictor = CancerScorePredictor(args.model_path, device)
    output_file = os.path.join(args.output_dir, 'cancer_score.csv')
    predictor.process_directory(args.input_dir, output_file)
    
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
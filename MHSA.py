
import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

class MHSA(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(MHSA, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context


class TextCNN(nn.Module):     #定义了一个名为DeepLION的类，并继承了nn.Module类
    def __init__(self, aa_num, feature_num, filter_num, kernel_size):
        super(TextCNN, self).__init__()
        self.aa_num=aa_num
        self.feature_num=feature_num
        self.kernel_size=kernel_size
        self.filter_num=filter_num
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(3))
            for idx, h in enumerate(self.kernel_size)
        ])

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        x = [conv(x) for conv in self.convs]
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(ConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将输入的维度由 [batch_size, seq_len, in_channels] 转为 [batch_size, in_channels, seq_len]
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        conv_outs = [F.max_pool1d(conv_out, conv_out.shape) for conv_out in conv_outs]  # 将每个卷积层的输出进行最大池化
        x = torch.cat(conv_outs, dim=1)  # 将三个池化后的结果在通道维度上拼接
        x = x.permute(0, 2, 1)  # 将维度转回 [batch_size, seq_len, out_channels*3]
        return x



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()

        query = self.query_proj(x).view(batch_size * self.num_heads, seq_len, self.head_dim).transpose(1,
                                                                                                       2)  # 将 query 的维度变为 [batch_size*num_heads, head_dim, seq_len]
        key = self.key_proj(x).view(batch_size * self.num_heads, seq_len, self.head_dim).transpose(1,
                                                                                                   2)  # 将 key 的维度变为 [batch_size*num_heads, head_dim, seq_len]
        value = self.value_proj(x).view(batch_size * self.num_heads, seq_len, self.head_dim).transpose(1,
                                                                                                       2)  # 将 value 的维度变为 [batch_size*num_heads, head_dim, seq_len]

        attn_weights = F.softmax(torch.bmm(query, key.transpose(1, 2)) / (self.head_dim ** 0.5),
                                 dim=-1)  # 计算注意力权重，使用 torch.bmm 计算两个矩阵的乘积
        attn_output = torch.bmm(attn_weights, value).transpose(1, 2)  # 将每个头的注意力输出拼接后，恢复成 [batch_size, seq_len, hidden_size] 的形状
        out = self.out_proj(attn_output)
        return out


class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=16, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class BiLSTMAtte(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTMAtte, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.weight_W = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)  # lstm_out shape: [batch_size, seq_len, hidden_size * 2]
        # Attention
        u = torch.tanh(torch.matmul(lstm_out, self.weight_W))  # u shape: [batch_size, seq_len, hidden_size * 2]
        att = torch.matmul(u, self.weight_proj)  # att shape: [batch_size, seq_len, 1]
        att_score = F.softmax(att, dim=1)  # att_score shape: [batch_size, seq_len, 1]
        scored_x = lstm_out * att_score  # scored_x shape: [batch_size, seq_len, hidden_size * 2]
        feat = torch.sum(scored_x, dim=1)  # feat shape: [batch_size, hidden_size * 2]
        return feat

class BiLSTMAttention(nn.Module):
    def __init__(self):
        super(BiLSTMAttention, self).__init__()
        self.bilstm = nn.LSTM(16, 16, batch_first=True, bidirectional=True)
        self.weight_W = nn.Parameter(torch.Tensor(16 * 2, 16 * 2))
        self.weight_proj = nn.Parameter(torch.Tensor(16 * 2, 1))

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)  # lstm_out shape: [batch_size, seq_len, hidden_size * 2]
        # Attention
        u = torch.tanh(torch.matmul(lstm_out, self.weight_W))  # u shape: [batch_size, seq_len, hidden_size * 2]
        att = torch.matmul(u, self.weight_proj)  # att shape: [batch_size, seq_len, 1]
        att_score = F.softmax(att, dim=1)  # att_score shape: [batch_size, seq_len, 1]
        scored_x = lstm_out * att_score  # scored_x shape: [batch_size, seq_len, hidden_size * 2]
        feat = torch.sum(scored_x, dim=1)  # feat shape: [batch_size, hidden_size * 2]
        return feat

    #这段代码实现了一个双向LSTM模型，并在此基础上加入了注意力机制。
    # 具体来说，输入数据x经过双向LSTM处理后得到输出lstm_out，然后计算注意力向量att_score，
    # 并利用该向量对lstm_out进行加权求和得到feat。最后将feat输入到全连接层中进行分类或回归等任务。

class Model(nn.Module):
    def __init__(self,drop_out):
        super(Model, self).__init__()
        self.conv_block = TextCNN(24,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MultiHeadSelfAttention(4,2)
        self.fusion_layer = FusionLayer()
        #self.ins_num=ins_num
        self.drop_out=drop_out
        self.fc = nn.Linear(3*16, 2)
        #self.fc_1 = nn.Linear(self.ins_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)
        m=[]
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])
            m.append(mx)
        x = torch.cat(m, dim=2)
        x = x.reshape(-1, 3*16)
        x = self.dropout(self.fc(x))
        #x= x.reshape(-1, self.ins_num)
        #x = self.dropout(self.fc_1(x))
        return x


class Text_CNN(nn.Module):
    def __init__(self, aa_num, feature_num, filter_num, kernel_size,drop_out):
        super(Text_CNN, self).__init__()
        self.aa_num=aa_num
        self.feature_num=feature_num
        self.kernel_size=kernel_size
        self.filter_num=filter_num
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))  #
            for idx, h in enumerate(self.kernel_size)
        ])
        self.drop_out = drop_out
        self.dropout = nn.Dropout(p=self.drop_out)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)     #(-1,14,1)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, 16)
        x = self.dropout(self.fc(x))
        return x


class CAT(nn.Module):
    def __init__(self, aa_num, feature_num, filter_num, kernel_size):
        super(CAT, self).__init__()
        self.aa_num=aa_num
        self.feature_num=feature_num
        self.kernel_size=kernel_size
        self.filter_num=filter_num
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(3))  #
            for idx, h in enumerate(self.kernel_size)
        ])

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        x = [conv(x) for conv in self.convs]
        return x

class CAT1(nn.Module):
    def __init__(self, aa_num, feature_num, filter_num, kernel_size):
        super(CAT1, self).__init__()
        self.aa_num=aa_num
        self.feature_num=feature_num
        self.kernel_size=kernel_size
        self.filter_num=filter_num
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))  #
            for idx, h in enumerate(self.kernel_size)
        ])

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)   #15,24
        x = [conv(x) for conv in self.convs]     #4,1   4,1
        return x

class ModelEa(nn.Module):
    def __init__(self,drop_out):
        super(ModelEa,self).__init__()
        self.conv_block=CAT(24,15,[4,4,4,4],[2,3,4,5])
        self.bilstm_att=BiLSTMAtte(4,4,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(8, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN  4,1
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)   #1,4
        x = torch.cat(x, dim=1)     #连接   [-1,1,4]
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

class ModelEa20(nn.Module):
    def __init__(self,drop_out):
        super(ModelEa20,self).__init__()
        self.conv_block=CAT(19,20,[4,4,4,4],[2,3,4,5])
        self.bilstm_att=BiLSTMAtte(4,4,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(8, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN  4,1
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)   #1,4
        x = torch.cat(x, dim=1)     #连接   [-1,1,4]
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

class ModelEa8(nn.Module):
    def __init__(self,drop_out):
        super(ModelEa8,self).__init__()
        self.conv_block=CAT(24,15,[8,8,8,8],[2,3,4,5])
        self.bilstm_att=BiLSTMAtte(8,8,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(16, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN  4,1
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)   #1,4
        x = torch.cat(x, dim=1)     #连接   [-1,1,4]
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

class ModelEa16(nn.Module):
    def __init__(self,drop_out):
        super(ModelEa16,self).__init__()
        self.conv_block=CAT(24,15,[16,16,16,16],[2,3,4,5])
        self.bilstm_att=BiLSTMAtte(16,16,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(16*2, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN  4,1
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)   #1,4
        x = torch.cat(x, dim=1)     #连接   [-1,1,4]
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

##cat单通道使用MHSA+Line
class ModelM(nn.Module):
    def __init__(self,ins_num,drop_out):
        super(ModelM, self).__init__()
        self.conv_block = CAT(24,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(2,16,16)
        self.fusion_layer = FusionLayer()
        self.ins_num=ins_num
        self.drop_out=drop_out
        self.fc = nn.Linear(16*1, 1)
        self.fc_1 = nn.Linear(self.ins_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)
        x=torch.cat(x,dim=1)
        x= x.permute(0, 2, 1)
        x=self.attention_layer(x)
        x = x.reshape(-1, 16*1)
        x = self.dropout(self.fc(x))
        x= x.reshape(-1, self.ins_num)
        x = self.dropout(self.fc_1(x))
        return x


class ModelEa3(nn.Module):
    def __init__(self,drop_out):
        super(ModelEa3,self).__init__()
        self.conv_block=CAT(20,15,[3,3,3,3],[2,3,4,5])
        self.attention_layer = MHSA(1, 3, 3)
        self.bilstm_att=BiLSTMAtte(3,3,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,3,12]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x


class ModelEa2(nn.Module):
    def __init__(self,drop_out):
        super(ModelEa2,self).__init__()
        self.conv_block=CAT(20,15,[2,2,2,2],[2,3,4,5])
        self.attention_layer = MHSA(1, 2, 2)
        self.bilstm_att=BiLSTMAtte(3,3,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,3,8]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

class ModelEa4(nn.Module):
    def __init__(self,drop_out):
        super(ModelEa4,self).__init__()
        self.conv_block=CAT(20,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(1, 4, 4)
        self.bilstm_att=BiLSTMAtte(3,3,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,3,8]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

class ModelEa5(nn.Module):
    def __init__(self,drop_out):
        super(ModelEa5,self).__init__()
        self.conv_block=CAT(20,15,[5,5,5,5],[2,3,4,5])
        self.attention_layer = MHSA(1, 5, 5)
        self.bilstm_att=BiLSTMAtte(3,3,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,3,8]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

#cat单通道使用3/5-maxpooling:MHSA+1-maxpooling+Line
class ModelQ(nn.Module):
    def __init__(self,ins_num,drop_out):
        super(ModelQ, self).__init__()
        self.conv_block = CAT(24,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(2,16,16)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion_layer = FusionLayer()
        self.ins_num=ins_num
        self.drop_out=drop_out
        self.fc = nn.Linear(16, 1)
        self.fc_1 = nn.Linear(self.ins_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)
        x=torch.cat(x,dim=1)
        x= x.permute(0, 2, 1)
        x=self.attention_layer(x)
        x = x.permute(0, 2, 1)
        x=self.avg_pool(x)
        x = x.reshape(-1, 1,16)
        x = self.dropout(self.fc(x))
        x= x.reshape(-1, self.ins_num)
        x = self.dropout(self.fc_1(x))
        return x

#多通道使用MHSA+cat+Line
class ModelN(nn.Module):
    def __init__(self,ins_num,drop_out):
        super(ModelN,self).__init__()
        self.conv_block=CAT(24,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(2, 4, 4)
        self.fusion_layer = FusionLayer()
        self.ins_num = ins_num
        self.drop_out = drop_out
        self.fc = nn.Linear(16* 1, 1)
        self.fc_1 = nn.Linear(self.ins_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])
            m.append(mx)
        x = torch.cat(m, dim=2)
        x = x.reshape(-1, 1*16)
        x = self.dropout(self.fc(x))
        x = x.reshape(-1, self.ins_num)
        x = self.dropout(self.fc_1(x))
        return x



class ModelC(nn.Module):
    def __init__(self,ins_num,drop_out):
        super(ModelC,self).__init__()
        self.conv_block=CAT(24,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(2, 4, 4)
        self.bilstm_att=BiLSTMAtte(3,3,1)
        self.ins_num = ins_num
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 1)   #？32*3
        self.fc_1 = nn.Linear(self.ins_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN    [-1,15,24]    [-1,4,3]
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)       #[-1,3,4]
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接    [-1,3,16]
        x=x.permute(0,2,1)     #[-1,16,3]
        x=self.bilstm_att(x)     #BiLSTM+Att
        x = self.dropout(self.fc(x))   #全连接层
        x = x.reshape(-1, self.ins_num)
        x = self.dropout(self.fc_1(x))
        return x

class ModelB(nn.Module):
    def __init__(self,drop_out):
        super(ModelB,self).__init__()
        self.conv_block=CAT(20,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(2, 4, 4)
        self.bilstm_att=BiLSTMAttention()
        self.drop_out = drop_out
        self.fc = nn.Linear(32, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,3,16]
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x



class ModelE(nn.Module):
    def __init__(self,drop_out):
        super(ModelE,self).__init__()
        self.conv_block=CAT(20,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(2, 4, 4)
        self.bilstm_att=BiLSTMAtte(3,3,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,3,16]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x



class Model2(nn.Module):
    def __init__(self,drop_out):
        super(Model2,self).__init__()

        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,3,16]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

#宽为2
class CAT2(nn.Module):
    def __init__(self, aa_num, feature_num, filter_num, kernel_size):
        super(CAT2, self).__init__()
        self.aa_num=aa_num
        self.feature_num=feature_num
        self.kernel_size=kernel_size
        self.filter_num=filter_num
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(2))  #
            for idx, h in enumerate(self.kernel_size)
        ])

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        x = [conv(x) for conv in self.convs]
        return x


#宽为2
class ModelE2(nn.Module):
    def __init__(self,drop_out):
        super(ModelE2,self).__init__()
        self.conv_block=CAT2(24,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(2, 4, 4)
        self.bilstm_att=BiLSTMAtte(2,2,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(4, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,2,16]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

class CAT4(nn.Module):
    def __init__(self, aa_num, feature_num, filter_num, kernel_size):
        super(CAT4, self).__init__()
        self.aa_num=aa_num
        self.feature_num=feature_num
        self.kernel_size=kernel_size
        self.filter_num=filter_num
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(4))  #
            for idx, h in enumerate(self.kernel_size)
        ])

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        x = [conv(x) for conv in self.convs]
        return x

class ModelE4(nn.Module):
    def __init__(self,drop_out):
        super(ModelE4,self).__init__()
        self.conv_block=CAT4(20,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(2, 4, 4)
        self.bilstm_att=BiLSTMAtte(4,4,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(8, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,2,16]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x


class CAT5(nn.Module):
    def __init__(self, aa_num, feature_num, filter_num, kernel_size):
        super(CAT5, self).__init__()
        self.aa_num=aa_num
        self.feature_num=feature_num
        self.kernel_size=kernel_size
        self.filter_num=filter_num
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(5))  #
            for idx, h in enumerate(self.kernel_size)
        ])

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        x = [conv(x) for conv in self.convs]
        return x

class ModelE5(nn.Module):
    def __init__(self,drop_out):
        super(ModelE5,self).__init__()
        self.conv_block=CAT5(20,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(2, 4, 4)
        self.bilstm_att=BiLSTMAtte(5,5,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(10, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,2,16]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x

class ModelEE(nn.Module):
    def __init__(self,drop_out):
        super(ModelEE,self).__init__()
        self.conv_block=CAT(20,15,[8,8,8,8],[2,3,4,5])
        self.attention_layer = MHSA(2, 8, 8)  #卷机核数量一多，这里就可以多设置注意力头
        self.bilstm_att=BiLSTMAtte(3,3,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN
        m = []
        for i in range(len(x)):
            x[i] = x[i].permute(0, 2, 1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=2)     #连接   [-1,3,16]
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x


class ModelG(nn.Module):
    def __init__(self,drop_out):
        super(ModelG,self).__init__()
        self.conv_block=CAT(20,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(1, 3, 3)
        self.bilstm_att=BiLSTMAtte(3,3,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN   [-1,4,3]
        m = []
        for i in range(len(x)):
            mx = self.attention_layer(x[i])   #多通道MHSA   seq=4  dim=3
            m.append(mx)
        x = torch.cat(m, dim=1)     #连接   [-1,16,3]
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x


class ModelH(nn.Module):
    def __init__(self,drop_out):
        super(ModelH,self).__init__()
        self.conv_block=CAT(20,15,[4,4,4,4],[2,3,4,5])
        self.attention_layer = MHSA(1, 1, 1)
        self.bilstm_att=BiLSTMAtte(1,1,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(2, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN   [-1,8,3]
        m = []
        for i in range(len(x)):
            x[i] = x[i].reshape(-1,12,1)
            mx = self.attention_layer(x[i])   #多通道MHSA
            m.append(mx)
        x = torch.cat(m, dim=1)     #连接   [-1,48,1]
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.fc(x)  #全连接层
        return x


class ModelK(nn.Module):
    def __init__(self,drop_out):
        super(ModelK,self).__init__()
        self.conv_block=CAT(20,15,[4,4,4,4],[2,3,4,5])
        self.bilstm_att=BiLSTMAtte(3,3,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.conv_block(x)   #多通道TextCNN   [-1,4,3]
        x = torch.cat(x, dim=1)     #连接   [-1,16,3]
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, hidden_size * 2]
        x = self.dropout(self.fc(x))   #全连接层
        return x


class ModelW(nn.Module):
    def __init__(self,drop_out):
        super(ModelW,self).__init__()
        self.bilstm_att=BiLSTMAtte(15,15,1)
        self.drop_out = drop_out
        self.fc = nn.Linear(30, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.bilstm_att(x)     #BiLSTM+Att     #[-1, 30]
        x = self.dropout(self.fc(x))   #全连接层
        return x

class DeepLION(nn.Module):     #定义了一个名为DeepLION的类，并继承了nn.Module类
    def __init__(self, aa_num, feature_num, filter_num, kernel_size, drop_out):
        super(DeepLION, self).__init__()
        self.aa_num = aa_num
        self.feature_num = feature_num
        self.filter_num = filter_num    #[3, 3, 3, 2, 2, 1]
        self.kernel_size = kernel_size   #[2, 3, 4, 5, 6, 7]
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)    #可借用
        self.fc = nn.Linear(sum(self.filter_num), 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, 1*sum(self.filter_num))
        out = self.dropout(self.fc(out))
        return out

class DeepCAT(nn.Module):
    def __init__(self, aa_num, feature_num, drop_out):
        super(DeepCAT, self).__init__()
        self.aa_num = aa_num
        self.feature_num = feature_num
        self.drop_out = drop_out
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(15,2),
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(1,2),
                stride=1
            ),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=(1,2),stride=1),
        )
        self.out = nn.Linear(16 *1* 16, 50)
        self.out_1 = nn.Linear(50, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1,1,self.feature_num, self.aa_num)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(-1, 16*1*16)
        x=self.dropout(self.out(x))
        x = self.dropout(self.out_1(x))
        return x






# 定义全连接网络模型
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):  # 对于一维输入，input_dim即为data的长度
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # 可选的正则化
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # Sigmoid激活适用于二分类问题
        out = torch.sigmoid(out)
        return out




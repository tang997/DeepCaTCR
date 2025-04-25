import re
import numpy as np
import torch
import math
import os
from MHSA import ModelB,ModelE,ModelEE,ModelG,ModelW,DeepCAT,DeepLION
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score,f1_score


    #下载模型
model = ModelE(0.4)
model.load_state_dict(torch.load('Pretrained_C2.pth'))

#model=DeepLION(20,15,[3,3,3,2,2,1],[2,3,4,5,6,7],0.4)
#model.load_state_dict(torch.load('Pretrained_C3.pth'))

#model=DeepCAT(20,15,0.4)
#model.load_state_dict(torch.load('Pretrained_C4.pth'))


model = model.eval()


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

#读取文件
def read_file(file):
    seq_list, freq_list = [], []
    pat = re.compile('[\\*_~XB]')
    with open(file, 'r') as f:
        head_line = True
        for line in f:
            if head_line:
                head_line = False
                continue
            items = line.strip().split('\t')
            aa_seq, freq = items[0], items[2]
            try:
                freq = float(items[2])
            except Exception as e:
                print("Clonal frequency is not a numeric type, please check your input file")
                continue
            if aa_seq[0] != 'C' or aa_seq[-1] != 'F':
                continue
            if len(aa_seq) < 10 or len(aa_seq) > 20:
                continue
            if len(pat.findall(aa_seq)) > 0:
                continue
            seq_list.append(aa_seq)
            freq_list.append(freq)
    f.close()
    return seq_list, freq_list


def prescore(seq,freq):
    sumz=np.sum(freq)
    fren=[float(ss/sumz) for ss in freq]  #加权频率
    encode_seq = [AAindexEncoding(ss) for ss in seq]
    encode_seq = np.array(encode_seq)
    input_pre = torch.Tensor(encode_seq).to(torch.device('cpu'))
    predict = model(input_pre)
    prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
    probb = [(float(1 / (1 + math.exp(-predict[i][1] + predict[i][0])))) for i in range(len(predict))]
    score = sum(x * y for x, y in zip(probb, fren))    #加权平均数
    score2 = np.mean(probb)    #平均数
    ss=[(pro-score2) * (pro-score2) for pro in probb]
    sss=np.mean(ss)     #方差
    return score,score2,sss



with open('zukus_score/predir.tsv', "w", encoding="utf8") as output_file:
    output_file.write("Sample\tweighedscore\tmeanscore\tvariance\n")

    for ff in os.listdir('2_pan5/'):
            # Read sample.
        seq, freq = read_file('2_pan5/' + ff)
        #s, s2, pro = prescore(seq, freq)
        s,s2,s3 = prescore(seq, freq)

            # Save result.
        output_file.write("{0}\t{1}\t{2}\t{3}\n".format(ff, s,s2,s3))
print("The prediction results have been saved to: ")












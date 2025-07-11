import random
import os

# 设置随机种子确保结果可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def split_dataset(original_file, train_file, test_file, split_ratio=0.8):
    """划分单个数据集为训练集和测试集"""
    with open(original_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 打乱数据（使用固定种子）
    random.shuffle(lines)
    total_lines = len(lines)
    train_lines_count = int(total_lines * split_ratio)
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as train_file:
        train_file.writelines(lines[:train_lines_count])
    
    # 写入测试集
    with open(test_file, 'w', encoding='utf-8') as test_file:
        test_file.writelines(lines[train_lines_count:])
    
    return train_lines_count, total_lines - train_lines_count

if __name__ == "__main__":
    # ========== 划分肿瘤数据 ==========
    tumor_train, tumor_test = split_dataset(
        original_file='caTCRs.txt',
        train_file='caTCRsTrain.txt',
        test_file='caTCRsTest.txt'
    )
    # ========== 划分正常数据 ==========
    normal_train, normal_test = split_dataset(
        original_file='non-caTCRs.txt',
        train_file='non-caTCRsTrain.txt',
        test_file='non-caTCRsTest.txt'
    )
    # 保存划分信息
    with open('dataset_split_info.txt', 'w') as f:
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"caTCRs Dataset: Total={tumor_train+tumor_test}, Train={tumor_train}, Test={tumor_test}\n")
        f.write(f"non-caTCRs Dataset: Total={normal_train+normal_test}, Train={normal_train}, Test={normal_test}\n")
    
    print("\n划分信息已保存到 dataset_split_info.txt")
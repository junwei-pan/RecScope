import pickle
import numpy as np
from tqdm import tqdm

# train_pre的作用是生成在计算互信息时，计算公式中用到条件概率所需要使用的查询表，一共有四个，notation以及对应的物理含义介绍如下
adj_p = {}  # 存储pair(i,j,pos), i代表当前广告，j代表历史行为，作为正样本(y=1)一共出现了几次
adj_n = {}  # 存储pair(i,j,pos), i代表当前广告，j代表历史行为，作为负样本(y=0)一共出现了几次
target_p = np.zeros(801)  # 存储target item对应的category id(或item id)作为正样本一共出现了几次， 801对应catgory ID的特征总数
target_n = np.zeros(801)  # 存储target item对应的category id(或item id)作为负样本一共出现了几次

with open('dataset.pkl', 'rb') as f:  # 训练数据集
    train_set = pickle.load(f)  # 2223958
    test_set = pickle.load(f)  # 192403
    eval_set = pickle.load(f)  # 192403
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)
    tmp = eval_set[0]

    for i in tqdm(range(len(train_set))):
        tmp = train_set[i]  # 数据集中某一条样本以list的方式存储形式为[103944, [17704, 37473], 53346, 0)，
        hist = tmp[1]  # 历史序列
        target = tmp[2]  # target item id
        l = len(tmp[1])
        y = tmp[3]  # 标签
        if y == 1:
            for j in range(len(hist)):
                if (cate_list[target], cate_list[hist[j]], l-j) in adj_p:  # 使用cate_list将item id映射到category id
                    adj_p[(cate_list[target], cate_list[hist[j]], l-j)] += 1
                else:
                    adj_p[(cate_list[target], cate_list[hist[j]], l-j)] = 1
            target_p[cate_list[target]] += 1
        elif y == 0:
            for j in range(len(hist)):
                if (cate_list[target], cate_list[hist[j]], l-j) in adj_n:
                    adj_n[(cate_list[target], cate_list[hist[j]], l-j)] += 1
                else:
                    adj_n[(cate_list[target], cate_list[hist[j]], l-j)] = 1
            target_n[cate_list[target]] += 1

    for i in tqdm(range(len(eval_set))):
        tmp = eval_set[i]  # (34945, [31709, 37167, 15260, 60902, 50938], (62221, 2141))
        hist = tmp[1]  # 历史序列
        l = len(hist)
        eval_p = tmp[2][0]  # positive target item id
        for j in range(len(hist)):
            if (cate_list[eval_p], cate_list[hist[j]], l-j) in adj_p:  # 使用cate_list将item id映射到category id
                adj_p[(cate_list[eval_p], cate_list[hist[j]], l-j)] += 1
            else:
                adj_p[(cate_list[eval_p], cate_list[hist[j]], l-j)] = 1
        target_p[cate_list[eval_p]] += 1

        eval_n = tmp[2][1]  # negative target item id
        for j in range(len(hist)):
            if (cate_list[eval_n], cate_list[hist[j]], l-j) in adj_n:
                adj_n[(cate_list[eval_n], cate_list[hist[j]], l-j)] += 1
            else:
                adj_n[(cate_list[eval_n], cate_list[hist[j]], l-j)] = 1
        target_n[cate_list[eval_n]] += 1

    for i in tqdm(range(len(test_set))):
        tmp = test_set[i]  # (34945, [31709, 37167, 15260, 60902, 50938], (62221, 2141))
        hist = tmp[1]  # 历史序列
        l = len(hist)
        eval_p = tmp[2][0]  # positive target item id
        for j in range(len(hist)):
            if (cate_list[eval_p], cate_list[hist[j]], l-j) in adj_p:  # 使用cate_list将item id映射到category id
                adj_p[(cate_list[eval_p], cate_list[hist[j]], l-j)] += 1
            else:
                adj_p[(cate_list[eval_p], cate_list[hist[j]], l-j)] = 1
        target_p[cate_list[eval_p]] += 1

        eval_n = tmp[2][1]  # negative target item id
        for j in range(len(hist)):
            if (cate_list[eval_n], cate_list[hist[j]], l-j) in adj_n:
                adj_n[(cate_list[eval_n], cate_list[hist[j]], l-j)] += 1
            else:
                adj_n[(cate_list[eval_n], cate_list[hist[j]], l-j)] = 1
        target_n[cate_list[eval_n]] += 1

    np.save('./source_new/adj_p_pos.npy', adj_p)
    np.save('./source_new/adj_n_pos.npy', adj_n)
    np.save('./source_new/target_p_pos.npy', target_p)
    np.save('./source_new/target_n_pos.npy', target_n)

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

with open('dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    eval_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

adj_p = np.load('./source_new/adj_p_pos.npy', allow_pickle=True).item()
adj_n = np.load('./source_new/adj_n_pos.npy', allow_pickle=True).item()
target_p = np.load('./source_new/target_p_pos.npy')
target_n = np.load('./source_new/target_n_pos.npy')

c_mul = np.zeros((10, 10))
target = 674
t_p = target_p[target]  # 当前样本中target item a对应的category id作为正样本出现的总次数
t_n = target_n[target]  # 当前样本中target item a对应的category id作为负样本出现的总次数
s = t_p + t_n  # 当前样本中target item a出现的总次数
hist = [281, 483, 674, 351, 44, 339, 115, 558, 157, 142]

for i in range(10):
    for p in range(1, 11):
        m = 0
        if (target, hist[i], p) in adj_p:
            x_p = adj_p[(target, hist[i], p)]
        else:
            x_p = 1e-5
        if (target, hist[i], p) in adj_n:
            x_n = adj_n[(target, hist[i], p)]
        else:
            x_n = 1e-5

        m1 = x_p / s * np.log2(x_p * s / ((x_p + x_n) * t_p))  # x_i = 1, y=1
        m2 = x_n / s * np.log2(x_n * s / ((x_p + x_n) * t_n))  # x_i = 1, y=0
        m3 = (t_p - x_p) / s * np.log2((t_p - x_p) * s / ((t_p + t_n - x_p - x_n) * t_p))  # x_i = 0, y = 1
        m4 = (t_n - x_n) / s * np.log2((t_n - x_n) * s / ((t_p + t_n - x_p - x_n) * t_n))  # x_i = 0, y = 0

        m = m1 + m2 + m3 + m4
        c_mul[i][p-1] = m

c_mul = normalization(c_mul)
print(c_mul)

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(
    pd.DataFrame(np.round(c_mul, 2), columns=[i+1 for i in range(10)], index=hist),
    annot=True, xticklabels=True, yticklabels=True, square=True, cmap="Blues",
    cbar=False)
# cbar_kws={"shrink": 0.8, "use_gridspec": False, "location": "right"}

ax.set_title('Category-wise Target-aware Correlation', fontsize=12)
ax.set_xlabel('Target-relative Position', fontsize=10)
ax.set_ylabel('Top-10 behavior categories', fontsize=10)
plt.savefig("amazon_category_%d.pdf" % target, dpi=200, bbox_inches='tight')
plt.show()
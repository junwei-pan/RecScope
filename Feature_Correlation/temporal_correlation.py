import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def gauss_normalize(x):
    return (x - np.mean(x)) / np.std(x)


model = "tin"
cat = np.load('cat.npy')
T = ['281', '483', '674', '351', '44', '339', '115', '558', '157', '142']
att = np.zeros(10)
val = np.zeros((10, 64))
att_all = np.zeros((10, 10))
val_all = np.zeros((10, 10))
M2 = np.zeros((10, 10))

if model == "tin":
    p = np.load('tin/p.npy')  # position embedding [431, 128]
    c = np.load("tin/c.npy")  # category embedding [801, 64]
    tar = c[674]
    ind = np.argpartition(cat, -10)[-10:]  # [10,]=T
    cw = c[ind]  # [10, 64]

    for i in range(10):
        ci = cw[i]
        print('category %d' % i)
        for j in range(1, 11):
            att[j - 1] = np.sum((ci + p[j][64:]) * (tar + p[0][64:]))  # 注意力交互
            print(np.linalg.norm(ci + p[j][64:]))
            val[j - 1] = (ci + p[j][64:]) * (tar + p[0][64:])  # 表征部分交互
            att_all[i][j - 1] = att[j - 1]
            val_all[i][j - 1] = np.linalg.norm(val[j - 1], ord=2)

elif model == "din":
    c = np.load("din/c.npy")  # category embedding [801, 64]
    tar = c[674]  # 给定目标广告类型为674, [64,]
    ind = np.argpartition(cat, -10)[-10:]  # [10,]=T
    cw = c[ind]  # [10, 64]

    for i in range(10):
        ci = cw[i]
        for j in range(1, 11):
            att[j - 1] = np.sum(ci * tar)  # 注意力交互
            val[j - 1] = ci  # 表征部分交互
            att_all[i][j - 1] = att[j - 1]
            val_all[i][j - 1] = np.linalg.norm(val[j - 1], ord=2)

elif model == "din_plus":
    c = np.load("din_plus/c.npy")  # category embedding [801, 64]
    tar = c[674]  # 给定目标广告类型为674, [64,]
    ind = np.argpartition(cat, -10)[-10:]  # [10,]=T
    cw = c[ind]  # [10, 64]

    for i in range(10):
        ci = cw[i]
        for j in range(1, 11):
            att[j - 1] = np.sum(ci * tar)  # 注意力交互
            val[j - 1] = ci * tar  # 表征部分交互
            att_all[i][j - 1] = att[j - 1]
            val_all[i][j - 1] = np.linalg.norm(val[j - 1], ord=2)


att_all = gauss_normalize(att_all)
att_all = softmax(att_all)
for i in range(10):
    for j in range(10):
        M2[i][j] = att_all[i][j] * val_all[i][j]
M2 = min_max_normalize(M2)

fig, ax = plt.subplots(figsize=(5, 5))
ind = [str(i + 1) for i in range(0, 10)]
sns.heatmap(
    pd.DataFrame(np.round(M2, 2), columns=ind, index=T),
    annot=True, xticklabels=True, yticklabels=True, square=True, cmap="Blues",
    cbar=False)

ax.set_title('Learned correlation given target category ID 674 ', fontsize=12)
ax.set_ylabel('Top-10 appeared history behaviors')
ax.set_xlabel('Target-relative Position')
plt.savefig("%s.pdf" % model, dpi=200, bbox_inches='tight')
plt.show()


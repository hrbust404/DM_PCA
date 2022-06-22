# -*- coding = utf-8 -*-
# @Time : 2022/6/20 15:54
# @Author : 李世伟

# 葡萄酒数据集+PCA
import matplotlib.pyplot as plt  # 画图工具
from sklearn import datasets
data = datasets.load_wine()
X = data['data']
y = data['target']

#利用PCA降维，降到二维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_p =pca.fit(X).transform(X)
explained_var_1 = pca.explained_variance_ratio_ # 获取贡献率
print(explained_var_1)
sun_bianji_1 = 0
for i in explained_var_1:
    sun_bianji_1 += i
print(sun_bianji_1)#输出累计方差贡献率约束80%
ax = plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
    plt.scatter(X_p[y == i, 0], X_p[y == i, 1], c=c, label=target_name)
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.title("wine")
plt.legend()
plt.show()



#标准化后做PCA
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit(X).transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=5)#降到5维
X_p =pca.fit(X).transform(X)
explained_var = pca.explained_variance_ratio_ # 获取贡献率
print(explained_var)
sun_bianji = 0
for i in explained_var:
    sun_bianji += i
print(sun_bianji)#输出累计方差贡献率约束80%
ax = plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], data.target_names):
    plt.scatter(X_p[y == i, 0], X_p[y == i, 1], c=c, label=target_name)
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.title("wine-standard-PCA")
plt.legend()
plt.show()
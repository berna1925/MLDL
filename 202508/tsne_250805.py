from sklearn.datasets import load_digits
digits = load_digits()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4)

for i in range(4) :
    ax[i].imshow(digits.images[i], cmap='viridis')
# plt.show()

X = digits.data
# shape 확인 결과 1797개의 8*8 사진
y = digits.target

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=0)
# 64차원 이미지 데이터를 PCA로 2차원 공간에 투영
X_tsne = tsne.fit_transform(X)

import matplotlib.patheffects as pe
import numpy as np
def plot_projection(x, colors) :
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    for i in range(len(np.unique(y))) :
        plt.scatter(x[colors == i, 0], x[colors == i, 1])

    for j in range(len(np.unique(y))) :
        x_text, y_text = np.median(x[colors == j, :], axis=0)
        # 표시할 데이터의 x, y좌표 값을 미리 따온 뒤 ax.text의 첫 두 인자로 활용
        # ax.text(x축 좌표, y축 좌표, 표시할 텍스트, kwargs)
        txt = ax.text(x_text, y_text, 'class' + str(j), fontsize=15)
        # matplotlib 하위 모듈 patheffects를 사용해 set_path_effects 메서드 활성화
        # 하위 모듈까지 다루기엔 뇌 용량이 모자라니 그때 그때 구글링해서 써먹어야
        txt.set_path_effects([
            pe.Stroke(linewidth=5, foreground='w'),
            pe.Normal()
        ])

plot_projection(X_tsne, y)
plt.show()
import numpy as np
import pandas as pd


class RegressionTree:
    def __init__(self, x, y, max_depth=12, batch_size=5, interval=5):
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self.min_batch = batch_size
        self.interval = interval
        self.Tree = 0

    def threshold(self, x, y):
        thresh = pd.DataFrame(columns=['feature', 'cutoff', 'ssr'])
        temp = pd.concat([x, y], axis=1)

        for feature in x.columns:
            cutoffs = pd.DataFrame(columns=['cutoff', 'ssr'])
            for i in range(0, x.shape[0] - self.interval):
                cutoff = np.mean(temp[feature][i:(i + self.interval)])
                left = temp[temp[feature] < cutoff]
                right = temp[temp[feature] >= cutoff]
                SSR = np.sum((left[self.y[0]] - np.mean(left[self.y[0]])) ** 2) + np.sum(
                    (right[self.y[0]] - np.mean(right[self.y[0]])) ** 2)
                cutoffs = cutoffs.append({'cutoff': cutoff, 'ssr': SSR}, ignore_index=True)

            t = cutoffs.sort_values(by='ssr').head(1)
            thresh = thresh.append({'cutoff': t['cutoff'].values[0],
                                    'ssr': t['ssr'].values[0],
                                    'feature': feature}, ignore_index=True)

        return thresh.sort_values(by='ssr').head(1)

    def splitter(self, data):
        feature, node, ssr = self.threshold(data[self.x], data[self.y]).values[0]
        left = data[data[feature] < node]
        right = data[data[feature] >= node]

        return left, right, node, feature

    def fit(self, data, depth=0):
        lSet, rSet, node, feature = self.splitter(data)
        Tree = {'node': node, 'feature': feature}

        if depth < self.max_depth and lSet.shape[0] >= self.min_batch:
            Tree['left'] = np.mean(lSet[self.y].values)
            Tree['right'] = self.fit(rSet, depth + 1)
        else:
            Tree['right'] = np.mean(rSet[self.y].values)

        self.Tree = Tree

        return self

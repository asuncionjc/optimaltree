import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from lsopt.tree import OptimalTreeClassifier

import graphviz

data_path = './data/TFG_dataset.csv'
# data_path = './data/Iris.csv'

# iris_data = pd.read_csv(data_path)
iris_data = pd.read_csv(data_path, sep=';', index_col=0)
# iris_data.drop(columns=["Id"], inplace=True)
X = iris_data.iloc[:, 0:2].to_numpy()
y = iris_data["labels"].to_numpy()
aa = 0

max_depth = 2  # 3
min_samples_leaf = 1
alpha = 0.5  # 0.01
time_limit = 5  # minute
mip_gap_tol = 0.5  # optimal gap percentage
mip_focus = 'balance'
mip_polish_time = None
warm_start = False
log_file = None

# Construct OCT classifier
oct_model = OptimalTreeClassifier(max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  alpha=alpha,
                                  criterion="gini",
                                  solver="gurobi",
                                  time_limit=time_limit,
                                  verbose=True,
                                  warm_start=warm_start,
                                  log_file=log_file,
                                  solver_options={'mip_cuts': 'auto',
                                                  'mip_gap_tol': mip_gap_tol,
                                                  'mip_focus': mip_focus,
                                                  'mip_polish_time': mip_polish_time
                                                  }
                                  )
aa = 0
oct_model.fit(X, y)
aa = 0

feature_names = iris_data.columns.values[:2]
class_names = ['1', '2']

dot_data = tree.export_graphviz(oct_model,
                                out_file=None,
                                feature_names=feature_names,
                                class_names=class_names,
                                label='all',
                                impurity=True,
                                node_ids=True,
                                filled=True,
                                rounded=True,
                                leaves_parallel=True,
                                special_characters=False)

graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render(filename='optimal_tree_iris', directory='', view=True)

aa = 0

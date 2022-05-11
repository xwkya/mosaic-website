import pickle
from strategies import AverageStrategyCosineFaiss

with open('name_to_index.pkl', 'rb') as f:
    name_to_index = pickle.load(f)

strat = AverageStrategyCosineFaiss(name_to_index, False)
with open('train_data/data_1.pkl', 'rb') as f:
    data=pickle.load(f)

with open('train_data/data_2.pkl', 'rb') as f:
    data2=pickle.load(f)

print(strat.find_best([data[0], data2[0]], get_dist=True))
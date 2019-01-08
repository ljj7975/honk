import numpy as np

class DataCollector:
    def __init__(self, name, unit):
        self.name = name
        self.unit = unit
        self.collection = np.array([]);

    def insert(self, data):
        self.collection = np.append(self.collection, data)

    def print_summary(self):
        print('< summary for {} ( unit : {} ) >'.format(self.name, self.unit))
        print('\tlength : ', len(self.collection))
        if len(self.collection) == 0:
            print('\tunable to compute other metrics because the array is empty')
            return
        print('\ttotal : ', np.sum(self.collection))
        print('\tminimum : ', np.min(self.collection))
        print('\tmaximum : ', np.max(self.collection))
        print('\taverage : ', np.mean(self.collection))
        print('\tP50 : ', np.percentile(self.collection, 50))
        print('\tP90 : ', np.percentile(self.collection, 90))
        print('\tP99 : ', np.percentile(self.collection, 99))

    def combine(self, data_collector):
        self.collection = np.add(self.collection, data_collector.collection);

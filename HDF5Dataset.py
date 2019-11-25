"""
Usage Example:
ds = HDF5Dataset(filename)
len(ds)                   # find length of the dataset
ds[10]                    # get 11th instance in the dataset
ds["start_hour"]          # get a column of start_hours of all instances 
ds[0]["start_hour"]       # get start_hour of the 1st instance in dataset
ds.X()                    # get the X matrix: n x d
ds.Y()                    # get the Y matrix: n x 1
ds.W()                    # get the weight matrix: n x 1
X, Y, W = ds()            # get all three matrices at once
ds.features()             # get the list of features
ds.num_of_features        # get the number of features 
"""

import h5py
from collections import namedtuple

Data = namedtuple("Data", ["start_hour", "date", "day_of_week", "isHoliday", 
                           "start_zone_latitude", "start_zone_longitude", 
                           "end_zone_latitude", "end_zone_longitude", 
                           "price", "distance", "ETA"])

class HDF5Dataset:

    def __init__(self, csv_filename):
        self.dataset, self.save_dir = self.parse_file(csv_filename)
        self.num_of_features = len(Data._fields) - 1

    def __len__(self):
        """Get length of the dataset"""
        return self.dataset.shape[0]

    def __getitem__(self, key):
        """Get single instance"""
        if type(key) is int:
            return Data._make(self.dataset[key])

        """Get single feature across all instances"""
        try:
            index = self.features().index(key)
            return self.dataset[:, index]
        except:
            print("Undefined key value")


    def features(self):
        """Get data fields"""
        return Data._fields

    def X(self):
        """Get X matrix: n instances x d features"""
        return self.dataset[:, : self.num_of_features]

    def Y(self):
        """Get Y matrix: n instances x label"""
        label_index = self.num_of_features
        return self.dataset[:, label_index]

    def W(self):
        """Get W matrix: n instances x weight (for weighted datapoints)"""
        pass

    def __call__(self):
        return self.X(), self.Y(), self.W()

    def parse_file(self, filename):
        """TODO(Eryn): parse raw csv file and return formatted hdf5 file (private method)"""
        pass

    


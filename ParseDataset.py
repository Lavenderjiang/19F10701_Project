import pandas as pd
import h5py
import numpy as np
from collections import namedtuple
import csv
import requests
from HDF5Dataset import HDF5Dataset
import time
import threading
from os import listdir
from os.path import isfile, join

program_dir = "/home/ubuntu/701project/"

# program_dir = "/home/erynqian/10701/19F10701_Project/"
hdf5_dir = program_dir + "HDF5s/"

csv_dir = program_dir + "testData/"
files = [f for f in listdir(csv_dir) if isfile(join(csv_dir, f))]

# Global Variables
# csv_dataset_sample = "/home/erynqian/10701/19F10701_Project/testData/sampled/first365.csv"
taxi_zone_lookup_chart = program_dir + "taxi_zone_lookup.csv"
datafields = ["VendorID","tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","PULocationID","DOLocationID","payment_type"]
DATA_LEN = 58
all_locations = []
distance_matrix = np.load( program_dir + "distance_matrix.npy")


# Load taxi zone lookup chart
with open(taxi_zone_lookup_chart) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first = 0
    for row in csv_reader:
        if first == 0:
            first += 1
        else:
            all_locations.append(row)
all_locations = all_locations[:-2]


# Parsed data object
class ParsedData:

    def __init__(self, row):
        self.row = row
    
    def start_hour(self):
        start_time = self.row[1].split(' ')[1]
        start_hour = start_time.split(':')[0]
        one_hot = [0] * 24
        one_hot[int(start_hour)] = 1
        return one_hot
    
    def month(self):
        """represent date as the kth day of the year; return k as int"""
        date = self.row[1].split(' ')[0]
        y,m,d = date.split('-')
        # days_in_feb = 29 if int(y) % 4 == 0 else 28
        # days_in_month = {1:31, 2:days_in_feb, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        # return sum([days_in_month[i] for i in range(1, int(m))]) + int(d)
        one_hot = [0] * 12
        one_hot[int(m)-1] = 1
        return one_hot

    def day_of_week(self):
        date = self.row[1].split(' ')[0]
        y,m,d = [int(i) for i in date.split('-')]
        t = [ 0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4 ] 
        y -= m < 3
        day_of_week = (( y + int(y / 4) - int(y / 100) + int(y / 400) + t[m - 1] + d) % 7) 
        one_hot = [0] * 7
        one_hot[day_of_week] = 1
        return one_hot

    def isHoliday(self):
        """    
        National holidays only
        reference: https://www.officeholidays.com/countries/usa/2017
                   https://www.officeholidays.com/countries/usa/2018
        """
        holidays2017 = [[1,2], [1,16], [5,29], [7,4], [9,4], [11,23], [12,25]]
        holidays2018 = [[1,1], [1,15], [5,28], [7,4], [9,3], [11,22], [12,25]]
        date = self.row[1].split(' ')[0]
        y,m,d = [int(i) for i in date.split('-')]
        holidays = holidays2017 if y == 2017 else holidays2018
        for h in holidays:
            if m == h[0] and d == h[1]:
                return [1]
        return [0]


    def start_and_end(self):
        """return 1 hot encoding of all start and end regions"""
        region_index = {"EWR": 0, "Queens": 1, "Bronx": 2, "Manhattan": 3, "Staten Island": 4, "Brooklyn": 5}
        pickup, dropoff = int(self.row[4]), int(self.row[5])
        PU_region, DO_region = all_locations[pickup-1][1], all_locations[dropoff-1][1]
        PU_index, DO_index = region_index[PU_region], region_index[DO_region]
        PU_1hot, DO_1hot = [0] * 6, [0] * 6
        PU_1hot[PU_index] = 1
        DO_1hot[DO_index] = 1
        return PU_1hot + DO_1hot

    def distance(self):
        """return distance in miles"""
        loc1, loc2 = int(self.row[4]), int(self.row[5])
        if loc1 < loc2:
            return [distance_matrix[loc1-1, loc2-1]]
        else:
            return [distance_matrix[loc2-1, loc1-1]]

    def ETA(self):
        """return ETA in min"""
        start = self.row[1].split(' ')[1].split(':')
        end = self.row[2].split(' ')[1].split(':')
        hour_diff = (int(end[0]) - int(start[0])) % 24 
        min_diff = int(end[1]) - int(start[1])
        sec_diff = int(end[2]) - int(start[2])
        return [hour_diff * 60 + min_diff + round(sec_diff / 60, 3)]

    def data(self):
        # data length = 24 + 12 + 7 + 1 + 12 + 1 + 1 = 58
        return  self.start_hour() + self.month() + self.day_of_week() + self.isHoliday() + \
                self.start_and_end() + self.distance() + self.ETA()


# TEST
# entry = ['1', '2017-07-30 00:27:25', '2017-07-30 00:39:09', '1', '170', '48', '1']
# d = ParsedData(entry)
# d = d.data()
# print("start hour", d[:24])
# print("month", d[24:36])
# print("dayofweek", d[36:43])
# print("isholiday", d[43])
# print("start/end", d[44:56])
# print("distance", d[56])
# print("ETA", d[57])

# Determine if data instance is valid
def isValid(row):

    # Pickup or Dropoff location is unknown
    if (int(row[4]) == 264 or int(row[4]) == 265 or int(row[5]) == 264 or int(row[5]) == 265):
        # print("Pickup or Dropoff location is unknown")
        return False

    # Pickup or Dropoff location is the same
    if int(row[4]) == int(row[5]):
        # print("Pickup or Dropoff location is the same")
        return False
    
    data = ParsedData(row)
    duration, distance = data.ETA()[0], data.distance()[0]

    # Abnormal trip duration
    if duration == 0:
        return False
    date = row[1].split(' ')[0]
    y0,m0,d0 = date.split('-')
    date = row[2].split(' ')[0]
    y1,m1,d1 = date.split('-')
    if (y0 != y1 or m0 != m1 or d0 != d1) : # if not on the same day
        start_time = row[1].split(' ')[1]
        start_hour = int(start_time.split(':')[0])
        end_time = row[2].split(' ')[1]
        end_hour = int(end_time.split(':')[0])
        if end_hour >= start_hour or duration > 300:
            # print("Abnormal trip duration")
            return False

    # Abnormal average driving speed
    mph = distance / (duration / 60)
    if mph >= 56:
        # print("Abnormal average driving speed")
        return False

    return True
    
# TEST
# rows = [["1","2017-07-30 00:20:56","2017-07-30 00:48:20","1","138","265","2"],
#         ["2","2017-07-30 00:43:37","2017-07-30 00:51:35","6","264","264","2"],
#         ["2","2017-07-30 00:15:58","2017-07-30 00:34:34","1","90","79","1"],
#         ["2","2017-07-30 00:09:23","2017-07-31 00:20:53","1","256","37","2"]]

# for r in rows:
#     print(isValid(r))


# parse csv
def parse_csv(filename, start_time):
    print("processing", filename.split('/')[-1], "...")
    df = pd.read_csv(csv_dir + filename, sep=',')
    file_len = len(df)
    print("file length:", file_len)

    hdf5_filename = hdf5_dir + filename.split('.')[0] + '.hdf5'
    valid_rows = 0
    with h5py.File(hdf5_filename, "w") as f:
        dset = f.create_dataset("mydataset", (file_len, DATA_LEN), dtype='f', chunks=True)
        parsedData = 0

        with open(csv_dir + filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            r = -1
            for row in csv_reader:
                if r == -1: #skip first line
                    r += 1
                    continue
                if not isValid(row):
                    continue
                else:
                    parsedData = ParsedData(row)
                    dset[r, :] = parsedData.data()
                    r += 1
                    valid_rows += 1
                if r % 10000 == 0:
                    print(r, "rows processed.")
                    
        print("Total valid rows:", valid_rows)
        dset.resize((valid_rows, DATA_LEN))
    
    now = time.time()
    print("Done processing", filename.split('/')[-1])
    print('Elapsed', (now - start_time) / 60, 'min')


def thread(csv_file, start_time):
    parse_csv(csv_file, start_time)

    # TEST
    hdf5_filename = hdf5_dir + csv_file.split('.')[0] + '.hdf5'
    dset = HDF5Dataset(hdf5_filename)
    print(len(dset))
    print(dset[3])


start_time = time.time()
for i in range(0, len(files), 16):
    threads = []
    for csv_file in files[i: i+16]:
        t = threading.Thread(target=thread, args=(csv_file, start_time,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()

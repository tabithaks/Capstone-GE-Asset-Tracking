import pandas as pd
import numpy as np
import datetime
from DistanceMetrics import EditDistance
from DistanceMetrics import dtw
from DistanceMetrics import Frechet

def preprop(df):
    scaler = MinMaxScaler()
    df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:])
    return(df)

def createtraj(data, size):
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["datetime"] = data["datetime"].apply(lambda v: v.timestamp())
    unpro = data.drop(["taxiid"], axis = 1)
    unpro = unpro.values.tolist()
    preprodata = preprop(data)
    preprodata = preprodata.values.tolist()
    i = 0
    n = len(data)
    preprotraj = []
    traj =[]
    while i + size < n:
        #print(data.iloc[i:i+size,:])
        traj.append(unpro[i:i+size])
        preprotraj.append(preprodata[i:i+size])
        i += size
    return (np.asarray(preprotraj), np.asarray(traj))

def createtrajdtw():

    np.random.seed(1000)
    num_selected = 500
    taxi_id_selected = np.random.choice(10357, num_selected, replace=False) + 1
    taxi_id_selected.sort()

    date_selected_orig = '2008-02-02'
    total_len_500_taxis_spec_day = 0
    list_500_taxis_spec_day = []
    colnames = ['taxi_id', 'date_time', 'longitude', 'latitude']
    for id in taxi_id_selected:
        date_selected = date_selected_orig

        filepath = "/Users/tabithasugumar/Documents/Capstone/Data/Data/" + str(id) + ".txt"
        data = pd.read_csv(filepath, sep=',', header=None, names=colnames, parse_dates=['date_time'])
        data = data.drop_duplicates()
        data = data.set_index('date_time')

        if data.index.empty:
            #print("{}.txt is empty.".format(id))
            continue

        first_date_available = data.index[0]
        first_date_available = first_date_available.strftime('%Y-%m-%d')
        if date_selected != first_date_available:
            date_selected = first_date_available
            #print("Taxi id {}\tfirst_date_available: {}".format(id, first_date_available))


        resampled_data = data.resample('5T').mean().interpolate('linear')
        resampled_data_spec_day = resampled_data.loc[date_selected]
        resampled_data_spec_day.reset_index(inplace=True)
        resampled_data_lon_lat_spec_day = resampled_data_spec_day[["date_time",'longitude', 'latitude']]
        resampled_data_lon_lat_spec_day = resampled_data_lon_lat_spec_day.values
        resampled_data_lon_lat_spec_day = resampled_data_lon_lat_spec_day.tolist()
        total_len_500_taxis_spec_day += len(resampled_data_lon_lat_spec_day)

        list_500_taxis_spec_day.append(resampled_data_lon_lat_spec_day)

    date_selected = date_selected_orig

    resampled_data = data.resample('5T').mean().interpolate('linear')
    resampled_data_spec_day = resampled_data.loc[date_selected]
    resampled_data_spec_day.reset_index(inplace=True)
    resampled_data_lon_lat_spec_day = resampled_data_spec_day[["date_time",'longitude', 'latitude']]
    resampled_data_lon_lat_spec_day = resampled_data_lon_lat_spec_day.values
    resampled_data_lon_lat_spec_day = resampled_data_lon_lat_spec_day.tolist()

    total_len_500_taxis_spec_day += len(resampled_data_lon_lat_spec_day)
    #print("\nTotal length of 500 taxi series on {}: {}\n".format(date_selected_orig, total_len_500_taxis_spec_day))

    list_500_taxis_spec_day.append(resampled_data_lon_lat_spec_day)
    X = np.asarray(list_500_taxis_spec_day)

    return X

def createtrajfrachet (df):
    duplicates = df.duplicated()
    df2 = df[~duplicates].copy()

    conditions = [(np.isnan(df2['time_interval'])),
                  (df2['time_interval'] >= 480),
                  (df2['time_interval'] < 480)]

    choices = [1, 1, 0]

    df2['traj_ind'] = np.select(conditions, choices, default=1)
    df2['traj_id'] = df2.groupby('taxi_id')['traj_ind'].cumsum()

    # copy dataframe of only trajectories
    dftraj = df2[['longitude', 'latitude', 'trajectory_id']].copy()
    dftraj.index = dftraj.trajectory_id
    dftraj = dftraj.drop(['trajectory_id'], axis=1)

    # from the trajectories df exclude 1 point trajectories
    id_count = dftraj.groupby('trajectory_id').count()
    id_1point = id_count[id_count.longitude == 1].index

    dftraj_2 = dftraj.loc[dftraj.index.difference(id_1point), :]
    print (dftraj_2)

# Main clustering algorithm
def tcluster(lines, eps, minlns, distance, c = 0):
    # Create array to hold cluster values
    Cluster = np.zeros(len(lines))
    # Set fist cluster to one
    cid = 1
    # For each trajectory
    for i in range(len(lines)):
        line = lines[i]

        # if the trajectory is unclassified, classify it or designate it as noise
        if Cluster[i] == 0:
            # Get neighbors of a trajectory
            N = neighbor(line, eps, lines, distance, c)
            # Check if trajectory has minimum number of neighbors
            if (len(N) >= minlns):
                # Give neighbors the cluster id
                for n in N:
                    Cluster[n] = cid
                queue = [n for n in N if n != i]
                # Expand cluster based on neighbors of neighbor
                Cluster = expandcluster(queue, cid, eps, minlns, Cluster, lines, distance, c)
                # Move to next cluster
                cid += 1
            # If line is unclassified and without enough neighbors to be it's own cluster designate it as noise
            else:
                Cluster[i] = -1
    # Return list of cluster designations, with indices corresponding to trajectories' indices in lines array
    return Cluster


# Expland cluster based on neighbors of neighbor
def expandcluster(Q, cid, eps, minlns, Cluster, lines, distance, c):
    while len(Q) != 0:
        M = Q.pop(0)
        line = lines[M]
        N = neighbor(line, eps, lines, distance, c)
        if (len(N) >= minlns):
            for n in N:
                if Cluster[n] in [0, -1]:
                    if Cluster[n] == 0:
                        Q.append(n)
                    Cluster[n] = cid
        return Cluster


# Get trajectory neighbor
def neighbor(trajectory, eps, lines, distance, c):
    if distance == "edit":
        distances = np.asarray([EditDistance.editdist(trajectory, t, c) for t in lines])
    elif distance == "dtw":
        distances = np.asarray([dtw.dtw(trajectory, t)[1] for t in lines])
    elif distance == "frechet":
        distances = np.asarray([Frechet.discrete_frechet(trajectory, t) for t in lines])
    neighbor_indices = np.where(distances <= eps)[0]
    return neighbor_indices


# # Define distance as sum of differences between two points
# def dist(trajectory, lines):
#     diff = np.absolute(np.subtract(lines, trajectory))
#     distances = np.sum(diff, axis=1)
#     return distances

# clusters = tcluster(lines, .15, 60)
# print("Lines", lines)
# print(clusters)

# vt.visclust(lines, clusters)
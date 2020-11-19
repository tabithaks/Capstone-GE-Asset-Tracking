import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def visualizeclust(lines, clusters, noise = True):
    #Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    #Get number of clusters
    nclust = np.unique(clusters)

    #For each cluster
    for n in nclust:
        #Don't graph noise if specified not to
        if (noise == False and n == -1):
            continue
        #Get lines that are part of that cluster
        lines_index = lines[np.nonzero(clusters == n)]
        print("lines_index", lines_index)
        #Format lines into lists of tuples of points
        linesplt = []
        for traj in lines_index:
            singleline = []
            for point in traj:
                singleline.append(tuple([point[0], point[1]]))
            linesplt.append(singleline)
        print(linesplt)
        #plot points in cluster
        x = [i[0] for j in linesplt for i in j]
        y = [i[1] for j in linesplt for i in j]
        p = ax.scatter(x, y, s=10, label="cluster " + str(n))

        #Get color of points
        color = p.get_facecolor()

        #Create and plot all lines in cluster
        lc = mc.LineCollection(linesplt, linewidths=2, color=color)
        ax.add_collection(lc)

    #Set labels and display plot
    ax.set_title("Taxi Trajectories Clustered by Trajectory")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend()
    plt.show()

# Takes the lines array and cluster output and visualizes
def visclust3D(lines, clusters, noise=True):
    # Create plot
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    # Get number of clusters
    nclust = np.unique(clusters)

    # For each cluster
    for n in nclust:
        # Don't graph noise if specified not to
        if (noise == False and n == -1):
            continue
        # Get lines that are part of that cluster
        lines_index = lines[np.nonzero(clusters == n)]
        # hours_index = hours[np.nonzero(clusters==n)]

        # Format lines into lists of tuples of points
        linesplt = []
        for traj in lines_index:
            singleline = []
            for point in traj:
                singleline.append(tuple([point[0], point[1], point[2]]))
            linesplt.append(singleline)

        # plot points in cluster
        x = [i[0] for j in linesplt for i in j]
        y = [i[1] for j in linesplt for i in j]
        z = [i[2] for j in linesplt for i in j]
        p = ax.scatter3D(x, y, z, s=10, label="cluster " + str(n))

        # Get color of points
        color = p.get_facecolor()
        # Create and plot all lines in cluster
        lc = Line3DCollection(linesplt, linewidths=2, color=color)
        ax.add_collection3d(lc)

    # Set labels and display plot
    ax.set_title("Taxi Trajectories Clustered by Trajectory")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # plt.legend()
    plt.show()
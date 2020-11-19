import pandas as pd
import numpy as np
import math as m
import datetime
from sklearn.preprocessing import MinMaxScaler

# Get rid of points taken too close together in time
def removered(t):
    torem = []
    for i in range(len(t) - 1):
        if ((t[i + 1][1] - t[i][1]).total_seconds()) <= 60:
            torem.append(i)
    return ([t[x] for x in range(len(t)) if x not in torem])


# Cost of deleting a point
def costdel(t, k, c):
    # Get all x, y, and time terms not in point
    x = np.array([t[i][2] for i in range(len(t)) if i != k])
    y = np.array([t[i][3] for i in range(len(t)) if i != k])
    ti = np.array([t[i][1] for i in range(len(t)) if i != k])
    n = len(t)

    # Calculate cost of displacement from deletion
    xpart = (((np.sum(x) + t[k][2]) / n) - (np.sum(x) / (n - 1))) ** 2
    ypart = (((np.sum(y) + t[k][3]) / n) - (np.sum(y) / (n - 1))) ** 2
    tpart = (((np.sum(ti) + t[k][1]) / n) - (np.sum(ti) / (n - 1))) ** 2
    cost = (1 - c) * (xpart + ypart) + c * (tpart)
    #     print("costdel xpart: ", xpart)
    #     print("costdel ypart: ", ypart)
    #     print("costdel tpart: ", tpart)
    #     print("costdel: ", cost)
    # Return cost
    return (m.sqrt(cost))

def costins(t, p, c):
    # Get all x, y, and time terms in the trajectory
    x = np.array([t[i][2] for i in range(len(t))])
    y = np.array([t[i][3] for i in range(len(t))])
    ti = np.array([t[i][1] for i in range(len(t))])
    n = len(t)

    # Calculate cost of insertion
    xpart = (((np.sum(x)) / n) - ((np.sum(x) + p[2]) / (n + 1))) ** 2
    ypart = ((np.sum(y) / n) - ((np.sum(y) + p[3]) / (n + 1))) ** 2
    tpart = ((np.sum(ti) / n) - ((np.sum(ti) + p[1]) / (n + 1))) ** 2
    cost = (1 - c) * (xpart + ypart) + c * (tpart)
    #     print("costins xpart: ", xpart)
    #     print("costins ypart: ", ypart)
    #     print("costins tpart: ", tpart)
    #     print("costins: ", cost)
    # Return cost
    return (m.sqrt(cost))


def costrep(t, k, p, c):
    # Get all x, y, and time terms not in point to be replaced
    x = np.array([t[i][2] for i in range(len(t)) if i != k])
    y = np.array([t[i][3] for i in range(len(t)) if i != k])
    ti = np.array([t[i][1] for i in range(len(t)) if i != k])
    n = len(t)

    # Calculate cost of displacement from replacement
    xpart = (((np.sum(x) + t[k][2]) / n) - ((np.sum(x) + p[2]) / n)) ** 2
    ypart = (((np.sum(y) + t[k][3]) / n) - ((np.sum(y) + p[3]) / n)) ** 2
    tpart = (((np.sum(ti) + t[k][1]) / n) - ((np.sum(ti) + p[1]) / n)) ** 2
    cost = (1 - c) * (xpart + ypart) + (c * tpart)
    #     print("costrep xpart: ", xpart)
    #     print("costrep ypart: ", ypart)
    #     print("costrep tpart: ", tpart)
    #     print("costrep: ", cost)
    # Return cost
    return (m.sqrt(cost))


# t's are formatted as [[id,x,y,time],[id,x,y,time],....] where time is pandas datetime
def editdist(t1, t2, c):
    # Revised trajectories
    # t1 = removered(t1)
    # t2 = removered(t2)

    # Get trajectory lengths
    n1 = len(t1)
    n2 = len(t2)
    #     print("n1: ", n1)
    #     print("n2: ", n2)

    # Create table to save prior solutions
    table = [[0 for i in range(n2 + 1)] for j in range(n1 + 1)]

    # Note, should t1 be updated before calculating cost of next insert/del?
    # Fill in table where i represents number of points in t1 and j the number of points in t2
    for i in range(n1 + 1):
        for j in range(n2 + 1):

            # Fill first row of table
            if i == 0:
                table[i][j] = sum([costins(t1, p, c) for p in t2[:j]])
            #                 print("table value with i = " +str(i) + "and j = " + str(j) + " : ", table[i][j])
            # Fill first column of table
            elif j == 0:
                table[i][j] = sum([costdel(t1, k, c) for k in range(i)])
            #                 print("table value with i = " +str(i) + "and j = " + str(j) + " : ", table[i][j])

            # Keep cost the same if points are the same
            elif t1[i - 1].all == t2[j - 1].all:
                table[i][j] = table[i - 1][j - 1]
            #                 print("table value with i = " +str(i) + "and j = " + str(j) + " : ", table[i][j])

            # Calculate whether insertion, deletion, or replacement costs the most, and save whichever one is least expensive
            else:
                valins = table[i][j - 1] + costins(t1, t2[j - 1], c)
                valdel = table[i - 1][j] + costdel(t1, i - 1, c)
                valrep = table[i - 1][j - 1] + costrep(t1, i - 1, t2[j - 1], c)
                #                 print("valins: ", valins)
                #                 print("valdel: ", valdel)
                #                 print("valrep: ", valrep)

                if (valins <= valdel) and (valins <= valrep):
                    table[i][j] = valins
                #                     print("valins in if: ", valins)
                #                     print("table value with i = " +str(i) + "and j = " + str(j) + " : ", table[i][j])

                elif (valdel <= valins) and (valdel <= valrep):
                    table[i][j] = valdel
                #                     print("valdel in if: ", valdel)
                #                     print("table value with i = " +str(i) + "and j = " + str(j) + " : ", table[i][j])

                elif (valrep <= valdel) and (valrep <= valins):
                    table[i][j] = valrep
    #                     print("valrep in if: ", valrep)
    #                     print("table value with i = " +str(i) + "and j = " + str(j) + " : ", table[i][j])

    return table[n1][n2]


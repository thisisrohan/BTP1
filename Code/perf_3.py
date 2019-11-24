import numpy as np
from Code.test5 import Delaunay2D as Del
from scipy.spatial import Delaunay
import triangle as tr
import time
import pandas as pd

if __name__ == "__main__":
    import sys
    import os
    lim = int(sys.argv[1])

    points = []
    for i in np.arange(1, lim+1):
        if i < 8:
            if i > 1:
                points.append(np.random.rand(int(2*0.2*(10**i))))
                points.append(np.random.rand(int(2*0.5*(10**i))))
            points.append(np.random.rand(2*(10**i)))
        if i == 8:
            points.append(np.random.rand(int(2*0.2*(10**i))))

    time_t5 = []
    time_tr = []
    time_scipy = []

    temp = np.random.rand(2*10)
    tempDT = Del(temp)
    tempDT.makeDT()
    del tempDT

    len_points = len(points)

    for j in range(3):
        print("------------------------------ RUN " + str(j+1) + " ------------------------------")
        for i in range(len_points):
            print("------------ " + str(int(0.5*len(points[i]))) + " points ------------")

            temp1 = points[i].copy().reshape((int(0.5*len(points[i])), 2))

            print("--- Python binding for Triangle ---")
            tri = {"vertices": temp1}
            start = time.time()
            D3 = tr.triangulate(tri, opts='i')
            end = time.time()
            print("Time taken to make the triangulation : " + str(end-start) + " s.")
            time_tr.append(end-start)
            del D3
            del tri

            if len(temp1) <= 10**7:
                print("--- scipy.spatial.Delaunay (QHull) ---")
                start = time.time()
                tri = Delaunay(temp1)
                end = time.time()
                time_scipy.append(end-start)
                print("Time taken to make the triangulation : " + str(end-start) + " s.")
                del tri

            del temp1

            print("--- test5 ---")
            temp2 = points[i].copy()
            start = time.time()
            D2 = Del(temp2)
            D2.makeDT()
            end = time.time()
            print("Time taken to make the triangulation : " + str(end-start) + " s.")
            time_t5.append(end-start)
            del D2
            del temp2

    for i in range(len_points):
        temp_tr = min(time_tr[i], time_tr[len_points+i], time_tr[2*len_points+i])
        time_tr.append(temp_tr)

        temp_scipy = min(time_scipy[i], time_scipy[len_points+i], time_scipy[2*len_points+i])
        time_scipy.append(temp_scipy)

        temp_t5 = min(time_t5[i], time_t5[len_points+i], time_t5[2*len_points+i])
        time_t5.append(temp_t5)

    #plt.show()

    f = open("result_5.txt", "w")

    f.write("time_tr \n")
    f.write(str(time_tr) + " \n")


    f.write("time_scipy \n")
    f.write(str(time_scipy) + " \n")


    f.write("time_t5 \n")
    f.write(str(time_t5) + " \n")

    df = pd.DataFrame({"scipy":time_scipy, "triangle":time_tr, "test5":time_t5})
    df.to_csv('times_result_6.csv')


    x = np.empty(len_points)
    x[1::3] = np.array([0.2*(10**i) for i in range(2, lim+1)])
    x[2::3] = np.array([0.5*(10**i) for i in range(2, lim+1)])
    x[0::3] = np.array([10**i for i in range(1, lim+1)])

    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    plt.loglog(x, time_tr[3*len_points:], '--', color='C1', linewidth=0.75, label='_nolegend_')
    plt.loglog(x, time_tr[3*len_points:], '.', color='C1')

    plt.loglog(x[0:-1], time_scipy[3*len_points:], '--', color='C2', linewidth=0.75, label='_nolegend_')
    plt.loglog(x[0:-1], time_scipy[3*len_points:], '.', color='C2')

    plt.loglog(x, time_t5[3*len_points:], '--', color='C3', linewidth=0.75, label='_nolegend_')
    plt.loglog(x, time_t5[3*len_points:], '.', color='C3')


    plt.legend([
        "Python Binding for Shewchuk's Triangle",
        "scipy.spatial.Delaunay (QHull)",
        "test5 with BRIO",
    ])

    plt.xlabel("Number of Points", size=15)
    plt.ylabel(r"Running Time ($s$)", size=15)

    plt.savefig("result_6.png", dpi=300, bbox="tight")
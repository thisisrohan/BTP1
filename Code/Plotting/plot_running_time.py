import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn")

df = pd.read_csv('/home/rohan/Documents/Sem 7/BTP - I/Jupyter Notebooks & Code/times_result_6.csv')

triangle_times = np.array(df['triangle'])
scipy_times = np.array(df['scipy'])
test5_times = np.array(df['test5'])

num = int(len(triangle_times)/3)

triangle_best = []
scipy_best = []
test5_best = []
for i in range(num):
    triangle_best.append(min(triangle_times[i], triangle_times[i+num], triangle_times[i+2*num]))
    test5_best.append(min(test5_times[i], test5_times[i+num], test5_times[i+2*num]))
    if i <= num-2:
        scipy_best.append(min(scipy_times[i], scipy_times[i+num], scipy_times[i+2*num]))

triangle_best = np.array(triangle_best)
scipy_best = np.array(scipy_best)
test5_best = np.array(test5_best)

test5_v_triangle = test5_best/triangle_best
test5_v_scipy = test5_best[0:-1]/scipy_best


x = np.empty(num)
x[0::3] = np.array([10**i for i in range(1, 8)])
x[1::3] = np.array([2*(10**i) for i in range(1, 8)])
x[2::3] = np.array([5*(10**i) for i in range(1, 7)])


plt.loglog(x, triangle_best, '--', color='C0', linewidth=0.75, label='_nolegend_')
plt.loglog(x, triangle_best, '.', color='C0')

plt.loglog(x[0:-1], scipy_best, '--', color='C1', linewidth=0.75, label='_nolegend_')
plt.loglog(x[0:-1], scipy_best, '.', color='C1')

plt.loglog(x, test5_best, '--', color='C2', linewidth=0.75, label='_nolegend_')
plt.loglog(x, test5_best, '.', color='C2')

plt.grid(True)

plt.xlabel(r"Number of Points")
plt.ylabel(r"Running Time ($s$)")

plt.legend([
    "Python binding for Schewchuk's Triangle (incremental mode)",
    "scipy.spatial.Delaunay (QHull)",
    "test5"
])

plt.title("Running times of the different triangulators", size=15)

plt.savefig("running_times.png", dpi=300, bbox_inches='tight')
# plt.show()




plt.clf()

plt.axhline(y=1, color='k', linewidth=0.75, label='_nolegend_')

plt.semilogx(x, test5_v_triangle, '--', color='C0', linewidth=0.75, label='_nolegend_')
plt.semilogx(x, test5_v_triangle, '.', color='C0')

plt.semilogx(x[0:-1], test5_v_scipy, '--', color='C1', linewidth=0.75, label='_nolegend_')
plt.semilogx(x[0:-1], test5_v_scipy, '.', color='C1')

plt.legend([
    "Factor w.r.t. Schewchuk's Triangle",
    "Factor w.r.t. QHull"
])

plt.ylim(0, 4.2)

#xticks = np.array([10])

plt.xlabel("Number of Points")
plt.ylabel("Factor of Running Time")

plt.title("Factor of running time of test5 w.r.t. other triangulators", size=15)

plt.savefig("factors.png", dpi=300, bbox_inches='tight')
# plt.show()
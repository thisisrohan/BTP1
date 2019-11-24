import numpy as np
import matplotlib.pyplot as plt
import test5 as t5

points = np.random.rand(2*10)

DT = t5.Delaunay2D(points)
DT.makeDT()
fig = DT.plotDT()

plt.plot(points[0::2], points[1::2], 'o', color='brown')

plt.xticks([])
plt.yticks([])

# plt.show()
plt.savefig('fig2.png', dpi=300, bbox_inches='tight')
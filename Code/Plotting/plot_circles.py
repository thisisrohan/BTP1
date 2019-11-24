import numpy as np
import matplotlib.pyplot as plt

#plt.style.use('seaborn')

def plot_circle(r, cx, cy):
    points = np.array([[cx+r*np.cos(2*np.pi*i/100), cy+r*np.sin(2*np.pi*i/100)] for i in range(101)])
    points = np.append(points[:, 0], points[:, 1]).reshape(2, 101)
    return points

def plot_angle(m):
    theta = np.arctan2((m[1][1] - m[2][1]), (m[1][0] - m[2][0]))
    points = np.array([[m[2][0]+0.25*np.cos(theta + m[0]*i/100), m[2][1]+0.25*np.sin(theta + m[0]*i/100)] for i in range(100)])
    # points = np.append([[m[1][0], m[1][1]]], points, axis=0)
    points = np.append(points[:, 0], points[:, 1]).reshape(2, 100)
    return points

def find_circumcenter_and_radius(aa, bb, cc):
    a_x = aa[0]
    a_y = aa[1]
    b_x = bb[0]
    b_y = bb[1]
    c_x = cc[0]
    c_y = cc[1]
    a_sq = (b_x-c_x)*(b_x-c_x)+(b_y-c_y)*(b_y-c_y)
    a = np.sqrt(a_sq)
    b_sq = (c_x-a_x)*(c_x-a_x)+(c_y-a_y)*(c_y-a_y)
    b = np.sqrt(b_sq)
    c_sq = (a_x-b_x)*(a_x-b_x)+(a_y-b_y)*(a_y-b_y)
    c = np.sqrt(c_sq)
    A = np.arccos((b_sq+c_sq-a_sq)/(2*b*c))
    B = np.arccos((c_sq+a_sq-b_sq)/(2*c*a))
    C = np.arccos((a_sq+b_sq-c_sq)/(2*a*b))
    temp = np.sin(2*A) + np.sin(2*B) + np.sin(2*C)
    circumcenter_x = (a_x*np.sin(2*A)+b_x*np.sin(2*B)+c_x*np.sin(2*C))/temp
    circumcenter_y = (a_y*np.sin(2*A)+b_y*np.sin(2*B)+c_y*np.sin(2*C))/temp
    radius1 = ((circumcenter_x-a_x)**2+(circumcenter_y-a_y)**2)**0.5
    radius2 = ((circumcenter_x-b_x)**2+(circumcenter_y-b_y)**2)**0.5
    radius3 = ((circumcenter_x-c_x)**2+(circumcenter_y-c_y)**2)**0.5
    radius = (radius1+radius2+radius3)/3

    min_angle = min(A, B, C)
    if min_angle == A:
        min_angle = [A, bb, aa, cc]
    if min_angle == B:
        min_angle = [B, cc, bb, aa]
    if min_angle == C:
        min_angle = [C, aa, cc, bb]

    return circumcenter_x, circumcenter_y, radius, min_angle

points = np.array([
    [0, 0],
    [1, 0],
    [1.2, 1.5],
    [0, 0.85]
])

### Triangulation 1
# 0-1-2, 0-2-3

cx1, cy1, r1, m1 = find_circumcenter_and_radius(
    points[0],
    points[1],
    points[2]
)

c1 = plot_circle(r1, cx1, cy1)

cx2, cy2, r2, m2 = find_circumcenter_and_radius(
    points[0],
    points[2],
    points[3]
)

if m1[0] < m2[0]:
    m = m1
else:
    m = m2

angle = plot_angle(m)

c2 = plot_circle(r2, cx2, cy2)

plt.clf()

plt.plot(c1[0], c1[1], '--', color='C0', linewidth=0.75)

plt.plot(c2[0], c2[1], '--', color='C1', linewidth=0.75)

plt.plot(angle[0], angle[1], '-', color='k')

plt.triplot(
    points[:, 0],
    points[:, 1],
    [[0, 1, 2], [3, 0, 2]],
    color='C3'
)

plt.plot(points[:, 0], points[:, 1], 'o', color='brown')

plt.axis('equal')

# plt.ylim(-1, 2.2)
plt.xlim(-0.6, 2.1)

plt.xticks([])
plt.yticks([])

# plt.show()

plt.savefig('not_del.png', dpi=300, bbox_inches='tight')

plt.clf()

### Triangulation 2
# 0-1-3, 1-2-3

cx1, cy1, r1, m1 = find_circumcenter_and_radius(
    points[0],
    points[1],
    points[3]
)

c1 = plot_circle(r1, cx1, cy1)

cx2, cy2, r2, m2 = find_circumcenter_and_radius(
    points[1],
    points[2],
    points[3]
)

c2 = plot_circle(r2, cx2, cy2)

if m1[0] < m2[0]:
    m = m1
else:
    m = m2

angle = plot_angle(m)

plt.plot(angle[0], angle[1], '-', color='k')

plt.plot(c1[0], c1[1], '--', color='C0', linewidth=0.75)

plt.plot(c2[0], c2[1], '--', color='C1', linewidth=0.75)

plt.triplot(
    points[:, 0],
    points[:, 1],
    [[0, 1, 3], [1, 2, 3]],
    color='C3'
)

plt.plot(points[:, 0], points[:, 1], 'o', color='brown')

plt.axis('equal')

# plt.ylim(-1, 2.2)
plt.xlim(-0.6, 2.1)

plt.xticks([])
plt.yticks([])

# plt.show()
plt.savefig('del.png', dpi=300, bbox_inches='tight')
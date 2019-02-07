import numpy as np


def quat_matrix(pt1, pt2):
    """ build quaternion matrix from pt1 and pt2 """
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    for pt in [pt1, pt2]:
        if len(pt.shape) != 1 or pt.shape[0] != 3:
            raise ValueError('Arguments should be 3-element vectors')

    xm, ym, zm = tuple(pt1 - pt2)
    xp, yp, zp = tuple(pt1 + pt2)

    matrix = np.array([
        [xm*xm + ym*ym + zm*zm,
         yp*zm - ym*zp,
         xm*zp - xp*zm,
         xp*ym - xm*yp],

        [yp*zm - ym*zp,
         yp*yp + zp*zp + xm*xm,
         xm*ym - xp*yp,
         xm*zm - xp*zp],

        [xm*zp - xp*zm,
         xm*ym - xp*yp,
         xp*xp + zp*zp + ym*ym,
         ym*zm - yp*zp],

        [xp*ym - xm*yp,
         xm*zm - xp*zp,
         ym*zm - yp*zp,
         xp*xp + yp*yp + zm*zm]
    ])
    return matrix

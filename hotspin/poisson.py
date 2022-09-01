import math

import numpy as np

from scipy.spatial.distance import pdist
from typing import Callable


# TODO: parallel Poisson?
# from scipy.spatial import KDTree
# T = scipy.spatial.KDtree
# def ThrowSample():
#     pass
# def Subdivide():
#     pass


def PoissonGrid(nx: int, ny: int, k=10):
    ''' Selects non-adjacent cells (neither diagonally nor orthogonally) in an <nx> x <ny> periodic grid.
        The cell at (0, 0) is always chosen: additional randomization should be performed by _select_grid().
        (this is basically an optimized version of poisson_disc_samples() for choosing supercells specifically)
        Inspired by https://github.com/emulbreh/bridson/blob/master/bridson/__init__.py.
    '''
    grid = np.zeros((nx, ny))
    annulus = [(-2, -1), (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2)] # Chess horse
    # annulus += [(-2, 0), (0, 2), (2, 0), (0, -2)] # Adding these increases grid-like bias in periodogram but gains 10% more samples
    annulus += [(-2, -2), (-2, 2), (2, 2), (2, -2)] # Adding this gives more euclidean grid sampling, without it's more hexagonal-like
    annulus += [(-3, -1), (-3, 0), (-3, 1), (-1, 3), (0, 3), (1, 3), (3, -1), (3, 0), (3, 1), (-1, -3), (0, -3), (1, -3)] # Adding this smoothens periodogram but reduces #samples by 15%

    def fits(x1, y1):
        for x2 in range(x1 - 1, x1 + 2):
            for y2 in range(y1 - 1, y1 + 2):
                if grid[x2 % nx, y2 % ny] == 1: # If (x,y) is already occupied
                    return False
        return True

    p = (0, 0)
    queue = [p]
    grid[p] = 1
    while queue:
        qi = int(np.random.random()*len(queue))
        qx, qy = queue.pop(qi)
        np.random.shuffle(annulus) # is in-place operation
        for other in annulus[:k]:
            px = (qx + other[0]) % nx
            py = (qy + other[1]) % ny
            if fits(px, py):
                p = (px, py)
                queue.append(p)
                queue.append((qx, qy)) # re-add the current point
                grid[p] = 1
                break
    return np.where(grid)


def distSqPBC(w: float, h: float, positions: np.ndarray):
    ''' Determines the squared distance between the two positions in <positions>,
        taking into account periodic boundaries between x=[0, w] and y=[0, h].
        Adapted from https://yangyushi.github.io/science/2020/11/02/pbc_py.html.
        @param positions [array(2,2)]: 2x2 NumPy array, format [[x0, y0], [x1, y1]]
        @return [float]: the distance between these points, squared.
    '''
    box = [w, h]
    dist_2d_sq = 0
    for dim in range(2):
        dist_1d = pdist(positions[:, dim][:, np.newaxis])
        dist_1d[dist_1d > box[dim]*.5] -= box[dim]
        dist_2d_sq += dist_1d**2 # d² = dx² + dy²
    return dist_2d_sq


def poisson_disc_samples(width, height, r, k=5, random: Callable = np.random.random, PBC=True):
    ''' Performs Poisson disk sampling in the box (width, height) with minimal
        distance between samples <r>. Only deviation from normal Poisson disk
        sampling, is that all points are at integer locations.
    '''
    r += 1
    cellsize = r/math.sqrt(2)
    rSq = r*r

    grid_width = math.ceil(width/cellsize)
    grid_height = math.ceil(height/cellsize)
    grid_size = grid_width*grid_height
    grid = [None]*grid_size

    def grid_coords(p):
        return (int(p[0]/cellsize), int(p[1]/cellsize))

    def fits(p, gx, gy):
        for x in range(gx - 2, gx + 3):
            for y in range(gy - 2, gy + 3):
                g = grid[(x + y*grid_width) % grid_size]
                if g is None:
                    continue
                if distSqPBC(width, height, np.asarray([p, g])) <= rSq:
                    return False
        return True

    p = int(width*random()), int(height*random())
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y*grid_width] = p

    while queue:
        qi = int(random()*len(queue))
        qx, qy = queue.pop(qi)
        for _ in range(k):
            alpha = math.tau*random()
            d = r*np.sqrt(3*random() + 1)
            px = qx + d*np.cos(alpha)
            py = qy + d*np.sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y*grid_width] = p

    if PBC:
        ok = lambda x, y: (x >= 0) & (y >= 0) & (x < max(r, width - r)) & (y < max(r, height - r))
    else:
        ok = lambda *args: True
    grid = [p for p in grid if p is not None]
    offset_x, offset_y = int(random()*width), int(random()*height)
    grid = [((p[0] + offset_x) % width, (p[1] + offset_y) % height) for p in grid if ok(*p)]
    return grid



class SequentialPoissonDiskSampling:
    # TODO: deprecate this in favor of poisson_disc_samples()
    neighbourhood = [
        [0, 0],   [0, -1],  [-1, 0],
        [1, 0],   [0, 1],   [-1, -1],
        [1, -1],  [-1, 1],  [1, 1],
        [0, -2],  [-2, 0],  [2, 0],
        [0, 2],   [-1, -2], [1, -2],
        [-2, -1], [2, -1],  [-2, 1],
        [2, 1],   [-1, 2],  [1, 2]
    ]
    neighbourhoodLength = len(neighbourhood)

    def __init__(self, width: int, height: int, radius: float, tries: int = 3, rng: Callable = np.random.random):
        ''' Class to implement poisson disk sampling. Code inspired by the Javascript version from
            https://github.com/kchapelier/fast-2d-poisson-disk-sampling/blob/master/src/fast-poisson-disk-sampling.js,
            adapted to our own requirements (e.g. integer positions, PBC)
            NOTE: this class is quite basic, it can only support integer positions with equal dx and dy,
                  and only provides a high probability of satisfying PBC, not 100%.
            @param width, height [int]: shape of the space
            @param radius [float]: minimum distance between separate points
            @param tries [int] (30): number of times the algorithm will try to place a point in the neighbourhood of other points
            @param rng [function] (np.random.random): RNG function between 0 to 1
        '''
        epsilon = 2e-14

        self.width = width
        self.height = height
        self.radius = max(radius, 1)
        self.maxTries = max(3, math.ceil(tries))
        self.rng = rng

        self.offset_x, self.offset_y = int(rng()*self.width), int(rng()*self.height)

        self.squaredRadius = self.radius*self.radius
        self.radiusPlusEpsilon = self.radius + epsilon
        self.cellSize = self.radius/math.sqrt(2)

        self.angleIncrement = math.pi*2/self.maxTries
        self.angleIncrementOnSuccess = np.pi/3 + epsilon
        self.triesIncrementOnSuccess = math.ceil(self.angleIncrementOnSuccess/self.angleIncrement)

        self.processList = []
        self.samplePoints = []

        # cache grid
        self.gridShape = [math.ceil(self.width/self.cellSize), math.ceil(self.height/self.cellSize)]
        self.grid = np.zeros(self.gridShape, dtype=int) # will store references to samplePoints

    def addRandomPoint(self):
        ''' Add a totally random point in the grid
            @return [list(2)]: the point added to the grid
        '''
        return self.directAddPoint([
            int(self.rng()*self.width),
            int(self.rng()*self.height),
            self.rng()*math.pi*2,
            0
        ])

    def directAddPoint(self, point):
        ''' Add a given point to the grid, without any check
            @param point [list(4)]: point represented as [x, y, angle, tries]
            @return [list(2)]: the point added to the grid as [x, y]
        '''
        coordsOnly = [point[0], point[1]]
        self.processList.append(point)
        self.samplePoints.append(coordsOnly)

        self.grid[int(point[0]/self.cellSize), int(point[1]/self.cellSize)] = len(self.samplePoints) # store the point reference
        return coordsOnly

    def inNeighbourhood(self, point):
        ''' Check whether a given point is in the neighbourhood of existing points.
            @param point [list(4)]: point represented as [x, y, angle, tries]
            @return [bool]: whether the point is in the neighbourhood of another point
        '''
        for neighbourIndex in range(self.neighbourhoodLength): # PBC dont work perfectly due to neighbourhood being a bit too small near edges
            gridIndex = ((int(point[0]/self.cellSize) + self.neighbourhood[neighbourIndex][0]) % self.gridShape[0],
                         (int(point[1]/self.cellSize) + self.neighbourhood[neighbourIndex][1]) % self.gridShape[1])

            if self.grid[gridIndex] != 0: # i.e. it is occupied
                positions = np.asarray([[point[0], point[1]], self.samplePoints[self.grid[gridIndex] - 1]]) # point, and already existing point
                dist_sq = distSqPBC(self.width, self.height, positions) # Determine with PBC if these two are too close or not
                if dist_sq < self.squaredRadius: return True
        return False

    def next(self):
        ''' Try to generate a new point in the grid.
            @return [list(2)|None]: the added point [x, y] or None
        '''
        while len(self.processList) > 0:
            index = int(len(self.processList)*self.rng()) | 0

            currentPoint = self.processList[index]
            currentAngle = currentPoint[2]
            tries = currentPoint[3]

            if tries == 0: currentAngle += (self.rng() - .5)*np.pi/3*4

            for tries in range(tries, self.maxTries):
                newPoint = [
                    int(currentPoint[0] + math.cos(currentAngle)*self.radiusPlusEpsilon*(1 + self.rng())) % self.width,
                    int(currentPoint[1] + math.sin(currentAngle)*self.radiusPlusEpsilon*(1 + self.rng())) % self.height,
                    currentAngle,
                    0
                ]

                if not self.inNeighbourhood(newPoint):
                    currentPoint[2] = currentAngle + self.angleIncrementOnSuccess + self.rng()*self.angleIncrement
                    currentPoint[3] = tries + self.triesIncrementOnSuccess
                    return self.directAddPoint(newPoint)
                
                currentAngle = currentAngle + self.angleIncrement

            if tries >= self.maxTries - 1:
                r = self.processList.pop()
                if index < len(self.processList): self.processList[index] = r
        
        return None
    
    def offset(self):
        samplePoints = np.asarray(self.samplePoints)
        samplePoints[:,0] = (samplePoints[:,0] + self.offset_x) % self.width
        samplePoints[:,1] = (samplePoints[:,1] + self.offset_y) % self.height
        self.samplePoints = samplePoints.tolist()

    def fill(self):
        ''' Automatically fill the grid, adding a random point to start the process if needed.
            @return [list(N,2)]: N sample points
        '''
        if len(self.samplePoints) == 0: self.addRandomPoint()
        while self.next(): pass
        self.offset()
        return self.samplePoints

    def getAllPoints(self):
        ''' Get all the points in the grid.
            @return [list(N,2)]: all N sample points
        '''
        return self.samplePoints

    def reset(self):
        ''' Reinitialize the grid as well as the internal state. '''
        self.grid = np.zeros_like(self.grid)
        self.samplePoints = []
        self.processList = []

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle


    w, h = 50, 50
    r = 2
    # points = SequentialPoissonDiskSampling(w, h, r, tries=3).fill() # 3 (or 2) tries is optimal in terms of points/second
    points = np.transpose(PoissonGrid(w, h))
    print(f'Sampled {len(points)} points.')

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for x, y in points:
        ax.add_artist(Circle(xy=(x, y), radius=r/2))
    ax.set_xlim([-.5, w-.5])
    ax.set_ylim([-.5, h-.5])
    if w <= 50 and h <= 50:
        ax.set_xticks(np.arange(-.5, w-.5))
        ax.set_yticks(np.arange(-.5, h-.5))
        ax.grid(linestyle=':')
    plt.show()

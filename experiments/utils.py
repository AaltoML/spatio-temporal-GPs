import numpy as np
import datetime
import itertools
import multiprocessing as mp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def hours_to_seconds(hr):
    return hr*(60*60)

def minutes_to_seconds(m):
    return m*(60)

def slice_array(A, arr):
    _A = []
    for a in arr:
        _A.append(A[a, :])
    return np.vstack(_A)

def slice_array_insert(A,B, arr):
    for i, a in enumerate(arr):
        _A_len = A[a].shape[0]

        A[a] = A[a] + B[i*_A_len:(i+1)*_A_len, :]

    return A

def st_batch_predict(model , XS, prediction_fn=None, batch_size=5000, verbose=False):
    """
        With KF models the prediction data must be at all timesteps. This function
            breaks up the space-time data into space-time batches (which spans across all time points)

    """
    #sort XS into grid/'kronecker' structure

    XS = np.roll(XS, -1, axis=1)
    #sort by time points
    grid_idx = np.lexsort(XS.T)
    #reset time axis
    XS = np.roll(XS, 1, axis=1)

    inv_grid_idx = np.argsort(grid_idx)

    XS = XS[grid_idx]


    time_points = np.unique(XS[:, 0])
    num_time_points = time_points.shape[0]
    num_spatial_points = int(XS.shape[0]/num_time_points)

    #number of spatial points that fit into batch

    num_spatial_points_per_batch = int(np.floor(batch_size/num_time_points))

    num_steps = max(1, int(np.floor(num_spatial_points/num_spatial_points_per_batch)))

    if verbose:
        print('num_time_points: ', num_time_points)
        print('num_spatial_points: ', num_spatial_points)
        print('num_steps: ', num_steps)
        print('num_spatial_points_per_batch: ', num_spatial_points_per_batch)

    #empty prediction data
    mean = np.zeros([XS.shape[0], 1])
    var = np.zeros_like(mean)

    for i in range(num_steps):
        if verbose:
            print(f"{i}/{num_steps}")

        batch = num_spatial_points_per_batch
        if i == num_steps-1:
            #select the remaining spatial points
            batch = num_spatial_points - i*num_spatial_points_per_batch

        #k*num_spatial_points is the index to the j'th time slice
        #i*num_spatial_points_per_batch is index to current spatial batch
        start_idx = lambda j: j*num_spatial_points + i*num_spatial_points_per_batch
        end_idx = lambda j: start_idx(j) + batch

        step_idx = [
            slice(start_idx(j), end_idx(j)) for j in range(num_time_points)
        ]

        _XS = slice_array(XS, step_idx)

        if prediction_fn is not None:
            _mean, _var = prediction_fn(_XS)
        else:
            _mean, _var = model.predict_y(_XS, diagonal_var=True)

        _mean = np.squeeze(_mean).reshape([-1, 1])
        _var = np.squeeze(_var).reshape([-1, 1])

        mean = slice_array_insert(mean, _mean, step_idx)
        var = slice_array_insert(var, _var, step_idx)

    #unsort grid/kronecker structre
    return [mean[inv_grid_idx]], [var[inv_grid_idx]]
    #return [mean], [var]

def batch_predict(XS, prediction_fn=None, batch_size=1000, verbose=False):
    # Ensure batch is less than the number of test points
    if XS.shape[0] < batch_size:
        batch_size = XS.shape[0]

    # Split up test points into equal batches
    num_batches = int(np.ceil(XS.shape[0] / batch_size))

    ys_arr = []
    ys_var_arr = []
    index = 0

    for count in range(num_batches):
        if verbose:
            print(f"{count}/{num_batches}")
        if count == num_batches - 1:
            # in last batch just use remaining of test points
            batch = XS[index:, :]
        else:
            batch = XS[index : index + batch_size, :]

        index = index + batch_size

        # predict for current batch
        y_mean, y_var = prediction_fn(batch)

        ys_arr.append(y_mean)
        ys_var_arr.append(y_var)

    y_mean = np.concatenate(ys_arr, axis=0)
    y_var = np.concatenate(ys_var_arr, axis=0)

    return y_mean, y_var

def key_that_starts_with(arr, s):
    for a in arr:
        if a.startswith(s):
            return a
    return False


#Normalise Data input
def normalise(x, wrt_to):
    return (x - np.mean(wrt_to))/np.std(wrt_to)

def normalise_df(x, wrt_to):
    return (x - np.mean(wrt_to, axis=0))/np.std(wrt_to, axis=0)

def un_normalise_df(x, wrt_to):
    return x* np.std(wrt_to, axis=0) + np.mean(wrt_to, axis=0)

def create_spatial_grid(x1, x2, y1, y2, n1, n2):
    x = np.linspace(x1, x2, n1)
    y = np.linspace(y1, y2, n2)
    grid = []
    for i in x:
        for j in y:
            grid.append([i, j])
    return np.array(grid)

def create_spatial_temporal_grid(time_points, x1, x2, y1, y2, n1, n2):
    x = np.linspace(x1, x2, n1)
    y = np.linspace(y1, y2, n2)
    grid = []
    for t in time_points:
        for i in x:
            for j in y:
                grid.append([t, i, j])
    return np.array(grid)

def numpy_to_list(a): 
    return [a[i][:, None] for i in range(a.shape[0])]

def datetime_str_to_epoch(s):
    p = '%Y/%m/%d %H:%M:%S'
    epoch = datetime.datetime(1970, 1, 1)
    return int((datetime.datetime.strptime(s, p) - epoch).total_seconds())

def datetime_to_epoch(datetime):
    """
        Converts a datetime to a number
        args:
            datatime: is a pandas column
    """
    return datetime.astype('int64')//1e9

def epoch_to_datetime(epoch):
    return datetime.datetime.fromtimestamp(epoch)

def epochs_to_datetime_list(epochs):
    return [datetime.datetime.fromtimestamp(epoch) for epoch in epochs]

def ensure_timeseries_at_each_locations(X, Y):
    """
        This removes all spatial locations that do not have a full timeseries 
        TODO: replace with Will's function
    """
    time_points = np.unique(X[:, 0])
    num_time_points = time_points.shape[0]

    X_space_only = X[:, 1:]
    new_arr, indices, counts = np.unique(X_space_only, return_index=True, return_counts=True, axis=0)
    spatial_to_remove = X_space_only[indices[counts != num_time_points]]
    for spatial_point in spatial_to_remove:
        idx = (X[:, 1:] != spatial_point)
        idx = np.all(idx, axis=1)

        X = X[idx, :]
        Y = Y[idx, :]

    return X, Y

#taken directly from https://github.com/AaltoML/spacetime-kalman/blob/master/kalmanjax/utils.py#L308
def discretegrid(xy, w, nt):
    """
    Convert spatial observations to a discrete intensity grid
    :param xy: observed spatial locations as a two-column vector
    :param w: observation window, i.e. discrete grid to be mapped to, [xmin xmax ymin ymax]
    :param nt: two-element vector defining number of bins in both directions
    """
    # Make grid
    x = nnp.linspace(w[0], w[1], nt[0] + 1)
    y = nnp.linspace(w[2], w[3], nt[1] + 1)
    X, Y = nnp.meshgrid(x, y)

    # Count points
    N = nnp.zeros([nt[1], nt[0]])
    for i in range(nt[0]):
        for j in range(nt[1]):
            ind = (xy[:, 0] >= x[i]) & (xy[:, 0] < x[i + 1]) & (xy[:, 1] >= y[j]) & (xy[:, 1] < y[j + 1])
            N[j, i] = nnp.sum(ind)
    return X[:-1, :-1].T, Y[:-1, :-1].T, N.T

class center_point_discretise_grid_inner_loop():
    def __init__(self, data, t_arr, centers_arr, input_dim):
        self.data = data
        self.t_arr = t_arr
        self.centers_arr = centers_arr
        self.input_dim = input_dim

    def __call__(self, i):
        in_range = lambda data, d, t, i: (data[:, d] >= t[i]) & (data[:, d] < t[i + 1])

        #create idx over each dim
        idx = [
            in_range(self.data, d, self.t_arr[d], i[d]) for d in range(self.input_dim)
        ]

        idx = np.logical_and.reduce(idx, axis=0)

        N_kij = np.sum(idx)

        _counts = N_kij
        _X = [self.centers_arr[d][i[d]] for d in range(self.input_dim)]

        return _X, _counts

def center_point_discretise_grid(data, bin_sizes = [5]):
    """
        data: [time] columns

        Returns:
            the binned counts, the center locations of the bins and the bin sizes
    """

    
    input_dim = len(bin_sizes)

    #helper functions
    lin = lambda i: np.linspace(
        np.min(data[:, i]), np.max(data[:, i]), bin_sizes[i]+1
    )

    centers = lambda A: np.array([np.mean([A[i], A[i+1]]) for i in range(A.shape[0] -1 )])


    #get grid cell positions
    t_arr = [lin(d) for d in range(input_dim)]

    #get size of a single cell
    binned_x_size_arr = [t_arr[d][1]-t_arr[d][0] for d in range(input_dim)]

    #get center locations of grid cells
    centers_arr = [centers(t_arr[d]) for d in range(input_dim)]


    #X.append([centers_arr[d][i[d]] for d in range(input_dim)])
    #counts.append(N_kij)

    bin_ranges = [range(0, bin_sizes[d]) for d in range(input_dim)]

    with mp.Pool(processes=mp.cpu_count()) as p:
        res = p.map(
            center_point_discretise_grid_inner_loop(
                data, 
                t_arr, 
                centers_arr,
                input_dim
            ), 
            [i for i in itertools.product(*bin_ranges)]
        )
        X, counts = zip(*res)


    X = np.array(X)
    Y = np.array(counts)[:, None]

    return X, Y, binned_x_size_arr


def center_point_st_discretise_grid(data, bin_sizes = [5, 10, 20]):
    """
        data: [time, lat, lon] columns

        Returns:
            the binned counts, the center locations of the bins and the bin sizes
    """
    lin = lambda i: np.linspace(
        np.min(data[:, i]), np.max(data[:, i]), bin_sizes[i]+1
    )

    centers = lambda A: np.array([np.mean([A[i], A[i+1]]) for i in range(A.shape[0] -1 )])

    #get grid cell positions
    t = lin(0)
    x = lin(1)
    y = lin(2)

    #get size of a single cell
    binned_x_sizes = [
        t[1]-t[0],
        x[1]-x[0],
        y[1]-y[0]
    ]   

    #get center locations of grid cells
    t_centers = centers(t)
    x_centers = centers(x)
    y_centers = centers(y)
    

    counts = []
    X = []

    #iterate each grid cell and count how many occurences are within it
    for k in range(bin_sizes[0]):
        for i in range(bin_sizes[1]):
            for j in range(bin_sizes[2]):
                idx = (data[:, 0] >= t[k]) & (data[:, 0] < t[k + 1]) & \
                      (data[:, 1] >= x[i]) & (data[:, 1] < x[i + 1]) & \
                      (data[:, 2] >= y[j]) & (data[:, 2] < y[j + 1])

                N_kij = np.sum(idx)
                X.append([t_centers[k], x_centers[i], y_centers[j]])
                counts.append(N_kij)


    X = np.array(X)
    Y = np.array(counts)[:, None]

    return X, Y, binned_x_sizes

def create_geopandas_spatial_grid(xmin, xmax, ymin, ymax, cell_size_x, cell_size_y, crs=None):
    """
        see https://james-brennan.github.io/posts/fast_gridding_geopandas/
    """
    import geopandas
    import shapely

    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax-cell_size_x, cell_size_x ):
        for y0 in np.arange(ymin, ymax-cell_size_y, cell_size_y):
            # bounds
            x1 = x0+cell_size_x
            y1 = y0+cell_size_y
            grid_cells.append( shapely.geometry.box(x0, y0, x1, y1)  )

    cell = geopandas.GeoDataFrame(grid_cells, columns=['geometry'], 
                                 crs=crs)
    return cell

def pad_with_nan_to_make_grid(X, Y):
    #converts data into grid

    N = X.shape[0]

    #construct target grid
    unique_time = np.unique(X[:, 0])
    unique_space = np.unique(X[:, 1:], axis=0)

    Nt = unique_time.shape[0]
    Ns = unique_space.shape[0]

    print('grid size:', N, Nt, Ns, Nt*Ns)

    X_tmp = np.tile(np.expand_dims(unique_space, 0), [Nt, 1, 1])

    time_tmp = np.tile(unique_time, [Ns]).reshape([Nt, Ns], order='F')

    X_tmp = X_tmp.reshape([Nt*Ns, -1])

    time_tmp = time_tmp.reshape([Nt*Ns, 1])

    #X_tmp is the full grid
    X_tmp = np.hstack([time_tmp, X_tmp])

    #Find the indexes in X_tmp that we need to add to X to make a full grid
    _X = np.vstack([X,  X_tmp])
    _Y = np.nan*np.zeros([_X.shape[0], 1])

    _, idx = np.unique(_X, return_index=True, axis=0)
    idx = idx[idx>=N]
    print('unique points: ', idx.shape)

    X_to_add = _X[idx, :]
    Y_to_add = _Y[idx, :]

    X_grid = np.vstack([X, X_to_add])
    Y_grid = np.vstack([Y, Y_to_add])

    #sort for good measure
    _X = np.roll(X_grid, -1, axis=1)
    #sort by time points first
    idx = np.lexsort(_X.T)

    return X_grid[idx], Y_grid[idx]


def train_test_split_indices(N, split=0.5, seed=0):
    np.random.seed(seed)
    rand_index = np.random.permutation(N)

    N_tr =  int(N * split) 

    return rand_index[:N_tr], rand_index[N_tr:] 

def lat_lon_to_polygon_buffer(row, actual_bin_sizes):
    import shapely
    from shapely.geometry import Polygon

    lat = row['Latitude']
    lon = row['Longitude']
    
    w1 = actual_bin_sizes[1]/2
    w2 = actual_bin_sizes[2]/2
    
    p1 = [lon-w1, lat-w2]
    p2 = [lon-w1, lat+w2]
    p3 = [lon+w1, lat+w2]
    p4 = [lon+w1, lat-w2]
    return Polygon([p1, p2, p3, p4, p1])


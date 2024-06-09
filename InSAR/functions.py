import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from scipy.spatial import Delaunay
import math
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.ndimage import gaussian_filter

def get_stack(path_to_netcdf_file="./data/s1_asc_t088_v2.nc"):
    """Load the netcdf file and return the xarray dataset object."""
    s1_stack = xr.load_dataset(path_to_netcdf_file, engine="h5netcdf")
    s1_stack = s1_stack.isel(time=slice(42, -1))
    return s1_stack

def plot_temporal_intensity_image(s1_stack):
    """
    Plot the temporal average intensity image of the stack and the GNS station on the map.
    """

    lat_gnss_station = 53.13637817
    lon_gnss_station = 6.44928672 

    int_tmpavg = np.mean(np.abs(s1_stack.complex.values)**2,axis=2)

    mean0 = np.mean(int_tmpavg)
    std0 = np.std(int_tmpavg)

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 10))

    mesh = ax.pcolormesh(s1_stack.lon, s1_stack.lat, int_tmpavg**0.25, transform=ccrs.PlateCarree(), 
                        cmap='inferno', vmax=(mean0+1*std0)**0.25)

    fig.colorbar(mesh, ax=ax, orientation='horizontal', label='Intensity')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'color': 'gray'}
    gl.ylabel_style = {'size': 12, 'color': 'gray'}

    ax.plot(lon_gnss_station, lat_gnss_station, marker='^', color='blue', markersize=10, transform=ccrs.PlateCarree(), label='RDN1 station')

    start_date = np.datetime_as_string(s1_stack.time.values[0], unit='M')
    end_date = np.datetime_as_string(s1_stack.time.values[-1], unit='M')

    plt.title(f"Temporal average intensity image of Groningen between {start_date} and {end_date}")

    plt.legend(loc='upper right')

    plt.show()
    plt.close()

def compute_amplitude_dispersion(s1_stack):
    """ Compute the amplitude dispersion of the stack and the calibrated stack. """
    stack_amps = np.abs(s1_stack.complex.values)
    intensity = np.abs(s1_stack.complex.values)**2

    intensity_mean_space = np.mean(intensity, axis=(0,1))

    Da = np.std(stack_amps, axis=-1)/np.mean(stack_amps, axis=-1)

    amp_cal = stack_amps / np.sqrt(intensity_mean_space).reshape((1,1,stack_amps.shape[2]))
    Da_cal = np.std(amp_cal, axis=-1)/np.mean(amp_cal, axis=-1)

    return Da, Da_cal

def find_points_within_polygon(lat_points, lon_points, lat_gnss_station, lon_gnss_station, radius):
    """
    Finds the indices of latitude and longitude points within a polygon around the GNSS station.

    Parameters:
    lat_points (2D array-like): The latitude values.
    lon_points (2D array-like): The longitude values.
    lat_gnss_station (float): The latitude of the GNSS station.
    lon_gnss_station (float): The longitude of the GNSS station.
    radius (float): The radius of the polygon around the GNSS station.

    Returns:
    list: Indices of points within the polygon.
    """
    station_point = Point(lon_gnss_station, lat_gnss_station)
    polygon = station_point.buffer(radius)
    
    flat_lat_points = lat_points.flatten()
    flat_lon_points = lon_points.flatten()
    
    indices_within_polygon = []
    
    for idx, (lat, lon) in enumerate(zip(flat_lat_points, flat_lon_points)):
        point = Point(lon, lat)
        if polygon.contains(point):
            indices_within_polygon.append(np.unravel_index(idx, lat_points.shape))
    
    return indices_within_polygon

def get_minimum_da(Da, points_around_gnss):
    """
    Get the minimum amplitude dispersion around the GNSS station.

    Parameters:
    Da (2D array-like): The amplitude dispersion values.
    points_around_gnss (list): The indices of points around the GNSS station.

    Returns:
    float: The minimum amplitude dispersion value.
    """
    min_da = np.inf
    min_idx = None
    
    for idx in points_around_gnss:
        if Da[idx] < min_da:
            min_da = Da[idx]
            min_idx = idx
    
    return min_da, min_idx

def compute_delaunay_arcs(Da_cal, Da_threshold, s1_stack, ref_point,
                          bad_point=None, starting_points=None, ts=0.7, preprocessing=True):

    if starting_points is None:
        ps_cal_pos = np.where(Da_cal < Da_threshold)
        points = np.array(ps_cal_pos).T

        slc_points = s1_stack.complex.values[points[:, 0], points[:, 1]]

        if preprocessing:

            bad_points_indexes = []

            for i in range(1, slc_points.shape[1]):
                image1 = slc_points[:, i-1]
                image2 = slc_points[:, i]

                coherence = calculate_sample_coherence(image1, image2, sigma=1.0)

                bad_points = np.where(coherence < ts)[0]

                bad_points_indexes.append(bad_points)

            bad_points_indexes = np.unique(np.concatenate(bad_points_indexes))

            print(f"Removing {len(bad_points_indexes)} points out of {len(points)} with coherence < {ts}.")

            points = np.delete(points, bad_points_indexes, axis=0)

    else:
        points = starting_points

    if bad_point is not None:
        points = np.delete(points, bad_point, axis=0)

    tri = Delaunay(points * np.array([4,1]).reshape((1,2)))

    def tri2arcs(dela):
        ntr = dela.simplices.shape[0]
        arcs = np.zeros((ntr,3,2))
        arcs[:,0,0] = dela.simplices[:,0]   
        arcs[:,0,1] = dela.simplices[:,1]
        arcs[:,1,0] = dela.simplices[:,1]
        arcs[:,1,1] = dela.simplices[:,2]
        arcs[:,2,0] = dela.simplices[:,2]
        arcs[:,2,1] = dela.simplices[:,0]
        # sort them by last index and then by first
        arcs = arcs.reshape((ntr*3, 2))
        arcs = np.sort(arcs, axis=1)
        dtype = arcs.dtype.descr * 2  # some code found in stackoverflow:
        # https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        struct = arcs.view(dtype)
        uniq = np.unique(struct)
        uniq = uniq.view(arcs.dtype).reshape(-1, 2)
        return uniq
    
    def create_star_network(points, idx_ref_point):
        n_points = points.shape[0]
        arcs = np.zeros((n_points-1, 2), dtype=np.int32)
        for i in range(1, n_points):
            arcs[i-1] = [idx_ref_point, i]

        return arcs
    
    idx_ref_point = np.where((points == ref_point).all(axis=1))[0][0]
    
    arcs_star = create_star_network(points, idx_ref_point)

    all_arcs = np.concatenate((tri2arcs(tri).astype(np.int32), arcs_star))

    return all_arcs, points

    # return tri2arcs(tri).astype(np.int32), points


def plot_delaunay_arcs(s1_stack, int_tmpavg, Da_threshold, arcs, points, 
                       lat_gnss_station, lon_gnss_station, mean0, std0, plot=True, verbose=False):
    if plot:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 10))

        mesh = ax.pcolormesh(s1_stack.lon, s1_stack.lat, int_tmpavg**0.25, transform=ccrs.PlateCarree(), 
                            cmap='bone', vmax=(mean0+1*std0)**0.25)

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)

        # Add gridlines and labels
        gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12, 'color': 'gray'}
        gl.ylabel_style = {'size': 12, 'color': 'gray'}

        ax.plot(lon_gnss_station, lat_gnss_station, marker='^', color='blue', markersize=10, transform=ccrs.PlateCarree(), label='RDN1 station')

        for i, arc in enumerate(arcs):
            arc_points_1 = points[arc][0]
            arc_points_2 = points[arc][1]
            if i == 0:
                ax.plot(
                    [s1_stack.lon.values[arc_points_1[0], arc_points_1[1]], s1_stack.lon.values[arc_points_2[0], arc_points_2[1]]],
                    [s1_stack.lat.values[arc_points_1[0], arc_points_1[1]], s1_stack.lat.values[arc_points_2[0], arc_points_2[1]]],
                    transform=ccrs.PlateCarree(), label="Dalaunay arcs")
                
            else:
                ax.plot(
                    [s1_stack.lon.values[arc_points_1[0], arc_points_1[1]], s1_stack.lon.values[arc_points_2[0], arc_points_2[1]]],
                    [s1_stack.lat.values[arc_points_1[0], arc_points_1[1]], s1_stack.lat.values[arc_points_2[0], arc_points_2[1]]],
                    transform=ccrs.PlateCarree())
            
            
        plt.legend(loc='upper right')

        plt.title(f"There are {len(points)} points with amplitude dispersion < {Da_threshold} forming {len(arcs)} arcs")

        plt.show()

    if verbose:
        print(f"There are {len(points)} points with amplitude dispersion < {Da_threshold} forming {len(arcs)} arcs")

def adjust_phase(phase_array):
    
    best_adjusted_phase = None
    smallest_residual = float('inf')
    best_tol = None

    k = 2

    for tol in [math.pi/2, 3, 4, 5.5]:
        jumps = [i-2 for i in range(k, len(phase_array)) if np.abs(phase_array[i] - phase_array[i-k]) > tol]
        jumps.append(len(phase_array))  # Add end of array as last jump point
        adjusted_phase = phase_array.copy()
    
        for i, jump in enumerate(jumps[:-1]):
            start = 0
            end = jumps[i+1]

            chunk = adjusted_phase[start:end]

            plus_2pi_chunk = chunk.copy()
            minus_2pi_chunk = chunk.copy()
            plus_2pi_chunk[jump+1:] += 2 * np.pi
            minus_2pi_chunk[jump+1:] -= 2 * np.pi

            original_residual = np.sum(np.abs(chunk - np.mean(chunk)))
            plus_2pi_residual = np.sum(np.abs(plus_2pi_chunk - np.mean(plus_2pi_chunk)))
            minus_2pi_residual = np.sum(np.abs(minus_2pi_chunk - np.mean(minus_2pi_chunk)))

            if plus_2pi_residual < original_residual and plus_2pi_residual < minus_2pi_residual:
                adjusted_phase[jump:end] = plus_2pi_chunk[jump-start:]
            elif minus_2pi_residual < original_residual:
                adjusted_phase[jump:end] = minus_2pi_chunk[jump-start:]

        current_residual = np.sum(np.abs(adjusted_phase - np.mean(adjusted_phase)))
        
        if current_residual < smallest_residual:
            smallest_residual = current_residual
            best_adjusted_phase = adjusted_phase
            best_tol = tol

    return best_adjusted_phase

def unwrap_phases(phases, tol1=np.pi):
    unwrapped_phase = np.copy(phases)
    rows, cols = unwrapped_phase.shape

    for i in range(1, cols):
        phase_diff = unwrapped_phase[:, i] - unwrapped_phase[:, i - 1]
        
        wrap_mask = (phase_diff < -tol1) | (phase_diff > tol1)
        
        unwrapped_phase[:, i][wrap_mask] += 2 * np.pi * np.sign(-phase_diff[wrap_mask])

    return unwrapped_phase

def calculate_sample_coherence(image1, image2, sigma=1.0):
    """
    Calculate sample coherence between two SAR images using Gaussian filtering.
    
    Parameters:
    - image1: np.ndarray, complex SAR image 1
    - image2: np.ndarray, complex SAR image 2
    - sigma: float, standard deviation for Gaussian kernel
    
    Returns:
    - coherence: np.ndarray, coherence image
    """
    
    conj_prod = image1 * np.conj(image2)
    mag1_sq = np.abs(image1) ** 2
    mag2_sq = np.abs(image2) ** 2

    conj_prod_mean = gaussian_filter(conj_prod, sigma=sigma)
    mag1_sq_mean = gaussian_filter(mag1_sq, sigma=sigma)
    mag2_sq_mean = gaussian_filter(mag2_sq, sigma=sigma)

    coherence = np.abs(conj_prod_mean) / np.sqrt(mag1_sq_mean * mag2_sq_mean)

    return coherence

def compute_arc_phase_uw(slc_points, best_ind, arcs, fix_jump=True):

    amplitudes_ps = np.abs(slc_points)

    phases_ps = np.angle(slc_points)
    phases_ps_minus_reference = phases_ps - phases_ps[best_ind][np.newaxis,:]

    slc_points_minus_ref_phase = amplitudes_ps * np.exp(1j*phases_ps_minus_reference)

    interf = slc_points_minus_ref_phase[:, 0:-1] * np.conj(slc_points_minus_ref_phase[:,1:])

    arc_phasor = interf[arcs[:,0], :] * np.conj(interf[arcs[:,1], :])

    arc_phase = np.angle(arc_phasor)

    # arc_phase_uw = np.cumsum(arc_phase, axis=1)

    arc_phase_uw = unwrap_phases(arc_phase)

    if fix_jump:

        arc_phase_uw_no_jump = []

        for i in range(arc_phase_uw.shape[0]):
            arc_phase_uw_no_jump.append(adjust_phase(arc_phase_uw[i]))

        arc_phase_uw = np.array(arc_phase_uw_no_jump)


    return arc_phase_uw


def compute_los_deformation_from_phase(arc_phase_uw):
    c = 299_792_458
    freq = 5.404*10**9

    lam_meters = c / freq
    
    wavelength = lam_meters * 10**3 # mm


    arc_deformation = wavelength * arc_phase_uw / 4 / math.pi
    
    return arc_deformation

def get_bad_points(arcs, bad_arcs):
    flat_bad_arcs = np.array(arcs[bad_arcs]).flatten()

    # print(flat_bad_arcs)

    flat_arcs = arcs.flatten()

    values_bad, count_bads = np.unique(flat_bad_arcs, return_counts=True)
    values_all, count_all = np.unique(flat_arcs, return_counts=True)

    frequency_bad_dict = dict(zip(values_bad, count_bads))
    frequency_all_dict = dict(zip(values_all, count_all))
    return frequency_bad_dict, frequency_all_dict

def get_bad_arcs(gradient_threshold, arc_deformation, arcs):

    bad_arcs = []

    for arcn in range(len(arcs)):
        if np.any(np.abs(np.gradient(arc_deformation[arcn, :])) > gradient_threshold):
            bad_arcs.append(arcn)

    return bad_arcs

def find_baddest_point(frequency_bad_dict, frequency_all_dict, best_ind):
    sorted_items = sorted(((key, value) for key, value in frequency_bad_dict.items() if key != best_ind), key=lambda item: item[1], reverse=True)

    worse_keys_list = []

    for key, value in sorted_items:
        if value == frequency_all_dict[key]:
            worse_keys_list.append(key)
            
    if len(worse_keys_list) > 0:
        print(f"Deleted {len(worse_keys_list)} 1x worse points")
        return worse_keys_list
    
    else:
        for key, value in sorted_items:
            if value >= 0.8* frequency_all_dict[key]:
                worse_keys_list.append(key)
            
        if len(worse_keys_list) > 0:
            print(f"Deleted {len(worse_keys_list)} 0.8x worse points")
            return worse_keys_list
        
        else:
            for key, value in sorted_items:
                if value >= 0.6* frequency_all_dict[key]:
                    worse_keys_list.append(key)
                
            if len(worse_keys_list) > 0:
                print(f"Deleted {len(worse_keys_list)} 0.6x worse points")
                return worse_keys_list
            
            else:
                for key, value in sorted_items:
                    if value >= 0.5* frequency_all_dict[key]:
                        worse_keys_list.append(key)
                    
                if len(worse_keys_list) > 0:
                    print(f"Deleted {len(worse_keys_list)} 0.5x worse points")
                    return worse_keys_list
                
                else:
                    print(f"Deleting last things....")
                    for key, value in sorted_items:
                        if value >= 0.5* frequency_all_dict[key]:
                            return key
            
    return None

def get_slc_and_Da_points(arcs, points, s1_stack, Da_cal):
    # azimuth index for beginning of arc
    az_1 = points[arcs[:,0] ,0]
    # azimuth index for end of arc
    az_2 = points[arcs[:,1] ,0]
    # etc
    rg_1 = points[arcs[:,0] ,1]
    rg_2 = points[arcs[:,1] ,1]
    # complex values time-series for selected points
    slc_points = s1_stack.complex.values[points[:, 0], points[:, 1]]
    Da_points = Da_cal[points[:, 0], points[:, 1]]

    return slc_points, Da_points

def preprocess(Da_cal, Da_cal_min, s1_stack, Da_threshold, gradient_threshold, verbose=False):

    baddest_point = None
    i = 1

    lat_gnss_station = 53.13637817
    lon_gnss_station = 6.44928672 

    points_around_gnss = find_points_within_polygon(s1_stack.lat.values, s1_stack.lon.values, lat_gnss_station, lon_gnss_station, 0.01)

    _, idx_cal_min = get_minimum_da(Da_cal, points_around_gnss)
        
    arcs, points = compute_delaunay_arcs(Da_cal, Da_threshold, s1_stack, idx_cal_min, baddest_point)

    slc_points = s1_stack.complex.values[points[:, 0], points[:, 1]]
    Da_points = Da_cal[points[:, 0], points[:, 1]]

    best_ind = np.where(Da_points == Da_cal_min)[0][0]

    best_point_beginning = points[best_ind]

    if verbose:
        print(f"Iteration number {i}.")
        print(f"Best point = {points[best_ind]}")

    arc_phase_uw = compute_arc_phase_uw(slc_points, best_ind, arcs)

    arc_deformation = compute_los_deformation_from_phase(arc_phase_uw)

    bad_arcs = get_bad_arcs(gradient_threshold, arc_deformation, arcs)


    if len(bad_arcs) > 0:
            frequency_bad_dict, frequency_all_dict = get_bad_points(arcs, bad_arcs)

            baddest_point = find_baddest_point(frequency_bad_dict, frequency_all_dict, best_ind)
            
            if verbose:
                print(f"Len bad arcs = {len(bad_arcs)} of {len(arcs)} arcs.")
                print(f"Frequency bad dict = {frequency_bad_dict}\nfrequency all dict = {frequency_all_dict}")
                print(f"baddest point = {baddest_point}")
            

    elif len(bad_arcs) == 0:
        baddest_point = None

        if verbose:
            print(f"baddest point = {baddest_point}")

    while baddest_point is not None:

        i += 1
        arcs, points = compute_delaunay_arcs(Da_cal, Da_threshold, s1_stack, idx_cal_min, baddest_point, points)
        
        slc_points = s1_stack.complex.values[points[:, 0], points[:, 1]]
        Da_points = Da_cal[points[:, 0], points[:, 1]]

        best_ind = np.where(Da_points == Da_cal_min)[0][0]

        if verbose:
            print("======================================")
            print(f"Iteration number {i}.")
            print(f"Best point = {points[best_ind]}")

        arc_phase_uw = compute_arc_phase_uw(slc_points, best_ind, arcs)

        arc_deformation = compute_los_deformation_from_phase(arc_phase_uw)

        bad_arcs = get_bad_arcs(gradient_threshold, arc_deformation, arcs)

        if len(bad_arcs) > 0:

            frequency_bad_dict, frequency_all_dict = get_bad_points(arcs, bad_arcs)
            
            baddest_point = find_baddest_point(frequency_bad_dict, frequency_all_dict, best_ind)
            
            if verbose:
                print(f"Frequency bad dict = {frequency_bad_dict}\nfrequency all dict = {frequency_all_dict}")
                print(f"baddest point = {baddest_point}")

        else:
            baddest_point = None

            if verbose:
                print(f"baddest point = {baddest_point}")


    if best_point_beginning[0] != points[best_ind][0] or best_point_beginning[1] != points[best_ind][1]:
        raise ValueError("The best point has changed during the iterations. Exiting. Check your code.")
    
    return arcs, points, arc_phase_uw, best_ind
        

        
def compute_uw_phase_network(arcs, points, arc_phase_uw, best_ind, prev_points=None, prev_ls_phase=None):
    neq = arcs.shape[0]
    npt = points.shape[0]

    eqs = np.concatenate((np.ones(neq+1), -1*np.ones(neq)))
    rows = np.concatenate((np.array([0]), 1+np.arange(neq), 1+np.arange(neq)))
    cols = np.concatenate((np.array([best_ind]), arcs[:,1], arcs[:,0]))
    A = csr_matrix((eqs, (rows, cols)), shape=(neq+1, npt))

    A = lil_matrix(A)

    known_indices = []
    if prev_points is not None and prev_ls_phase is not None:
        for pp in prev_points:
            index = np.where((points == pp).all(axis=1))[0]
            if len(index) > 0:
                # raise ValueError(f"Previous point {pp} not found in points. It should be there.")
                known_indices.append(index[0])

        for index in known_indices:
            A[index, :] = 0  
            A[index, index] = 1 

    ls_phase_arr = []

    A = csr_matrix(A)

    for t in range(arc_phase_uw.shape[1]):
        b = np.concatenate((np.array([0]), arc_phase_uw[:, t]))

        if prev_points is not None and prev_ls_phase is not None:
            for index, value in zip(known_indices, prev_ls_phase[:, t]):
                b[index] = value

        full_solution = lsqr(A, b)[0]
        ls_phase_arr.append(full_solution)

    ls_phase_arr = np.array(ls_phase_arr).T
    return A, ls_phase_arr


def compute_MSE(A, ls_phase_arr, arc_phase_uw_uw):
    y_hat = A @ ls_phase_arr
    y_hat = y_hat[1:]

    residuals = arc_phase_uw_uw - y_hat

    mse = np.mean(residuals**2, axis=0)

    return mse


def drop_isel(data_array, indices, dim):
    """
    Drop specified indices from a given dimension of an xarray DataArray using isel.

    Parameters:
    data_array (xarray.DataArray): The DataArray to drop indices from.
    indices (list): The list of indices to drop.
    dim (str): The name of the dimension from which to drop indices.

    Returns:
    xarray.DataArray: The DataArray with specified indices dropped.
    """
    # Create a boolean mask for the indices to keep
    mask = np.ones(data_array.sizes[dim], dtype=bool)
    mask[indices] = False
    return data_array.isel({dim: mask})
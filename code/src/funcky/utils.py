import warnings

import os
from tqdm.auto import tqdm

import numpy as np
import scipy

# nwb specific imports 
import pynwb
from nwbwidgets import nwb2widget
from hdmf_zarr import NWBZarrIO 

def load_nwb_session(session_name, data_dir):
    """Load NWB path to specified session.
    
    Parameters
    ----------
    session_name : int.
        Session name, obtained from `V1DD_metadata.csv`.

    data_dir : str.
        Path to parent data directory.

    Returns
    -------
    nwb object

    """

    # construct path (relative to data_dir) to session
    subject_id = session_name.split('_')[0]
    is_golden_mouse = (int(subject_id) == 409828)

    subdir_name = f"{subject_id}_V1DD"
    subdir_name += "_GoldenMouse" if is_golden_mouse else ""

    data_fpath = os.path.join(subdir_name, session_name, session_name+'.nwb.zarr')

    # Load nwb path
    nwb_path_zarr = os.path.join(data_dir, data_fpath)
    io = NWBZarrIO(nwb_path_zarr, mode='r')
    nwb = io.read()

    return nwb


def get_aligned_session_data(nwb, remove_known_bad_planes=True):
    """Get aligned dF/F and behavior data traces for a given session.
    
    Modified from Sven's `data_extraction/2Ppreprocessing.ipynb` notebook
    to work with newer pynwb format.

    Parameters
    ----------
    nwb : NWB file object
        File object for a session of data.

    remove_known_bad_planes : bool.
        Remove plane 6 of (column 1, volume 5).
        Default: True.

    Returns
    -------
    dict of np.ndarray, consisting of:
        timestamps : np.ndarray, shape (n_timestamps,)
            Timestamps that all `*_traces` are aligned to.

        dff_traces : np.ndarray, shape (n_timestamps, n_total_cells,)
            dF/F traces for all valid neurons across all planes, aligned to `timestamps`

        roi_ids : np.ndarray, shape (n_total_cells,).
            Unique ID of cell, for the given plane
        
        plane_ids : np.ndarray, shape (n_total_cells,)
            Unique plane ID of cell, in ranges [0,6).

        volume_ids : np.ndarray, shape (n_total_cells,)
            Unique volume ID of cell, ranges from [1,7).

        column_ids : np.ndarray, shape (n_total_cells,)
            Unique column ID of cell, ranges from [1,6).

        behavior_traces : np.ndarray, shape (n_timestamps, n_behaviors)
            Raw / unprocessed behavioral variable traces
        
        behavior_names : np.ndarray, shape (n_behaviors,)
            List of behavioral variable names

    """
    plane_ids = range(6)

    # [nit] this seems like a pretty brittle way to extract column and volume data
    # but not sure how else to get this info from the nwb
    desc = nwb.experiment_description
    column = int(str.split(desc, ' ')[-3][0])  # ignore comma attached to column index
    volume = int(str.split(desc, ' ')[-1])
    if remove_known_bad_planes:
        if column == 1 and volume == 5:
            plane_ids = range(5)
    
    # Use plane-0 timestamps as reference timestamps
    refr_timestamps = nwb.processing['plane-0'].data_interfaces['dff'].timestamps[:]

    # ------------------------------------------------------------------------------------
    # Load calcium imaging data
    # ------------------------------------------------------------------------------------
    all_dff_traces, all_roi_ids, all_plane_ids = [], [], []
    for plane_id in tqdm(plane_ids):
        this_plane = nwb.processing[f'plane-{plane_id}'].data_interfaces

        # Get dF/F traces for valid rois
        rois = this_plane['image_segmentation']['roi_table'][:]
        good_rois = rois[rois.is_soma==True].roi.values
        
        dff_traces = this_plane['dff'].data[:,good_rois].T  # shape (n_cells, n_timestamps)
        
        # Interpolate traces from these timestamps to reference timestamps
        these_timestamps = this_plane['dff'].timestamps[:]
        f_interp = scipy.interpolate.interp1d(
            these_timestamps, dff_traces, 
            kind='linear', bounds_error=False, fill_value="extrapolate",
            axis=-1, # timestamp axis
        )
        interp_dff_traces = f_interp(refr_timestamps)
        
        all_dff_traces.extend(interp_dff_traces)
        all_roi_ids.extend(good_rois)
        all_plane_ids.extend([plane_id] * len(good_rois))
        
    all_dff_traces = np.array(all_dff_traces).T                       # (n_timestamps, n_total_cells)
    all_roi_ids = np.array(all_roi_ids, dtype=int)                    # (n_total_cells,)
    all_plane_ids = np.array(all_plane_ids, dtype=int)                # (n_total_cells,)
    all_volume_ids = np.ones(len(all_plane_ids), dtype=int) * volume  # (n_total_cells,)
    all_column_ids = np.ones(len(all_plane_ids), dtype=int) * column  # (n_total_cells,)
    
    # ------------------------------------------------------------------------------------
    # Load behavioral data
    # ------------------------------------------------------------------------------------
    this_behavior = nwb.processing['behavior'].data_interfaces
    all_behavior_names, all_behavior_traces = [], []
    for behavior_name in this_behavior.keys():
        if behavior_name == 'running_speed':
            behavior_traces = this_behavior[behavior_name].data[:]
            these_timestamps = this_behavior[behavior_name].timestamps[:]

        elif behavior_name == 'pupil':
            behavior = this_behavior[behavior_name].to_dataframe()
            behavior_traces = behavior['area'].values
            these_timestamps = behavior['timestamps'].values

        else:  # ignore corneal_reflection
            continue

        # Align timestamps to reference timestamp
        f_interp = scipy.interpolate.interp1d(
            these_timestamps, behavior_traces, 
            kind='linear', bounds_error=False, fill_value="extrapolate",
            axis=-1, # timestamp axis
        )
        behavior_traces = f_interp(refr_timestamps)

        # Interpolate over any NaNs in the traces
        nan_idxs = np.nonzero(np.isnan(behavior_traces))[0]
        if len(nan_idxs) > 0:
            warnings.warn(
                f"{behavior_name}: {len(nan_idxs)} NaNs detected; fillin in via interpolation."
            )
            valid_idxs = np.nonzero(~np.isnan(behavior_traces))[0]
            behavior_traces[nan_idxs] = np.interp(
                nan_idxs, valid_idxs, behavior_traces[valid_idxs]
            )

        all_behavior_names.append(behavior_name)
        all_behavior_traces.append(behavior_traces)

    all_behavior_names = np.array(all_behavior_names)    # shape (n_behaviors,)
    all_behavior_traces = np.stack(all_behavior_traces, axis=-1)  # shape (n_timestamps, n_behaviors)

    return {
        "timestamps": refr_timestamps,
        "dff_traces": all_dff_traces,
        "roi_ids": all_roi_ids,
        "plane_ids": all_plane_ids,
        "volume_ids": all_volume_ids,
        "column_ids": all_column_ids,
        "behavior_traces": all_behavior_traces,
        "behavior_names": all_behavior_names,
    }


def get_epoch_data(session_data, start_time, stop_time):
    """Return session data for a specific epoch of time, [start_time, stop_time).

    Parameters
    ----------
    session_data : dict, including
        - `timestamps` : ndarray, shape (T,)
        - `*_traces` : ndarray, shape (T,...)

    start_time : float
    stop_time : float
        Start and stop times to filter data by.
    
    Returns
    -------
    epoch_data : dict.
        Same members as `session_data`, but time-varying values are filtered as follows:
            - `timestamps` : list of ndarrays, each with shape (t_i,)
            - `*_traces` : list of ndarray, each with shape (..., t_i)
        where `t_i` is the number of timestamps between `start_times[i]` and `stop_times[i]`
        
        Non-time-varying key-value pairs in `session_data` are copied as is.

    """
    
    timestamps = session_data['timestamps']
    mask = (timestamps >= start_time) & (timestamps <= stop_time)

    epoch_data = {}
    for k, v in session_data.items():
        # if time-varying, filter/mask
        # an alternative identify by where len(v) == T
        if (k=='timestamps') or ('_traces' in k):
            epoch_data[k] = v[mask]
        else:
            epoch_data[k] = v

    return epoch_data


def bin_and_avg(arr, timestamps, window_size, window_overlap=0.):
    """Sliding window average of data, based on timestamps.

    Parameters
    ----------
    arr : np.ndarray, shape (T,).
        Array to bin and average.

    timestamps : np.ndarray, shape (T,)
        Timestamps to calculate bins.

    window_size : float.
        Window size, in same units as timestamps.

    window_overlap : float, default=0.
        Window overlap, in same units as `timestamps`.
        Default : 0, no overlap.

    Returns
    -------
    binned_arr : np.ndarray, shape (T_bins,...).
        Arrays with values binned and averaged.

    bin_starts : np.ndarray, shape (T_bins,).
        Array of timestamps associated with bin starts.

    """

    T = len(timestamps)
    if len(arr) != T:
        raise ValueError(
            f"Expect arrays to have length {T}, but got shape={arr.shape}."
        )

    # Calculate bin start times
    step_size = window_size - window_overlap
    bin_starts = np.arange(timestamps[0], timestamps[-1] - window_size + step_size, step_size)
    bin_ends = bin_starts + window_size
    
    binned_arr = np.zeros((len(bin_starts),) + arr.shape[1:])
    for i_bin, (start, end) in enumerate(zip(bin_starts, bin_ends)):
        # Find indices of timestamps within the current bin
        mask = (timestamps >= start) & (timestamps <= end)

        if mask.sum() > 0:  # Calculate the mean of the array within the current bin
            binned_arr[i_bin] = arr[mask].mean(axis=0)
        
        else: # Else, empty bin
            binned_arr[i_bin] = np.nan

    return binned_arr, bin_starts

def bin_and_avg_data(session_data, window_size, window_overlap=0.):
    """Helper function for applying `bin_and_avg` to session data dictionary."""

    binned_dd = {}
    for k, v in session_data.items():
        if (k=='timestamps'):  # we'll handle timestamps at the end
            continue
        elif '_traces' in k:       # apply binning
            binned_v, binned_ts = bin_and_avg(
                v, session_data['timestamps'], window_size, window_overlap
            )
            binned_dd[k] = binned_v
        else:  # else, do nothing
            binned_dd[k] = v
    
    # addd binned_ts
    binned_dd['timestamps'] = binned_ts

    return binned_dd
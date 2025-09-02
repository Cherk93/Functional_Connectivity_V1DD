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
    volume : int.
        Imaging volume (depth), 1 = top volume, 5 = bottom volume..

    remove_known_bad_planes : bool.
        Remove plane 6 of (column 1, volume 5).
        Default: True.

    column : int.
        Imaging column. 1 = center, 2-5 = surround.
        Default: 1, center column.

    subject_id : int
        Default: 409828, i.e. the Golden Mouse

    Returns
    -------
    dict of np.ndarray, consisting of:
        timestamps : shape (n_timestamps,)
            Timestamps that all `*_traces` are aligned to.

        dff_traces : shape (n_cells, n_timestamps)
            dF/F traces for all valid neurons across all planes, aligned to `timestamps`

        plane_ids : shape (n_cells,)
            Unique ID of plane of cell, ranges from 1-6

        roi_ids
            Unique ID of cell, for the given plane

        behavior_traces : shape (n_behaviors, n_timestamps)
            Raw / unprocessed behavioral variable traces
        
        behavior_names : shape (n_behaviors,)
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
        
    all_dff_traces = np.array(all_dff_traces)               # (n_total_cells, n_timestamps)
    all_plane_ids = np.array(all_plane_ids, dtype=int)      # (n_total_cells,)
    all_roi_ids = np.array(all_roi_ids, dtype=int)          # (n_total_cells,)
    
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
    all_behavior_traces = np.array(all_behavior_traces)  # shape (n_behaviors, n_timestemps)

    return {
        "timestamps": refr_timestamps,
        "dff_traces": all_dff_traces,
        "plane_ids": all_plane_ids,
        "roi_ids": all_roi_ids,
        "behavior_traces": all_behavior_traces,
        "behavior_names": all_behavior_names,
    }
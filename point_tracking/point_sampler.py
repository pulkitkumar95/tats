import os
import time
import pickle
import numpy as np
from functools import partial


def get_random_points(num_points, points_to_sample):
    if type(points_to_sample) == int:
        return np.random.choice(num_points, points_to_sample, replace=False)
    sampled_points = {}
    for num_sample in points_to_sample:
        sampled_points[num_sample] = np.random.choice(num_points, num_sample,
                                                      replace=False)
    return sampled_points



def point_sampler(pred_tracks, pred_visibility, per_point_queries,
                   points_to_sample=256, sampling_type='random', index_select=None):
    """Take any number of points, but sample only N points
    
    Args:
        pred_tracks (np.ndarray, float): Predicted points (T x N x 2)
        pred_visibility (np.ndarray, bool): Predicited points visibility (T x N)
        per_point_queries (np.ndarray, int) : Frame at which the point was queried (N)
        points_to_sample (int): Number of points to sample.
        sampling_type (str): Type of sampling to use. 
    Return:
        filtered_points (np.ndarray, bool): Points that are sampled.
        point_order (np.ndarray, int): Order of the points sampled. (points_to_sample)

    """

    if index_select is not None:
        pred_tracks = pred_tracks[index_select]
        pred_visibility = pred_visibility[index_select]
    
    num_points = pred_tracks.shape[1]
    filtered_points = np.zeros(num_points, dtype=bool) # All points are False initially.
    ids_to_consider = None
    
    if sampling_type == 'random':
        sampled_points = get_random_points(num_points, points_to_sample)
    else:
        raise NotImplementedError(f"Sampling type {sampling_type} not implemented")
    if type(sampled_points) == dict:
        final_dict = {}
        for num_points, sampled_indices in sampled_points.items():
            final_dict[num_points] = {}
            final_dict[num_points]['sampled_indices'] = sampled_indices
            final_dict[num_points]['ids_to_consider'] = ids_to_consider[sampled_indices] if ids_to_consider is not None else None
        
        return final_dict, None
    filtered_points[sampled_points] = True
    point_order = sampled_points
    return filtered_points, point_order


def run_point_sampler( points_to_sample=256, pt_dict=None):
    pred_tracks = pt_dict['pred_tracks'].squeeze(0) # t x N x 2
    pred_visibility = pt_dict['pred_visibility'].squeeze(0) # t x N
    per_point_queries = pt_dict['per_point_queries'] # N
    point_sampler_fn = partial(point_sampler, pred_tracks, pred_visibility, 
                                                            per_point_queries)
    
    num_points_to_samples = [points_to_sample]
    sampling_types = ['random']
    points_sampled = {}
    start_time = time.time()
    for sampling_type in sampling_types:

        sampled_points, _ = point_sampler_fn(points_to_sample=num_points_to_samples, 
                                                sampling_type=sampling_type)
        points_sampled[sampling_type] = sampled_points

    pt_dict.update(points_sampled)
    return pt_dict
    
        

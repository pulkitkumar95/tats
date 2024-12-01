import numpy as np

from contextlib import contextmanager

@contextmanager
def temp_seed(seed):
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            np.random.set_state(state)

def visible_point_distance(points, visible_points):
    # Filter out the visible points
    visible_indices = np.where(visible_points)[0]
    visible_points = points[visible_indices]

    # Calculate pairwise distances between consecutive visible points
    pairwise_distances = np.linalg.norm(np.diff(visible_points, axis=0), axis=1)
    # Sum up the distances to get the total distance
    total_dist = np.sum(pairwise_distances)
    return total_dist

def get_distance_of_all_points(pred_tracks, pred_visibility):
    num_points = pred_tracks.shape[1]
    all_points_distance = []
    for i in range(num_points):
        points = pred_tracks[:,i]
        visible_points = pred_visibility[:,i]
        point_distance = visible_point_distance(points, visible_points)
        all_points_distance.append(point_distance)
    return np.array(all_points_distance)

def get_point_query_mask(point_queries, init_mask):

    temporal_length = init_mask.shape[0]
    point_query_masks = []
    for point_query in point_queries:
        point_mask = np.zeros(temporal_length, dtype=bool)
        point_mask[point_query:] = True
        point_query_masks.append(point_mask[:, None])
    point_query_masks = np.concatenate(point_query_masks, axis=1)

    return point_query_masks
  
        


def stratified_sampling(data, num_samples):
    try:
        hist_counts, hist_bins = np.histogram(data, bins='auto')
    except:
        # if auto fails, use the default 10 bins
        hist_counts, hist_bins = np.histogram(data)
    points_per_bin = num_samples // len(hist_counts)
    sampled_points = []
    
    for bin_idx in range(len(hist_counts)):
        bin_start = hist_bins[bin_idx]
        bin_end = hist_bins[bin_idx + 1]
        bin_indices = np.where((data >= bin_start) & (data < bin_end))[0]
        sampled_indices = np.random.choice(bin_indices, min(points_per_bin, len(bin_indices)), replace=False)
        sampled_points.extend(sampled_indices)

    remaining_samples = num_samples - len(sampled_points)
    all_indices = np.arange(len(data))
    remaining_indices = np.setdiff1d(all_indices, sampled_points)
    additional_samples = np.random.choice(remaining_indices, remaining_samples, replace=False)
    sampled_points.extend(additional_samples)

    return np.array(sampled_points)



def point_sampler(cfg, pt_dict, pred_tracks, pred_visibility, per_point_queries,
                   points_to_sample=256, sampling_type='random', index_select=None,
                   split='train'):
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

    num_points = pred_tracks.shape[1]
    filtered_points = np.zeros(num_points, dtype=bool) # All points are False initially.
    point_order = np.arange(points_to_sample, dtype=int)
    if (split=='train' and cfg.POINT_INFO.PT_FIX_SAMPLING_TRAIN) or (split=='test' and cfg.POINT_INFO.PT_FIX_SAMPLING_TEST):
        fix_sample_stratergy = sampling_type
        fix_sample_num_points = points_to_sample
        sampled_dict = pt_dict[fix_sample_stratergy][fix_sample_num_points]
        sampled_indices = sampled_dict['sampled_indices']
        ids_to_consider = sampled_dict['ids_to_consider']
        if ids_to_consider is None:
            ids_to_consider = np.arange(num_points).astype(int)
        filtered_points[sampled_indices] = True

        return filtered_points, ids_to_consider


    if index_select is not None:
        pred_tracks = pred_tracks[index_select]
        pred_visibility = pred_visibility[index_select]
    
    if sampling_type == 'random':
        sampled_points = np.random.choice(num_points, points_to_sample, replace=False)
        filtered_points[sampled_points] = True
    elif sampling_type == 'stratified':

        pred_tracks_normalised = pred_tracks / pred_tracks.max()
        all_points_distance = get_distance_of_all_points(pred_tracks_normalised, pred_visibility)
        sampled_points = stratified_sampling(all_points_distance, points_to_sample)
        filtered_points[sampled_points] = True
        distances_sampled = all_points_distance[filtered_points]
        point_order = np.argsort(distances_sampled)
    else:
        raise NotImplementedError(f"Sampling type {sampling_type} not implemented")
    return filtered_points, point_order
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import pandas as pd
import pickle
from vid_readers import get_vid_info
from omni_vis import make_vis
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from tqdm import tqdm
from point_sampler import run_point_sampler

def filter_points(points, point_queries, point_visibility, query_frames, max_dist=16):
    """
    Filter out points that are not redundant when new points are queried.
    This is just based on simple distance for now.

    args:
        points: np.array of shape (T, N, 2)
        query_frames: list of frame_ids to queried for point extraction
    """
    points = points.squeeze(0).cpu().numpy()
    point_queries = point_queries.cpu().numpy()
    point_visibility = point_visibility.squeeze(0).cpu().numpy()

    points_to_take_flag = np.zeros_like(point_queries, dtype=bool)
    points = rearrange(points, 't n c -> n t c')
    point_visibility = rearrange(point_visibility, 't n -> n t')
    query_frame = query_frames[0]
    points_to_take_flag[point_queries == query_frame] = True
    for query_frame in query_frames[1:]:
        prev_points = points[points_to_take_flag].copy()
        prev_points_visibility = point_visibility[points_to_take_flag]
        #checking if the visibility of previous points at qeury frame
        # if not visibile, discard from the check
        prev_points_visibility = prev_points_visibility[:, query_frame]
        prev_points = prev_points[prev_points_visibility]
        

        new_query_point_indices = np.argwhere(point_queries == query_frame)[:, 0]
        new_points = points[new_query_point_indices].copy()
        if prev_points.shape[0] == 0:
            # All points vanished, take all new points.
            points_to_take_flag[new_query_point_indices] = True
            continue

        # only compute distances for points that are not already taken
        # also only till from the start of the new points
        prev_points_from_qeury = prev_points[:, query_frame:]
        new_points_from_query = new_points[:, query_frame:]
        dists = np.linalg.norm(new_points_from_query[:, None] - 
                                prev_points_from_qeury[None], axis=-1)
        #Average distance of each point with every other point
        dists = np.mean(dists, axis=2)
        min_dists = np.min(dists, axis=1)
        points_to_take_flag[new_query_point_indices] = min_dists > max_dist

    return points_to_take_flag


# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        # default="./checkpoints/cotracker.pth",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=16, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )


    parser.add_argument(
        "--csv_to_process",
        type=str,
        required=True,
        help="Dataset csv to process",
    )

    parser.add_argument(
        "--query_type",
        type=str,
        default='uniform',
        help="Dataset csv to process",
    )


    parser.add_argument(
        "--make_vis",
        action="store_true",
        help="Make visualisation of the tracks",
    )


    parser.add_argument(
        "--uniform_chunk_size",
        type=int,
        default=8,
        help="Number of chunks to divide the video into",
    )

    parser.add_argument(
        "--num_points_to_sample",
        type=int,
        default=256,
        help="Number of points to sample",
    )
    
    parser.add_argument(
        "--base_feat_dump_path", 
        type=str,
        default='/fs/cfar-projects/actionloc/bounce_back/point_tracking/feat_dumps/',
        help="Base path for feature dump files",
    )

    parser.add_argument(
        "--base_vis_dump_path",
        type=str, 
        default='/fs/cfar-projects/actionloc/bounce_back/point_tracking/vis_dumps/',
        help="Base path for visualization dump files",
    )

    

    args = parser.parse_args()
    if args.query_type != 'beginning':
        query_str = f'_{args.query_type}'
        if args.query_type == 'uniform':
            query_str += f'_{args.uniform_chunk_size}'
        if args.query_type == 'interval':
            query_str += f'_{args.interval}'
    else:
        query_str = ''
  
    point_tracker_name = f'cotracker3_{args.grid_size}{query_str}_corrected'
    csv_path = args.csv_to_process
    if not csv_path.endswith('.csv'):
        csv_path = csv_path + '.csv'


    ds_df = pd.read_csv(csv_path).iloc[args.start_idx:args.end_idx + 1]
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(DEFAULT_DEVICE)
    print(f"Processing {len(ds_df)} videos")
    for index, row in tqdm(ds_df.iterrows()):
        base_vid_info = row.to_dict()   
        vid_info = get_vid_info(base_vid_info)
       
        vid_feat_dump_path = vid_info['vid_feat_dump_path']
        dataset = vid_info['dataset']
        vid_feat_dump_path = os.path.join(args.base_feat_dump_path, dataset, 
                                          point_tracker_name, vid_feat_dump_path)
        dump_dict = {}
        if os.path.exists(vid_feat_dump_path):
            print(f"Skipping {vid_feat_dump_path}")
            continue
        if os.path.exists(vid_feat_dump_path):
            dump_dict = pickle.load(open(vid_feat_dump_path, 'rb'))
        os.makedirs(os.path.dirname(vid_feat_dump_path), exist_ok=True)
     
                
        video = vid_info['video'].copy()
        height, width = video.shape[1:3]

        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        use_video = video[0].permute(0, 2, 3, 1).numpy().astype(np.uint8)



        video = video.to(DEFAULT_DEVICE)
        if 'mask' in vid_info:
            segm_mask = vid_info['mask']
            segm_mask = torch.from_numpy(segm_mask).float()[None, None].to(DEFAULT_DEVICE)
        else:
            segm_mask = None
        if args.query_type == 'beginning':
            queried_frames = [0]
        elif args.query_type == 'middle':
            queried_frames = [0, video.shape[1]//2]
        elif args.query_type == 'uniform':
            queried_frames = np.linspace(0, video.shape[1] - 1, args.uniform_chunk_size).astype(int)[:-1]
        elif args.query_type == 'interval':
            queried_frames = list(range(0, video.shape[1] - 1 , args.interval))
        elif args.query_type == 'bbox':
            assert dataset == 'finegym', 'bbox query only supported for finegym'
            queried_frames = [0]



        else:
            raise ValueError(f'Unknown query type {args.query_type}')

        
        # video = video[:, :20]
        
        pred_tracks = []
        pred_visibility = []
        og_visibilty = []
        per_point_queries = []
        for query_frame in queried_frames:
            pred_tracks_query, pred_visibility_query = model(
                video,
                grid_size=args.grid_size,
                grid_query_frame=query_frame,
                segm_mask=segm_mask
            )
            num_points = pred_tracks_query.shape[2]
            pred_visibility_query[:, :query_frame] = False
            pred_tracks.append(pred_tracks_query)
            pred_visibility.append(pred_visibility_query)
            per_point_query = torch.ones(num_points) * query_frame
            per_point_queries.append(per_point_query)
        pred_tracks = torch.cat(pred_tracks, dim=2)
        pred_visibility = torch.cat(pred_visibility, dim=2)
        per_point_queries = torch.cat(per_point_queries, dim=0)

        filtered_points = filter_points(pred_tracks, per_point_queries, 
                                                pred_visibility,  queried_frames)

        vid_feat_dump_path = vid_info['vid_feat_dump_path']
        dataset = vid_info['dataset']
        vid_feat_dump_path = os.path.join(args.base_feat_dump_path, dataset, 
                                        point_tracker_name, vid_feat_dump_path)
        os.makedirs(os.path.dirname(vid_feat_dump_path), exist_ok=True)
        # save info of only filterd points
        total_points = pred_tracks.shape[2]
        pred_tracks = pred_tracks[:, :, filtered_points]
        pred_visibility = pred_visibility[:,:, filtered_points]
        per_point_queries = per_point_queries[filtered_points]

        dump_dict = {
            "pred_tracks": pred_tracks.cpu().half(),
            "pred_visibility": pred_visibility.cpu(), 
            'per_point_queries': per_point_queries.numpy().astype(int),
            'total_points': total_points,
            }
        dump_dict = run_point_sampler(points_to_sample=args.num_points_to_sample, pt_dict=dump_dict)


            
        if args.make_vis:
            vid_gif_path = vid_info['gif_path']
            vid_gif_path = os.path.join(args.base_vis_dump_path, dataset, 
                                        point_tracker_name, vid_gif_path)
            print(vid_gif_path)
            os.makedirs(os.path.dirname(vid_gif_path), exist_ok=True)
            make_vis(vid_info, pred_tracks,pred_visibility, per_point_queries, 
                                                None, vid_gif_path,
                                                use_video=use_video)
        if not args.make_vis:
            pickle.dump(dump_dict, open(vid_feat_dump_path, 'wb'))

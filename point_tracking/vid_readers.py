import os
import cv2
import torch
import math
import json
import numpy as np
from PIL import Image
from decord import VideoReader

def transform_bounding_box(bbox, original_size, new_size):
    """
    Transform bounding box coordinates from original image size to new image size.
    
    Args:
    bbox (tuple): Original bounding box (x, y, w, h) where x and y can be either center or top-left corner
    original_size (tuple): Original image size (width, height)
    new_size (tuple): New image size (width, height)
    
    Returns:
    mask (np.array): Mask of the bounding box in the new image size
    """
    x, y, w, h = bbox
    orig_w, orig_h = original_size
    new_w, new_h = new_size
    
    # Scale factors
    w_scale = new_w / orig_w
    h_scale = new_h / orig_h
    
    # Transform coordinates
    new_w = w * w_scale
    new_h = h * h_scale
    new_x = x * w_scale
    new_y = y * h_scale
    
    # Prepare different formats
    center_x = new_x + new_w / 2
    center_y = new_y + new_h / 2

    mask = np.zeros((new_size[0], new_size[1]))
    mask[int(new_y):int(new_y + new_h), int(new_x):int(new_x + new_w)] = 1
    mask = mask.astype(bool)
    mask = mask * 255
    
    return  mask


def get_vid_info(vid_info, max_fps=10, make_vis=False, load_video=True):
    video_path = vid_info['video_path']
    dataset = vid_info['dataset']
    if load_video:
        if 'ek100' in dataset:
            video, fps = read_epic_kitchens_video(vid_info, max_fps)
        elif video_path.endswith('.webm'):
            video, fps = read_webm_video(video_path)
        else:
            # For other video formats, we use decord to read the video with max fps 10
            video, fps = read_video(video_path, max_fps=max_fps)
        vid_info['video'] = video
    else:
        fps = None
    
    if 'kinetics' in dataset:
        split, label, vid_name = video_path.split('/')[-3:]
        vid_feat_name = vid_name.split('.')[0] + '.pkl'
        split = split.split('_')[0]
        label = label.replace('_', ' ')
        vid_feat_dump_path = os.path.join(split, label, vid_feat_name)
    elif 'ek100' in dataset:
        vid_name = vid_info['narration_id']
        vid_feat_dump_path = f'{vid_name}.pkl'
    
    else:
        vid_name = video_path.split('/')[-1].split('.')[0]
        vid_feat_dump_path = f'{vid_name}.pkl'

    if 'gif_path' not in vid_info:
        vid_info['gif_path'] = vid_name + '.gif' # for visualisation
    vid_info['vid_feat_dump_path'] = vid_feat_dump_path
    vid_info['fps'] = fps
    if 'gt_bbox' in vid_info:
        original_bbox = vid_info['gt_bbox']  
        original_bbox = np.array(json.loads(original_bbox))
        # Taking only the bbox in the first frame
        original_bbox = original_bbox[0]
        original_size = json.loads(vid_info['img_shape'])
        new_size = vid_info['video'][0].shape[:2]
        bbox_mask = transform_bounding_box(original_bbox, original_size, new_size)
        vid_info['mask'] = bbox_mask

        
    return vid_info

def read_video(video_path, max_fps=10, indices_to_take=None):
   
    vr = VideoReader(video_path)
    fps = np.around(vr.get_avg_fps())
    num_frames = len(vr)
    duration = vr.get_frame_timestamp(num_frames-1)[-1]
    if duration < 0:
        cap = cv2.VideoCapture(video_path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    frame_start = 0 
    frame_end = len(vr) - 1
    frames_in_clip = int(duration * max_fps)
    try:
        frames_to_take = np.linspace(frame_start, frame_end, frames_in_clip).astype(int)
    except:
        breakpoint()
    if indices_to_take is not None:
        frames_to_take = frames_to_take[indices_to_take]
    frames = [vr[i].asnumpy()[None] for i in frames_to_take]
    frames = np.concatenate(frames, axis=0)
    return frames, max_fps

def read_webm_video(webm_file_path):
    cap = cv2.VideoCapture(webm_file_path)
    # get fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()

    frames = []

    # Read and store each frame in a list
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #convert to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame[None])

    # Release the VideoCapture object
    cap.release()

    # Convert the list of frames to a NumPy array
    video_array = np.concatenate(frames, axis=0)
    return video_array, fps

def get_epic_frame_path(vid_path, frame_num):
    frame_path = os.path.join(vid_path, f'frame_{frame_num:010d}.jpg')
    return frame_path

def read_epic_kitchens_video(video_stuff, max_fps=10):
    vid_path = video_stuff['video_path']
    start_frame = video_stuff['start_frame']

    end_frame = video_stuff['stop_frame']
    vid_fps = video_stuff['fps']
    #fixing fps issue with epic vid_info file
    vid_fps = 60 if vid_fps != 50 else 50
    if vid_fps > max_fps:
        step = int(np.round(vid_fps/max_fps))
    else:
        step = 1
    frames = []

    for frame_num in range(start_frame, end_frame+1, step):
        frame_path = get_epic_frame_path(vid_path, frame_num)
        frame = Image.open(frame_path)
        frame = np.array(frame).astype(np.uint8)
        frames.append(frame[None])
    frames = np.concatenate(frames, axis=0)
    return frames, max_fps




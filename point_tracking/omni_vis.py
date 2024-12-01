import cv2
import imageio
import colorsys
import numpy as np
import torch
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

def squeeze_and_convert_to_numpy(tensor):
    if type(tensor) == torch.Tensor:
        tensor = tensor.cpu().squeeze().numpy()
    return tensor


def make_vis(vid_info, keypoints, keypoints_vis, keypoint_queries, point_to_take=None, gif_path=None, use_video=None, line_thickness=1):
    if use_video is not None:
        video = use_video
        keypoints = keypoints[0].cpu().numpy()
        keypoints_vis = keypoints_vis[0].cpu().numpy()
        try:
            keypoint_queries = keypoint_queries.cpu().numpy()
        except:
            pass
        point_to_take = None
    else:
        video = vid_info['video'].copy()
    frames = vis_trail(video, keypoints, keypoints_vis, keypoint_queries, point_to_take, vid_info['fps'],
                       line_thickness=line_thickness)
    if gif_path is None:
        return frames
    frames = [frames] # make it a list of lists for vis
    save_final_gif(frames, gif_path)
    return frames[0]
    
def add_point_info_to_frame(frame, point_info_name):
    image_width, image_height = frame.size
    new_image = Image.new("RGB", (image_width, image_height + 20), (255, 255, 255)) 
    new_image.paste(frame, (0, 20))
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.truetype("Helvetica.ttf", size=12)
    text = point_info_name
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    # Calculate the position to center the text on the strip
    text_position = ((image_width - text_width) // 2, 5)  # Centered horizontally, 5 pixels from the top
    draw.text(text_position, text, font=font, fill=(0, 0, 0))  # You can specify the text color
    return new_image

def save_final_gif(frames, gif_path, point_infos_to_use=None):
    if point_infos_to_use and len(point_infos_to_use)>1:
        final_frames = []
        for frame_num in range(len(frames[0])):
            curr_frame = []
            for vid_num in range(len(frames)):
                frame_to_use = frames[vid_num][frame_num]
                frame_to_use = add_point_info_to_frame(frame_to_use, 
                                                    point_infos_to_use[vid_num])
                frame_to_use = np.array(frame_to_use)

                h, w = frame_to_use.shape[:2]
                curr_frame.append(frame_to_use)
                padding = np.zeros((h, 10, 3)) * 255
                curr_frame.append(padding.astype(np.uint8))

            curr_frame = Image.fromarray(np.hstack(curr_frame))

            final_frames.append(curr_frame)
    else:
        final_frames = frames[0]
    imageio.mimsave(gif_path, final_frames, loop=0, duration = 0.01)


def vis_trail(video, kpts, kpts_vis, kpts_queries=None,point_to_take=None, fps=10, cluster_ids=None,
line_thickness=1):
    """
    This function calculates the median motion of the background, which is subsequently
    subtracted from the foreground motion. This subtraction process "stabilizes" the camera and
    improves the interpretability of the foreground motion trails.

    Args:
        video (np.ndarray): Video frames (T, H, W, C)
        kpts (np.ndarray): Keypoints (T, N, 2)
        kpts_vis (np.ndarray): Keypoint visibility (T, N)
        kpts_queries (np.ndarray): Frame at which the point was queried (N)
        fps (float): Frames per second
    """
    color_map = cm.get_cmap("jet")

    images = video
    max_height = 200
    if point_to_take is None:
        point_to_take = np.ones(kpts.shape[1], dtype=bool)
    if kpts_queries is None:
        kpts_queries = np.zeros(kpts.shape[1], dtype=int)
    if kpts_vis is None:
        kpts_vis = np.ones(kpts.shape[:2], dtype=bool)

    frames = []
    back_history = 1*fps

    #sample only the points that are needed to be taken 
    kpts = kpts[:, point_to_take]
    kpts_vis = kpts_vis[:, point_to_take]
    kpts_queries = kpts_queries[point_to_take]
    point_to_take = point_to_take[point_to_take]
    
    num_imgs, num_pts = kpts.shape[:2]

    for i in range(num_imgs):

        img_curr = images[i]

        for t in range(i):


            img1 = img_curr.copy()
            # changing opacity
            if i - t < back_history:
                alpha = max(1 - 0.9 * ((i - t) / ((i + 1) * .99)), 0.1)
            else:
                alpha = 0.0
            # alpha = 0.6

            for j in range(num_pts):
                if (kpts_queries[j] > t) or (not point_to_take[j]):
                    continue
                if (kpts_vis[t:, j] == 0).all():
                    continue

                if cluster_ids is not None:
                    color  = np.array(color_map(cluster_ids[j]/ max(cluster_ids))[:3]) *255
                else:
                    color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255

                color_alpha = 1

                hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
                color = colorsys.hsv_to_rgb(hsv[0], hsv[1]*color_alpha, hsv[2])

                pt1 = kpts[t, j]
                pt2 = kpts[t+1, j]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))

                cv2.line(img1, p1, p2, color, thickness=line_thickness, lineType=16)

            img_curr = cv2.addWeighted(img1, alpha, img_curr, 1 - alpha, 0)

        for j in range(num_pts):
            if (kpts_queries[j] > i) or (not point_to_take[j]):
                    continue
            if (kpts_vis[i:, j] == 0).all():
                 continue
            if cluster_ids is not None:
                color  = np.array(color_map(cluster_ids[j]/ max(cluster_ids))[:3]) *255
            else:
                color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255
            pt1 = kpts[i, j]
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            cv2.circle(img_curr, p1, 2, color, -1, lineType=16)
        
        # height, width, _ = img_curr.shape
        # if height > max_height:
        #     new_width = int(width * max_height / height)
        # else:
        #     new_width = width
        # img_curr = cv2.resize(img_curr, (new_width, max_height))
        
        frames.append(Image.fromarray(img_curr.astype(np.uint8)))
    
    return frames

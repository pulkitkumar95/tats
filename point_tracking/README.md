# Point Extraction using CoTracker

This code implements point tracking and extraction using the CoTracker framework. While the original paper used CoTracker2, this implementation uses CoTracker3. Our supplementary materials demonstrate that the results are not significantly dependent on the specific version of point tracker used.

## Overview

The code provides functionality to:
- Extract and track points across video frames
- Filter redundant points based on spatial proximity
- Support different point querying strategies:
  - Beginning: Query points only from the first frame
  - Middle: Query points from first frame and middle frame
  - Uniform: Query points uniformly across video duration
  - Interval: Query points at regular intervals
  - Bbox: Query points within bounding boxes (FineGym dataset only)

## Key Features

- Uses CoTracker3 for robust point tracking
- Supports various input video formats
- Includes point filtering to remove redundant tracks
- Optional visualization of tracked points
- Configurable grid size and sampling parameters
- GPU acceleration support (CUDA)

## Dependencies

Please refer to the [CoTracker repository](https://github.com/facebookresearch/co-tracker) for complete dependency information. 

## Input Format

The script expects a CSV file containing video information. The CSV must include a `video_path` column that contains the full path to each video file that needs to be processed for point extraction.

## Usage

The script can be run from the command line with various arguments to control the point extraction process. Key parameters include:
- `--grid_size`: Size of the regular grid for point sampling
- `--query_type`: Strategy for selecting frames to query points from
- `--num_points_to_sample`: Number of points to sample
- `--make_vis`: Flag to generate visualizations
- `--csv_to_process`: Path to the CSV file containing video information


For example, the following command shows the configuration used for all experiments in our paper:

```bash
python extract_points.py --grid_size 16 --query_type uniform --num_points_to_sample 256 --csv_to_process /path/to/csv/file.csv
```

### Path Configuration
- `--base_feat_dump_path`: Base directory path for dumping extracted features
- `--base_vis_dump_path`: Base directory path for saving visualizations

## Note

While this implementation uses CoTracker3, our main paper uses CoTracker2. We have found that the results are not significantly dependent on the specific version of point tracker used and even the type of point tracker used. Please refer to the supplementary materials for more details.

import numpy as np
from pathlib import Path
import open3d as o3d
from rosbags.rosbag2 import Reader
# 1. Import the modern TypeStore instead of direct serde functions
from rosbags.typesys import Stores, get_typestore

# Define paths and topic
bag_path = Path("bags/cloud_odom_verlab_dcc/cloud_odom_verlab_dcc_0.db3")
topic_to_extract = "/livox/scan"   
frame_idx = 0

# Initialize the ROS2 Humbly type store (works across almost all ROS2 db3 versions)
typestore = get_typestore(Stores.ROS2_HUMBLE)

# 2. Open the bag file
with Reader(bag_path) as reader:
    # Find the connection object for your point cloud topic
    connections = [x for x in reader.connections if x.topic == topic_to_extract]
    
    if not connections:
        print(f"Topic '{topic_to_extract}' not found in the bag file.")
        exit()

    # Iterate sequentially through messages
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        
        # 3. Use the typestore object to deserialize the data safely
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        # Parse the PointCloud2 binary data manually into a NumPy array
        num_points = len(msg.data) // msg.point_step
        
        # Create a view of the raw data as rows matching point_step
        raw_bytes = np.frombuffer(msg.data, dtype=np.uint8).reshape(num_points, msg.point_step)
        
        # Slice X, Y, Z coordinates (assuming standard float32 offsets at 0, 4, 8)
        x_coords = np.frombuffer(raw_bytes[:, 0:4].tobytes(), dtype=np.float32)
        y_coords = np.frombuffer(raw_bytes[:, 4:8].tobytes(), dtype=np.float32)
        z_coords = np.frombuffer(raw_bytes[:, 8:12].tobytes(), dtype=np.float32)
        
        # Combine into an Nx3 array and clean up NaNs
        points = np.stack([x_coords, y_coords, z_coords], axis=-1)
        points = points[~np.isnan(points).any(axis=1)]
        
        if points.shape[0] == 0:
            continue
            
        # 4. Save to a highly optimized binary PLY file via Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        
        ply_filename = f"bags_out/{bag_path.stem}frame_{frame_idx:04d}.ply"
        o3d.io.write_point_cloud(ply_filename, pcd, write_ascii=False)
        
        print(f"Successfully saved {ply_filename} with {points.shape[0]} valid points")
        frame_idx += 1

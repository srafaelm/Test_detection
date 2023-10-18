from sensor_streaming import hl2ss, hl2ss_3dcv
import numpy as np
import threading
import copy


class SensorStreamer:
    def __init__(self, host):
        self.rgb = None
        self.depth = None
        self.host = host
        self.enable_streams = True

        # Start rgb and depth threads to exhaust the input stream frames
        t1 = threading.Thread(target=self._reader_rgb)
        t1.daemon = True
        t1.start()

        t2 = threading.Thread(target=self._reader_depth)
        t2.daemon = True
        t2.start()

    def _reader_rgb(self):
        # HoloLens address
        host = self.host

        # Camera parameters
        width = 640
        height = 360 #360
        framerate = 30
        profile = hl2ss.VideoProfile.H265_MAIN
        bitrate = 5 * 1024 * 1024

        # Setup RGB streamer
        hl2ss.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        rgb_client = hl2ss.rx_decoded_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss.ChunkSize.PERSONAL_VIDEO, hl2ss.StreamMode.MODE_1, width, height, framerate, profile, bitrate, 'bgr24')
        rgb_client.open()

        # Update images with latest frames
        while (self.enable_streams):
            self.rgb = rgb_client.get_next_packet()

        # Close stream
        rgb_client.close()
        hl2ss.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    def _reader_depth(self):
        # HoloLens address
        host = self.host

        # Camera parameters
        profile = hl2ss.VideoProfile.H265_MAIN
        # focus = 1000

        # Setup Depth streamer
        depth_client = hl2ss.rx_decoded_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.ChunkSize.RM_DEPTH_LONGTHROW, hl2ss.StreamMode.MODE_1, profile)
        depth_client.open()

        # Update images with latest frames
        while (self.enable_streams):
            self.depth = depth_client.get_next_packet()

        #Close stream
        depth_client.close()



def load_calibration_data(calibration_path):
    depth_calibration = hl2ss_3dcv._load_calibration_rm_depth_longthrow(calibration_path + "rm_depth_longthrow/")
    rgb_calibration = hl2ss_3dcv._load_calibration_pv(calibration_path + "personal_video/" + "1000_640_360/")
    rgb_extrinsics = hl2ss_3dcv._load_extrinsics_pv(calibration_path + "personal_video/")
    rgb_calibration = hl2ss_3dcv._Mode2_PV_E(rgb_calibration, rgb_extrinsics)

    # Reverse sign of RGB intrinsics fx and fy, to match the directions of the displaying axis
    fixed_rgb_calibration = copy.deepcopy(rgb_calibration)
    fixed_rgb_calibration.intrinsics[0, 0] = - fixed_rgb_calibration.intrinsics[0, 0]
    fixed_rgb_calibration.intrinsics[1, 1] = - fixed_rgb_calibration.intrinsics[1, 1]

    return depth_calibration, rgb_calibration, fixed_rgb_calibration


def get_depth_to_pv_map(depth_calibration, rgb_calibration, depth_image, rgb_image):
    # Get 3D coordinates of points in depth camera's frame with intrinsics and depth value
    xy1 = hl2ss_3dcv.to_homogeneous(hl2ss_3dcv.compute_uv2xy(depth_calibration.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT))
    depth = hl2ss_3dcv.rm_depth_normalize(depth_image, depth_calibration.undistort_map, depth_calibration.scale * np.linalg.norm(xy1, axis=2))
    xyz1 = hl2ss_3dcv.to_homogeneous(hl2ss_3dcv.rm_depth_to_points(depth, xy1))

    # Map 3D coordinates from depth camera's frame to RBB camera's frame
    depth_to_rgb = hl2ss_3dcv.projection(rgb_calibration.intrinsics, hl2ss_3dcv.camera_to_camera(depth_calibration.extrinsics, rgb_calibration.extrinsics))
    uv, _ = hl2ss_3dcv.project_to_image(xyz1, depth_to_rgb)
    u = uv[:, 0].reshape(depth.shape)
    v = uv[:, 1].reshape(depth.shape)
    depth[(u < 0) | (u > (rgb_image.shape[1] - 1)) | (v < 0) | (v > (rgb_image.shape[0] - 1))] = 0

    return depth, u, v


def get_point_depth(depth, u, v, point):
    has_neighbours = False
    radius = 5
    while not has_neighbours and radius < 50:
        # Get valid depth values around bbox point
        neighbor_points = depth[(u < np.floor(point[0] + radius)) & (u > np.floor(point[0] - radius)) & (v < np.floor(point[1] + radius)) & (v > np.floor(point[1] - radius))]
        neighbor_points = neighbor_points[neighbor_points != 0]

        # if there are no points inside this area increase radius and try again
        if len(neighbor_points) != 0:
            has_neighbours = True
        else:
            radius += 5
            continue

    if len(neighbor_points) == 0:
        return None
    return np.min(neighbor_points)


def get_3d_bbox(bbox, depth_image, u, v, rgb_calibration):
    # Get bbox center and add it to the array
    center = np.array([np.floor(np.average(bbox[:, 0])), np.floor(np.average(bbox[:, 1]))])
    bbox = np.vstack((bbox, center.reshape((1, 2))))

    # Get depth of bbox center
    center_depth = get_point_depth(depth_image, u, v, center)
    if center_depth is None:  # Point outside of depth image's FOV
        return None

    center_depth = center_depth + 0.1

    # Change bbox 2D coordinates to homogeneous coordinates (xz, yz, z), z being the depth of bbox center
    bbox_homogeneous = np.concatenate((bbox, center_depth * np.ones((5, 1))), axis=1)
    bbox_homogeneous[:, 0] = bbox_homogeneous[:, 0] * bbox_homogeneous[:, 2]
    bbox_homogeneous[:, 1] = bbox_homogeneous[:, 1] * bbox_homogeneous[:, 2]

    # Transform bbox coordinates to 3D in RGB camera's frame
    bbox_3d = bbox_homogeneous @ hl2ss_3dcv.image_to_camera(rgb_calibration.intrinsics[:3, :3])
    return bbox_3d


def transform_to_world(bbox_3d, rgb_pose):
    world_bbox = hl2ss_3dcv.to_homogeneous(bbox_3d) @ hl2ss_3dcv.reference_to_world(rgb_pose)
    return world_bbox[:, :3]

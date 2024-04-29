#roslaunch rtabmap_launch rtabmap.launch rtabmap_args:="--delete_db_on_start" rgb_topic:=/airsim_node/Hydrone/stereo/Scene depth_topic:=/airsim_node/Hydrone/stereo/DepthPerspective camera_info_topic:=/airsim_node/Hydrone/stereo/Scene/camera_info odom_topic:=/airsim_node/Hydrone/odom_local_ned imu_topic:=/airsim_node/Hydrone/imu/Imu visual_odometry:=false frame_id:=stereo_optical approx_sync:=false rgbd_sync:=true queue_size:=1000 scan_cloud_topic:=\points gen_cloud_voxel:=0.5
    # call(["gnome-terminal", "--", "sh", "-c", f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:={settings.ip}"])
    
    server = libtmux.Server()
    zero = server.new_session(session_name="bash", kill_session=True, attach=False)
    one = zero.new_window(attach=True, window_name="airsim_ros_pkgs")
    two = zero.new_window(attach=True, window_name="rtabmap")
    three = zero.new_window(attach=True, window_name="gym-airsim")
    
    bash = zero.active_pane
    airsimpkg = one.active_pane
    rtabmap = two.active_pane
    gym_airsim = three.active_pane
    
    
    airsimpkg.send_keys(f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:={settings.ip}")
    # pane2.send_keys('ls -al')

    print('oi')
    server.attach_session(target_session="bash")
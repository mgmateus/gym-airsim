

from . import PointOfViewTwins, DictToClass



class AirPointOfView:
    domain = 'air'
    markers = DictToClass({'name' : 'Cube'})
    altitude = -1.0

class GymPointOfView(Env):
    task_name = "point-of-view"
    max_episode_steps = 200

    @staticmethod
    def set_markers(markers_name, markers_num : int):
        if isinstance(markers_name, str):
            m = [ markers_name ]
            m = m + [f"{markers_name}{i}" for i in range(1, markers_num+1)]
            return set(m)
        
        am = [ markers_name[0] ]
        am = am + [f"{ markers_name[0]}{i}" for i in range(1, markers_num[0]+1)]
        um = [ markers_name[1] ]
        um = um + [f"{ markers_name[1]}{i}" for i in range(1, markers_num[1]+1)]
        return set(am+um)
    
    def __init__(self, observation_type : str, 
                 observation_stack : int, 
                 pre_aug : tuple, 
                 domain : str,
                 ue4 : str, 
                 markers : DictToClass,
                 target_range : DictToClass):
        
        Simulation.__init__(self, observation_type, ue4, markers.name)
        Env.__init__(self)
        # rospy.init_node(f"gym-{self.task_name}")

        self.observation_space = Stack(observation_type, observation_stack, pre_aug)
        self.action_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.current_step = 0

        self.original_markers_len = markers.num
        self.markers_len =  markers.num
        self.past_markers_len =  markers.num
        self.markers_backup = self.set_markers(markers.name, markers.num)
        self.markers_need_to_visit = copy.deepcopy(self.markers_backup)
        self.range_to_get_target = markers.range_to_get
        self.target_range = target_range
        self.domain = domain
        self.heliport = True if domain.endswith('air') else False

        self.__pre_aug = pre_aug

    def _normalize_action(self, action):

        def _normalize_value(x, min_val, max_val, a, b):
            return ((x - min_val) / (max_val - min_val)) * (b - a) + a
        
        x, y, z, yaw_angle, gimbal_pitch_angle = action
        xmin, xmax = self.action_range.x
        ymin, ymax = self.action_range.y
        zmin, zmax = self.action_range.z
        yaw_min, yaw_max =self.action_range.yaw
        gimbal_pitch_min, gimbal_pitch_max = self.action_range.gimbal_pitch
        
        px = _normalize_value(x, -1, 1, xmin, xmax)
        py = _normalize_value(y, -1, 1, ymin, ymax)
        pz = _normalize_value(-1 * z, -1, 1, zmin, zmax)
        yaw = _normalize_value(yaw_angle, -1, 1, yaw_min, yaw_max)
        gimbal_pitch = _normalize_value(gimbal_pitch_angle, -1, 1, gimbal_pitch_min, gimbal_pitch_max)
        
        return px, py, pz, yaw, gimbal_pitch
    
    def _get_obs(self):
        def _pre_aug_obs_shape(obs : NDArray, dim : tuple, type= 'image'):
            if type.endswith('depth'):
                img_ = obs.copy()
                nan_location = np.isnan(img_)
                img_[nan_location] = np.nanmax(img_)
                norm_image =  (img_)*255./5.
                norm_image[0,0] = 255.
                norm_image = norm_image.astype('uint8')
                norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2BGR)
                
                return cv2.resize(norm_image.copy(), dim, interpolation = cv2.INTER_AREA).transpose(2, 0, 1) 
            
            if type.endswith('point_cloud'):
                centroid = np.mean(obs, axis=0)
                xy_projection = obs[:, :2]
                center_index = np.argmin(np.linalg.norm(xy_projection - centroid[:2], axis=1))
                half = 512
                obs = obs[center_index - half : center_index + half, :]
                print(obs.shape, center_index - half, center_index + half)
                return obs

            return cv2.resize(obs.copy(), dim, interpolation = cv2.INTER_AREA).transpose(2, 0, 1)
        
        def _parse_obs(obs : dict):
            _obs = dict()
            for k, v in obs.items():
                if k.endswith('tf'):
                    _obs[k] = v

                elif k.endswith('point_cloud'):
                    # max_points = self.__pre_aug[0] * self.__pre_aug[1]
                    # _obs[k] = np.pad(v, ((0, max_points - v.shape[0]), (0, 0)), mode='constant', constant_values=0)
                    _obs[k] = _pre_aug_obs_shape(v, self.__pre_aug, 'point_cloud')

                elif k.endswith('depth'):
                    _obs[k] = _pre_aug_obs_shape(v, self.__pre_aug, 'depth')

                else:
                   _obs[k] = _pre_aug_obs_shape(v, self.__pre_aug)
            
            return _obs


        observation = _parse_obs(self.client.observation)
        self.observation_space.stack = observation
        return self.observation_space.stack
    
    def _get_info(self):
        info = f"Current markers lenght : {self.markers_len}"
        return {'info' : info}
    
    def _get_state(self):
        
        viewd_markers, distances = self.client.detections()
        # print(f'vm {viewd_markers}')
        d = self.client.detection_distance()
        done = False
        reset_pose = False
        distance = 0
        self.markers_len = self.past_markers_len
       
        if distances:
            self.markers_need_to_visit -= set(viewd_markers)
            self.markers_len = len(self.markers_need_to_visit)
            dmin, dmax = self.range_to_get_target
            
            if (abs(d - min(distances))) < dmin or abs(max(distances) - d) > dmax:
                distance = 1
                reset_pose = True
                
            if not self.markers_len or (self.original_markers_len - self.markers_len) >= .97*self.original_markers_len :
                done = True
                
        else:
            reset_pose = True
        
        return self._get_obs(), self.markers_len, distance, reset_pose, done, self._get_info()
    
    def _reward(self, markers_len, distance, reset_pose, done):
        # print(f"original : {self.original_markers_len} {markers_len}")
        # print(f"CURRENT_STEP : {self.current_step} --- MAX_STEPS : {self.max_episode_steps}")
        alpha = 1e-2
        beta = 1e-3 * self.original_markers_len
        gamma = 2 * beta
        delta = 1e-1


        if done:
            return delta * self.original_markers_len, done
        
        if self.current_step == self.max_episode_steps -1 :
            return 0, True
        
        if reset_pose:
            self.client.go_home()
            if distance:
                # print(f'medium : {-10 * alpha}')
                return -alpha, done
            # print(f'minimum : {-20 * alpha}')
            return -beta, done
        
        if markers_len == self.past_markers_len:
            return 0, done
        
        self.past_markers_len = markers_len
        # print(f"bigger : {(self.original_markers_len - markers_len) * beta}")
        
        return alpha * (self.original_markers_len - markers_len), done

    def reset_random(self):
        pose = self.client.random_pose(self.action_range.x, self.action_range.y, 
                                       self.target_range.x, self.target_range.y, 
                                       self.centroide, self.heliport)
        self.client.home = pose
        self.client.go_home()

    def reset(self, seed= None, options= None):
        seed = np.random.seed()
        super().reset(seed=seed)
        self.current_step = 0
        
        self.client.go_home()
        self.markers_need_to_visit = copy.deepcopy(self.markers_backup)
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        norm_action = self._normalize_action(action)
        self.client.next_point_of_view(norm_action)
        observation, len_markers, distance, reset_pose, done, info = self._get_state()
        reward, done = self._reward(len_markers, distance, reset_pose, done)
        self.current_step += 1
        return observation, reward, done, info
    
    def close(self):
        os.killpg(self.airsim, signal.SIGINT)
        sys.exit()  



class BasicAirPOV(AirPointOfView, GymPointOfView):
    markers = AirPointOfView.markers.update({
                "num" : 79,
                "range_to_get" : [2, 80]
            })
    target_range = DictToClass({
                "x" : [2, 80],
                "y" : [2, 80]
            })

    def __init__(self, observation_type : str, 
                 observation_stack : int, 
                 pre_aug : tuple, 
                 max_ep_steps : int,
                 ue4 : str) -> None:
        
        AirPointOfView.__init__(self)
        GymPointOfView.__init__(self, observation_type, observation_stack, pre_aug, 
                                self.domain, ue4, self.markers, self.target_range)
                
        self.max_episode_steps = max_ep_steps
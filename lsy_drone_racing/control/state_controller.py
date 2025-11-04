"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import random

from lsy_drone_racing.control import Controller
from lsy_drone_racing.utils import utils

if TYPE_CHECKING:
    from numpy.typing import NDArray


def length(v):
    return np.linalg.norm(v)

def normalize(v):
    lengthV = length(v)
    if lengthV != 0:
        return v/lengthV
    return v

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

def compute_evasion_angles(V1, V2, evas):
    original_dir = V2 - V1
    evas_dir_0 = V2 - evas[0]
    evas_dir_1 = V2 - evas[1]

    angle_0 = angle_between(original_dir, evas_dir_0)
    angle_1 = angle_between(original_dir, evas_dir_1)

    return abs(angle_0), abs(angle_1)

def sort_by_distance(points, reference_point):
    # Berechne die Distanzen zu jedem Punkt
    distances = [np.linalg.norm(p - reference_point) for p in points]
    
    # Kombiniere Punkte mit ihren Distanzen
    paired = list(zip(points, distances))
    
    # Sortiere nach Distanz
    paired_sorted = sorted(paired, key=lambda x: x[1])
    
    # Extrahiere die sortierten Punkte
    sorted_points = [p for p, d in paired_sorted]
    
    return sorted_points



class Pipe:
    def __init__(self, center_pos, direction, ri, ra, h):
        self.center_pos = np.array(center_pos, dtype=float)
        self.direction = np.array(direction,dtype=float)
        self.half_h = 0.5*h  
        self.ri = ri  
        self.ra = ra 

        self.axis_start = self.center_pos - self.direction * self.half_h
        self.axis_end = self.center_pos + self.direction * self.half_h

        self._compute_bounds()
        self.fix_evasion_pos = []
        if h < 1:
            fix_pos = self.center_pos + [0,0,self.ra] + self.direction*0.1 
            self.fix_evasion_pos = [fix_pos,fix_pos]


    def _compute_bounds(self):
        # Alle Punkte entlang der Achse plus Radius in alle Richtungen
        points = [
            self.axis_start + np.array([dx, dy, dz])
            for dx in [-self.ra, self.ra]
            for dy in [-self.ra, self.ra]
            for dz in [-self.ra, self.ra]
        ] + [
            self.axis_end + np.array([dx, dy, dz])
            for dx in [-self.ra, self.ra]
            for dy in [-self.ra, self.ra]
            for dz in [-self.ra, self.ra]
        ]
        points = np.array(points)
        self.bbox_min = np.min(points, axis=0)
        self.bbox_max = np.max(points, axis=0)


    def contains_point(self, point):
        """
        Checks if point is part of the pipe
        """
        v = point - self.axis_start
        proj_len = np.dot(v, self.direction)

        if proj_len < 0 or proj_len > 2 * self.half_h:
            return False 

        self.proj_point = self.axis_start + proj_len * self.direction
        radial_dist = np.linalg.norm(point - self.proj_point)

        return self.ri <= radial_dist <= self.ra


    def is_colliding(self, V1, V2):
        # Schneller Bounding-Box-Test
        seg_min = np.minimum(V1, V2)
        seg_max = np.maximum(V1, V2)
        if np.any(seg_max < self.bbox_min) or np.any(seg_min > self.bbox_max):
            return False  # garantiert keine Kollision

        # Detaillierte Punktprüfung entlang der Strecke
        seg_dir = V2 - V1
        seg_len = np.linalg.norm(seg_dir)
        if seg_len == 0:
            return self.contains_point(V1)

        steps = int(seg_len / 0.05) + 1  # alle 5 cm prüfen
        for i in range(steps + 1):
            alpha = i / steps
            point = V1 + alpha * seg_dir
            if self.contains_point(point):
                if self.fix_evasion_pos:
                    self.evasion_pos = self.fix_evasion_pos
                else:
                    self.evasion_pos = []
                    cross = normalize(np.cross(seg_dir,self.direction))
                    self.evasion_pos.append(self.proj_point + self.ra * cross)
                    self.evasion_pos.append(self.proj_point - self.ra * cross)
                
                return True

        return False

    def set_up_evasion(self, obstacles):
        total_vec = np.zeros(3)

        for other in obstacles:
            if other is not self:
                vec = self.center_pos - other.center_pos
                if np.linalg.norm(vec) > 0:
                    total_vec += vec / np.linalg.norm(vec)  # Einheitsvektor

        if np.linalg.norm(total_vec) == 0:
            # Fallback: beliebige Richtung, z. B. orthogonal zur Achse
            d = self.direction / np.linalg.norm(self.direction)
            total_vec = np.cross(d, [1, 0, 0])
            if np.linalg.norm(total_vec) == 0:
                total_vec = np.cross(d, [0, 1, 0])

        evas_dir = total_vec / np.linalg.norm(total_vec) * self.ra
        self.evas_dir = evas_dir
        
class Nav:
    def __init__(self, path, dx=0.1):
        self.original_path = np.array(path)
        self.dx = dx
        self.points = self.interpolate_path_with_spacing(self.original_path, dx)

    def interpolate_path_with_spacing(self, points, dx):
        new_path = [points[0]]  # Startpunkt bleibt erhalten

        for i in range(1, len(points)):
            p0 = points[i - 1]
            p1 = points[i]
            segment = p1 - p0
            length = np.linalg.norm(segment)

            if length == 0:
                continue  # überspringe doppelte Punkte

            direction = segment / length
            steps = int(np.floor(length / dx))

            for j in range(1, steps + 1):
                new_point = p0 + direction * (j * dx)
                new_path.append(new_point)

            new_path.append(p1)  # Endpunkt des Segments

        return np.array(new_path)

    def displace(self, obstacles, factor=1.2, window=1,iterations=1):
        displaced = self.points.copy()
        N = len(displaced)
        for _ in range(iterations):
            for obs in obstacles:
                for i in range(N):
                    if obs.contains_point(displaced[i]):
                        center = obs.proj_point
                        vec = obs.evas_dir
                        for j in range(i - window, i + window + 1):
                            if 0 <= j < N:
                                #vec = displaced[j] - center
                                displaced[j] = displaced[j] + vec * factor

        self.points = displaced


class Pathfinder:
    def __init__(self,obs):
        self.gate_pos_offset = 0.3
        self.gate_ri = 0.2
        self.gate_ra = 0.6
        self.gate_h = 0.2
        self.stab_ra = 0.3
        self.fly_offset = 0.15
        self.fly_speed = 1 #points per second

        self.start_pos = obs['pos']
        self.current_pos = self.start_pos
        self.fly_end = self.start_pos
        self.path_free_i = 0
        self.is_rrt = False
        self.last_t = -0.001




        #self.nav = Nav(self.path_eva)
        #self.nav.displace(self.obstacles)
    

    def update(self,obs):
        self.current_pos = obs['pos']
        self.set_obs(obs)
        self.check_path()
        self.interpolate_path()

    def set_obs(self,obs):
        self.obstacles = []
        self.current_pos = obs['pos']
        self.path_free = [self.start_pos]
        for gate_i,gate_pos in enumerate(obs['gates_pos']):
            gate_before, gate_after,gate_dir = self.get_gate_pos_and_dir(gate_pos,obs['gates_quat'][gate_i])
            self.path_free.append(gate_before)
            self.path_free.append(gate_pos)
            self.path_free.append(gate_after)
            self.obstacles.append(Pipe(gate_pos,gate_dir,self.gate_ri,self.gate_ra,self.gate_h))
            #self.obstacles.append(Pipe(gate_pos-[0,0,0.6],[0,0,1],0,self.stab_ra,1))

        for stab_pos in obs['obstacles_pos']:
            self.obstacles.append(Pipe(stab_pos,[0,0,1],0,self.stab_ra,4))

    def check_path(self):
        self.path_eva = [self.start_pos]
        i = 1
        while i+2 < len(self.path_free):
            pos_before_before = self.path_free[i-1]
            pos_before = self.path_free[i]
            self.add_evasion_pos(pos_before_before,pos_before)
            self.path_eva.append(pos_before)
            i += 1
            self.path_eva.append(self.path_free[i])
            i += 1
            self.path_eva.append(self.path_free[i])
            i += 1
        
    #     i = 2
    #     while i <  len(self.path_eva):
    #         Vb = self.path_eva[i-2]
    #         Va = self.path_eva[i]

    #         if length(Va-Vb) < self.fly_offset:
    #             self.path_eva.pop(i-1)
    #             self.path_eva.pop(i-2)
    #         i += 1

    # def check_path(self):
    #     self.path_eva = [self.start_pos]
    #     for i in range(1,len(self.path_free)):
    #         self.add_evasion_pos(self.path_free[i-1],self.path_free[i])
    #         self.path_eva.append(self.path_free[i])
        

    def add_evasion_pos(self,V1,V2):
        new_evasion_pos = []
        for obstacle in self.obstacles:
            if obstacle.is_colliding(V1,V2):
                #self.path_eva.append(obstacle.evasion_pos)
                evas_pos = obstacle.evasion_pos
                da0,da1 = compute_evasion_angles(V1,V2,evas_pos)
                i_c = 0
                if da0 < da1:
                    i_c = 0
                else:
                    i_c = 1

                for obst in self.obstacles:
                    if obst is not obstacle:
                        if obst.contains_point(evas_pos[i_c]):
                            i_c = 1-i_c
                            break
                new_evasion_pos.append(evas_pos[i_c])

        new_evasion_pos = sort_by_distance(new_evasion_pos,V1)

        self.path_eva += new_evasion_pos

        
        


    def get_gate_pos_and_dir(self,pos,quat):
        rot = R.from_quat(quat)
        gate_dir = rot.apply([-1,0,0])
        shift = gate_dir*self.gate_pos_offset
        return pos + shift, pos - shift,gate_dir

    def is_path_free(self,path):
        for i in range(1,len(path)):
            V1 = path[i-1]
            V2 = path[i]
            for obstacle in self.obstacles:
                if obstacle.is_colliding(V1,V2):
                    return False
        return True

    def interpolate_path(self):
        self.path = np.asarray(self.path_eva)  # shape (N, 3)
        N = len(self.path)
        t_values = np.linspace(0, N/self.fly_speed, N)  # shape (N,)

        # Interpolator über axis=0 → gibt direkt (3,) Vektor zurück
        self.spline = interp1d(t_values, self.path, axis=0)
        #self.spline = CubicSpline(t_values,self.path)

    def des_pos(self,t):
        return self.spline(t)
        

        

class StateController(Controller):
    """State controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        self._t_total = 30 

        self.pf = Pathfinder(obs)

        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        self.pf.update(obs)

        utils.draw_line(info['env'],np.asarray(self.pf.path_eva))
        des_pos = self.pf.des_pos(t)
        
        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0

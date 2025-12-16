"""This module implements a Pathfinding Algorithm.

It uses Pipe to describe Obstacles and Gates of the environment and find based on the 
trjacectory and Pipe collsion suitable evasion points
"""



# Future imports (always placed at the very top)
from __future__ import annotations

# Standard library imports
from typing import TYPE_CHECKING, List, Tuple, Dict, Any
from datetime import datetime

# Third-party imports
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

# Local application imports
from lsy_drone_racing.control import Controller
from lsy_drone_racing.utils import utils  # Uncomment when needed

# Type checking
if TYPE_CHECKING:
    from numpy.typing import NDArray

def length(v: np.ndarray) -> float:
    """Computes the Euclidean length of a vector.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        float: The norm (length) of the vector.
    """
    return np.linalg.norm(v)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector to unit length.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized vector. If the input has zero length, returns the original vector.
    """
    length_v = length(v)
    if length_v != 0:
        return v / length_v
    return v


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates the angle in degrees between two vectors.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.

    Returns:
        float: Angle between the vectors in degrees.
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)


def compute_evasion_angles(V1: np.ndarray, V2: np.ndarray, evas: List[np.ndarray]) -> Tuple[float, float]:
    """Computes the angular deviation between the original direction and two evasive directions.

    Args:
        V1 (np.ndarray): Starting point.
        V2 (np.ndarray): Target point.
        evas (List[np.ndarray]): List containing two evasive points.

    Returns:
        Tuple[float, float]: Absolute angles between original direction and each evasive direction.
    """
    original_dir = V2 - V1
    evas_dir_0 = V2 - evas[0]
    evas_dir_1 = V2 - evas[1]

    angle_0 = angle_between(original_dir, evas_dir_0)
    angle_1 = angle_between(original_dir, evas_dir_1)

    return abs(angle_0), abs(angle_1)


def sort_by_distance(points: List[np.ndarray], reference_point: np.ndarray):
    """
    Sorts a list of points by their distance to a reference point,
    returning triples (index, point, distance).
    """
    distances = [np.linalg.norm(p - reference_point) for p in points]
    # enumerate liefert (index, point)
    paired = [(i, p, d) for i, (p, d) in enumerate(zip(points, distances))]
    paired_sorted = sorted(paired, key=lambda x: x[2])  # sortiere nach distance
    return paired_sorted

def project_point_on_line(before_pos,after_pos,ref_pos):
    b = np.array(before_pos, dtype=float)
    a = np.array(after_pos, dtype=float)
    r = np.array(ref_pos, dtype=float)

    # Richtungsvektor der Linie
    ab = a - b
    if length(ab) < 1e-3:
        print("small direction")

    # Projektion: b + ((r-b)·ab / (ab·ab)) * ab
    t = np.dot(r - b, ab) / np.dot(ab, ab)
    projected = b + t * ab

    return projected


class DroneLogging:
    def __init__(self, record_name):
        self.relative_folder_path = "DroneLogging"
        self.record_name = record_name
        self.sep = ","
        self.rows = []

        self.header = [
            "time",
            "pos_x","pos_y","pos_z",
            "quat_w","quat_x","quat_y","quat_z",
            "vel_x","vel_y","vel_z",
            "ang_vel_x","ang_vel_y","ang_vel_z",
            "target_gate_idx",
            "gate0_pos_x","gate0_pos_y","gate0_pos_z",
            "gate1_pos_x","gate1_pos_y","gate1_pos_z",
            "gate2_pos_x","gate2_pos_y","gate2_pos_z",
            "gate3_pos_x","gate3_pos_y","gate3_pos_z",
            "gate0_quat_w","gate0_quat_x","gate0_quat_y","gate0_quat_z",
            "gate1_quat_w","gate1_quat_x","gate1_quat_y","gate1_quat_z",
            "gate2_quat_w","gate2_quat_x","gate2_quat_y","gate2_quat_z",
            "gate3_quat_w","gate3_quat_x","gate3_quat_y","gate3_quat_z",
            "gate0_visited","gate1_visited","gate2_visited","gate3_visited",
            "obs0_pos_x","obs0_pos_y","obs0_pos_z",
            "obs1_pos_x","obs1_pos_y","obs1_pos_z",
            "obs2_pos_x","obs2_pos_y","obs2_pos_z",
            "obs3_pos_x","obs3_pos_y","obs3_pos_z",
            "obs0_visited","obs1_visited","obs2_visited","obs3_visited",
            "output_pos_x","output_pos_y","output_pos_z"
        ]

    def save_file(self):
        now = datetime.now()
        formatted = now.strftime("%Y_%m_%d__%H_%M_%S")
        path = f"{self.relative_folder_path}/DroneLog__{self.record_name}__{formatted}.csv"

        text = ""

        for head in self.header:
            text += head + self.sep
        text += "\n"

        for row in self.rows:
            text += self.get_row_text(row) + "\n"

        file = open(path,"w")
        if file:
            file.write(text)
            file.close()
        else:
            print("Error: File not writeable:",path)

    def add_row(self,t,obs,next_pos):
        row = [np.float32(t)]
        for key in obs:
            row.extend(obs[key].flatten())
        
        row.extend(next_pos.flatten())
        self.rows.append(row)

    def get_row_text(self,row):
        out = ""
        for obj in row:
            if isinstance(obj, np.bool_):
                if obj:
                    out += "1"
                else:
                    out += "0"
            else:
                out += str(obj)
            out += self.sep
        return out


class History:
    def __init__(self,pos):
        self.obstacles_id = []
        self.wrongCourse = False
        self.onAlternitiveLength = 1
        self.posHistory = [pos]

    def update(self,newPos,obstacle_id):
        if obstacle_id not in self.obstacles_id:
            self.obstacles_id.append(obstacle_id)

        if len(self.obstacles_id) > self.onAlternitiveLength:
            self.wrongCourse = True
        self.posHistory.append(newPos)

    def getCurrentAlternitive(self):
        startPos = self.posHistory[0]
        lastPos  = self.posHistory[-1]
        self.onAlternitiveLength = length(self.obstacles_id)
        
        return startPos - 3.0 * (lastPos-startPos)


class Point:
    def __init__(self,pos):
        self.pos = pos
        self.history = History(pos)
    
    def update(self,newPos,obstacle_id):
        self.history.update(newPos,obstacle_id)
        self.pos = newPos
        return self.history.wrongCourse

    def getCurrentAlternitive(self):
        return self.history.getCurrentAlternitive()

class RRT:
    def __init__(self):
        self.points = []
    
    def update(self,path_free,obstacles,dx,points_per_distance):
        self.interpolate_path(path_free,points_per_distance)
        self.displace_colliding_points(obstacles,dx)
        
    def interpolate_path(self, path_free, points_per_distance):
        for i in range(len(path_free) - 1):
            p1 = np.array(path_free[i])
            p2 = np.array(path_free[i + 1])
            distance = length(p2 - p1)
            num_points = max(1, int(distance * points_per_distance))

            for j in range(num_points):
                t = j / num_points
                interpolated = (1 - t) * p1 + t * p2
                self.points.append(interpolated)

        # Optionally add the last point
        self.points.append(path_free[-1])

    def displace_colliding_points(self, obstacles, dx):
        """
        Verschiebt Punkte radial aus kollidierenden Pipes heraus.

        Args:
            obstacles (list): Liste von Pipe-Objekten.
            dx (float): Verschiebungsbetrag.
        """
        displaced_points = []

        for pt in self.points:
            p = np.array(pt.pos, dtype=float)
            moved = False

            for pipe_i,pipe in enumerate(obstacles):
                if pipe.contains_point(p):
                    # Projektion des Punktes auf die Pipe-Achse
                    v = p - pipe.center_pos
                    pipe_dir = pipe.direction
                    proj_length = np.dot(v, pipe_dir)
                    proj_point = pipe.center_pos + proj_length * pipe_dir

                    # Radialrichtung vom projizierten Punkt zum Punkt
                    radial = p - proj_point
                    norm = length(radial)

                    if norm > 1e-6:
                        radial_dir = radial / norm
                    else:
                        # Fallback: wähle eine beliebige senkrechte Richtung
                        arbitrary = np.array([1.0, 0.0, 0.0])
                        if np.allclose(pipe_dir, arbitrary):
                            arbitrary = np.array([0.0, 1.0, 0.0])
                        radial_dir = np.cross(pipe_dir, arbitrary)
                        radial_dir /= length(radial_dir)

                    newPos = p + dx * radial_dir
                    wrongCourse = pt.update(newPos,pipe_i)
                    break # so only one Collision at a time can happen
                        

    def get_points(self):
        out = []
        for point in self.points:
            out.append(point.pos)
        return out

class Pipe:
    """Represents a cylindrical pipe segment in 3D space, used for collision detection and evasion logic."""

    def __init__(
        self,
        center_pos: List[float],
        direction: List[float],
        ri: float,
        ra: float,
        h: float
    ) -> None:
        """Initializes a Pipe object.

        Args:
            center_pos (List[float]): Center position of the pipe.
            direction (List[float]): Direction vector of the pipe's axis.
            ri (float): Inner radius of the pipe.
            ra (float): Outer radius of the pipe.
            h (float): Height of the pipe.
        """
        self.center_pos = np.array(center_pos, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.half_h = 0.5 * h
        self.ri = ri
        self.ra = ra

        self.axis_start = self.center_pos - self.direction * self.half_h
        self.axis_end = self.center_pos + self.direction * self.half_h

        self._compute_bounds()
        self.fix_evasion_pos: List[np.ndarray] = []

        if h < 1:
            fix_pos = self.center_pos + np.array([0, 0, 1.1*self.ra]) - self.direction * 0.15
            self.fix_evasion_pos = [fix_pos, fix_pos]

    def _compute_bounds(self) -> None:
        """Computes the bounding box of the pipe based on its radius and axis."""
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

    def contains_point(self, point: np.ndarray) -> bool:
        """Checks whether a given point lies within the pipe's volume.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point is inside the pipe, False otherwise.
        """
        v = point - self.axis_start
        proj_len = np.dot(v, self.direction)

        if proj_len < 0 or proj_len > 2 * self.half_h:
            return False

        self.proj_point = self.axis_start + proj_len * self.direction
        radial_dist = np.linalg.norm(point - self.proj_point)

        return self.ri <= radial_dist <= self.ra

    def is_colliding(self, V1: np.ndarray, V2: np.ndarray) -> bool:
        """Checks whether a line segment between V1 and V2 collides with the pipe.

        Args:
            V1 (np.ndarray): Start point of the segment.
            V2 (np.ndarray): End point of the segment.

        Returns:
            bool: True if the segment collides with the pipe, False otherwise.
        """
        seg_min = np.minimum(V1, V2)
        seg_max = np.maximum(V1, V2)

        if np.any(seg_max < self.bbox_min) or np.any(seg_min > self.bbox_max):
            return False

        seg_dir = V2 - V1
        seg_len = np.linalg.norm(seg_dir)

        if seg_len == 0:
            return self.contains_point(V1)

        steps = int(seg_len / 0.05) + 1
        for i in range(steps + 1):
            alpha = i / steps
            point = V1 + alpha * seg_dir
            if self.contains_point(point):
                if self.fix_evasion_pos:
                    self.evasion_pos = self.fix_evasion_pos
                else:
                    self.evasion_pos = []
                    cross = normalize(np.cross(seg_dir, self.direction))
                    self.evasion_pos.append(self.proj_point + self.ra * cross)
                    self.evasion_pos.append(self.proj_point - self.ra * cross)
                return True

        return False

    def set_up_evasion(self, obstacles: List["Pipe"]) -> None:
        """Computes an evasion direction based on surrounding obstacles.

        Args:
            obstacles (List[Pipe]): List of other pipe objects to avoid.
        """
        total_vec = np.zeros(3)

        for other in obstacles:
            if other is not self:
                vec = self.center_pos - other.center_pos
                if np.linalg.norm(vec) > 0:
                    total_vec += vec / np.linalg.norm(vec)

        if np.linalg.norm(total_vec) == 0:
            d = self.direction / np.linalg.norm(self.direction)
            total_vec = np.cross(d, [1, 0, 0])
            if np.linalg.norm(total_vec) == 0:
                total_vec = np.cross(d, [0, 1, 0])

        evas_dir = total_vec / np.linalg.norm(total_vec) * self.ra
        self.evas_dir = evas_dir

    def combine_obstacles(obstacles: List["Pipe"],new_radius_increase: float) -> Pipe:
        new_center_pos = np.zeros(3)
        new_radius = 0
        for obstacle in obstacles:
            new_center_pos += obstacle.pos
        new_center_pos = new_center_pos / len(obstacles)
        
        for obstacle in obstacles:
            dist = length(obstacle.pos - new_center_pos)
            if dist > new_radius:
                new_radius = dist

        new_radius += new_radius_increase

        return Pipe(new_center_pos,np.array([0,0,1.0]),0,new_radius,5)


class Pathfinder:
    """Computes and manages a navigable path through gates and obstacles using evasion logic."""

    def __init__(self, obs: Dict[str, Any]) -> None:
        """Initializes the Pathfinder with environment data.

        Args:
            obs (Dict[str, Any]): Observation dictionary containing position, gates, and obstacles.
        """
        self.gate_pos_offset = 0.5
        self.update_path_normal_distance = 0.1
        self.gate_ri = 0.1
        self.gate_ra = 0.6
        self.gate_h = 0.2
        self.stab_ra = 0.2
        self.fly_offset = 0.15
        self.fly_speed = 3  # points per second
        self.cpffeb = 0.0 #0.1 # "correct path free for entry behaviour": this the geometric shift of entry, middle and exit position based on the place, where the drone is coming from
        self.obs_comb_dist = 0.4
        self.max_gate_angle = 50

        self.start_pos = obs['pos']
        self.current_pos = self.start_pos
        self.path_free = [np.array([100,100,100])]
        self.fly_end = self.start_pos
        self.path_eva_i = 1
        self.path_free_target_i = 1
        self.t_offset = 0.0
        self.new_path = True

    def update(self, obs: Dict[str, Any],t:float) -> None:
        """Updates the current position and recalculates the path.

        Args:
            obs (Dict[str, Any]): Updated observation data.
        """
        self.current_pos = obs['pos']
        self.set_obs(obs)

        # # check if the next target pos is reached
        # if length(self.current_pos - self.path_free[self.path_free_target_i]) < self.fly_offset:
        #     self.path_free_target_i += 1
        #     self.new_path = True
        #     if self.path_free_target_i >= len(self.path_free):
        #         self.path_free_target_i = len(self.path_free) - 1

        if self.new_path:
            print("new path")
            self.check_path()
            self.interpolate_path(t)

    def set_obs(self, obs: Dict[str, Any]) -> None:
        """Sets up obstacles and gates based on the observation data.

        Args:
            obs (Dict[str, Any]): Observation data containing gates and obstacles.
        """
        last_path_free = self.path_free.copy()
        self.obstacles: List[Pipe] = []
        self.current_pos = obs['pos']
        self.path_free = [self.start_pos]
        self.gate_dir = []

        gate_after = None
        for gate_i, gate_pos in enumerate(obs['gates_pos']):
            gate_before, gate_after, gate_dir = self.get_gate_pos_and_dir(gate_pos, obs['gates_quat'][gate_i])
            self.path_free.extend([gate_pos])
            self.gate_dir.append(gate_dir)
            self.obstacles.append(Pipe(gate_pos, gate_dir, self.gate_ri, self.gate_ra, self.gate_h))
        
        self.path_free.append(gate_after)

        for stab_pos in obs['obstacles_pos']:
            self.obstacles.append(Pipe(stab_pos, [0, 0, 1], 0, self.stab_ra, 4))

        #check if the path has changed (normalized position offset)
        self.new_path = False
        for pos_i in range(len(self.path_free)):
            if length(last_path_free[pos_i] - self.path_free[pos_i]) > self.update_path_normal_distance:
                self.new_path = True
                break
        

        # obs_comb = [None,None,None,None]
        # for obs_i in range(4):
        #     for obs_e in range(obs_i+1,4):
        #         if obs_i != obs_e:
        #             obstacle_1 = self.obstacles[obs_i]
        #             obstacle_2 = self.obstacles[obs_e]
        #             pos2D_1 = obstacle_1.pos[:2]
        #             pos2D_2 = obstacle_2.pos[:2]
        #             if length(pos2D_1 - pos2D_2) < self.obs_comb_dist:
        #                 if obs_comb[obs_i] == None:
        #                     obs_comb[obs_i] = [self.obstacles[obs_i],self.obstacles[obs_e]]
        #                 else:
        #                     obs_comb[obs_i].append(self.obstacles[obs_e])

        # for obs_i in range(4):
        #     if obs_comb[obs_i] != None:
        #         obs_comb[obs_i] = Pipe.combine_obstacles(obs_comb[obs_i])-

    def check_path(self) -> None:
        """Constructs the path with evasion points between gates and obstacles."""
        self.path_generation = [self.start_pos]

        for i in range(1,len(self.path_free)-1):
            gate_pos = self.path_free[i]
            gate_dir = self.gate_dir[i-1]
            point_before = self.path_generation[-1]
            point_after = self.path_free[i+1]

            add_angel_pos, angle_pos = self.add_better_gate_angle_pos(gate_pos,gate_dir,point_before)
            if add_angel_pos:
                self.path_generation.append(angle_pos)
            self.path_generation.append(gate_pos)
            
            add_angel_pos, angle_pos = self.add_better_gate_angle_pos(gate_pos,gate_dir,point_after)
            if add_angel_pos:
                self.path_generation.append(angle_pos)

        self.path_eva = [self.path_generation[0]]
        for pos_i in range(1,len(self.path_generation)):
            self.add_evasion_pos(self.path_generation[pos_i-1],self.path_generation[pos_i])
            self.path_eva.append(self.path_generation[pos_i])

        self.path_eva.append(self.path_free[-1])



    def add_better_gate_angle_pos(self,gate_pos,gate_dir,point):
        alpha = angle_between(gate_dir,point - gate_pos)
        print(gate_pos,alpha)
        abs_alpha = abs(alpha)
        if abs_alpha > self.max_gate_angle:
            #r_vec = R.from_rotvec(-abs_alpha/alpha*(abs_alpha - self.max_gate_angle) * np.cross(gate_dir,point - gate_pos))
            if abs_alpha <= 90:
                return True, gate_pos + self.gate_pos_offset * gate_dir
            elif abs_alpha < 130:
                return True, gate_pos - self.gate_pos_offset * gate_dir
        return False, None

    def add_evasion_pos(self, V1: np.ndarray, V2: np.ndarray) -> None:
        """Adds evasion points between two path segments if a collision is detected.

        Args:
            V1 (np.ndarray): Start point of the segment.
            V2 (np.ndarray): End point of the segment.
        """
        new_evasion_pos: List[np.ndarray] = []

        for obstacle in self.obstacles:
            if obstacle.is_colliding(V1, V2):
                evas_pos = obstacle.evasion_pos
                da0, da1 = compute_evasion_angles(V1, V2, evas_pos)
                i_c = 0 if da0 < da1 else 1

                for obst in self.obstacles:
                    if obst is not obstacle and obst.contains_point(evas_pos[i_c]):
                        i_c = 1 - i_c
                        break

                new_evasion_pos.append(evas_pos[i_c])

        new_evasion_pos_set = sort_by_distance(new_evasion_pos, V1)
        new_evasion_pos = []
        for i,p,d in new_evasion_pos_set:
            new_evasion_pos.append(p)
        self.path_eva += new_evasion_pos

    def get_gate_pos_and_dir(self, pos: np.ndarray, quat: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes gate direction and offset positions based on quaternion orientation.

        Args:
            pos (np.ndarray): Gate position.
            quat (List[float]): Quaternion representing gate orientation.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Offset positions before and after the gate, and direction vector.
        """
        rot = R.from_quat(quat)
        gate_dir = rot.apply([1, 0, 0])
        shift = gate_dir * self.gate_pos_offset
        return pos - shift, pos + shift, gate_dir

    def is_path_free(self, path: List[np.ndarray]) -> bool:
        """Checks whether a given path is free of collisions.

        Args:
            path (List[np.ndarray]): List of path points.

        Returns:
            bool: True if the path is collision-free, False otherwise.
        """
        for i in range(1, len(path)):
            V1 = path[i - 1]
            V2 = path[i]
            for obstacle in self.obstacles:
                if obstacle.is_colliding(V1, V2):
                    return False
        return True

    def interpolate_path(self, t:float) -> None:
        """Interpolates the path using linear interpolation for smooth navigation."""
        self.t_offset = 0.0
        self.path = np.asarray(self.path_eva)
        self.rrt = RRT()
        self.rrt.interpolate_path(self.path,5)
        self.path = np.asarray(self.rrt.points)
        self.N = len(self.path)
        t_total = self.N / self.fly_speed
        t_values = np.linspace(0,t_total , self.N)
        # self.pos_spline = interp1d(t_values, self.path, axis = 0)
        self.pos_spline = CubicSpline(t_values,self.path)
        self.vel_spline = self.pos_spline.derivative()

    def des_pos(self, t: float) -> np.ndarray:
        """Returns the desired position at time t along the interpolated path.

        Args:
            t (float): Time value.

        Returns:
            np.ndarray: Interpolated position at time t.
        """
        if t >= self.N / self.fly_speed:
            return self.current_pos
        return self.spline(t) - [0,0,0.05]





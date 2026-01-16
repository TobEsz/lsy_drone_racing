"""This module implements a Pathfinding Algorithm.

It uses Pipe to describe Obstacles and Gates of the environment and find based on the 
trjacectory and Pipe collsion suitable evasion points
"""



# Future imports (always placed at the very top)
from __future__ import annotations

# Standard library imports
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
from datetime import datetime

# Third-party imports
import numpy as np
from scipy.interpolate import interp1d
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
    dist_sorted = sorted(paired, key=lambda x: x[2])  # sortiere nach distance
    return dist_sorted

def project_point_on_line(
    before_pos: np.ndarray,
    after_pos: np.ndarray,
    ref_pos: np.ndarray,
) -> np.ndarray:
    """
    Project a reference point onto the line defined by two positions.

    Args:
        before_pos (np.ndarray): First point defining the line.
        after_pos (np.ndarray): Second point defining the line.
        ref_pos (np.ndarray): Point to project onto the line.

    Returns:
        np.ndarray: The projected point on the line.
    """
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

def rotated_offset_choice(center:np.ndarray, off:np.ndarray, away:np.ndarray,dist:float=0.05) -> np.ndarray:
    """
    This function moves the point off slightly to left or right to move away from position away

    Args:
        center (np.ndarray): Gate center position
        off (np.ndarray): Either gate entry or exit
        away (np.ndarray): the point that is next or before off on the trajectory

    Returns:
        np.ndarray: moved off
    """
    v = off - center
    n = length(v)
    if n == 0:
        return off
    v = normalize(v)

    # Rotation around Z-Axe of +90°
    # (x, y, z) → (-y, x, z)
    v_rot = np.array([-v[1], v[0], v[2]])

    p_plus  = off + dist * v_rot
    p_minus = off - dist * v_rot

    d_plus  = np.linalg.norm(away - p_plus)
    d_minus = np.linalg.norm(away - p_minus)

    return p_plus if d_minus < d_plus else p_minus

#region Pipe
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
        self.evasion_radius_factor = 2

        self.axis_start = self.center_pos - self.direction * self.half_h
        self.axis_end = self.center_pos + self.direction * self.half_h

        self._compute_bounds()
        self.fix_evasion_pos: List[np.ndarray] = []

        if h < 1:
            fix_pos = self.center_pos + np.array([0, 0, 1.5*self.ra]) - self.direction * 0.15
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
            print("WarnWarnWarn No evasion point is calculated") # this is super rare to happen, on occurence it did not lead to failure
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
                    self.evasion_pos.append(self.proj_point + self.evasion_radius_factor * self.ra * cross)
                    self.evasion_pos.append(self.proj_point - self.evasion_radius_factor * self.ra * cross)
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

    def get_safe_evasion_point(
        self,
        V1: np.ndarray,
        V2: np.ndarray,
        obstacles: List["Pipe"],
    ) -> Optional[np.ndarray]:
        """
        Select a valid evasion point that is not inside any obstacle.

        Args:
            V1 (np.ndarray): Start point of the segment being evaluated.
            V2 (np.ndarray): End point of the segment being evaluated.
            obstacles (List[Pipe]): All obstacles to test against.

        Returns:
            Optional[np.ndarray]: A safe evasion point if one exists, otherwise None.
        """
        evas_pos = self.evasion_pos
        in_obstacle = []
        for eva_i, eva_pos in enumerate(evas_pos):
            for obst in obstacles:
                if obst is not self and obst.contains_point(eva_pos):
                    in_obstacle.append(eva_i)
                    break

        if len(evas_pos) - len(in_obstacle) > 0:
            for eva_i in in_obstacle:
                evas_pos.pop(eva_i)
            if len(evas_pos) == 2:
                da0, da1 = compute_evasion_angles(V1, V2, evas_pos)
                i_c = 0 if da0 < da1 else 1
                new_evasion_pos = evas_pos[i_c]
            else:
                new_evasion_pos = evas_pos[0]
            return new_evasion_pos

        return None


    @staticmethod
    def combine_obstacles(
        obstacles: List["Pipe"],
        new_radius_increase: float,
    ) -> "Pipe":
        """
        Combine multiple obstacles into a single enlarged pipe.

        The new pipe is centered at the average position of all obstacles,
        and its radius is expanded to cover all of them plus an additional margin.

        Args:
            obstacles (List[Pipe]): Obstacles to merge.
            new_radius_increase (float): Additional radius added to the combined obstacle.

        Returns:
            Pipe: A new pipe representing the merged obstacle region.
        """
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

        return Pipe(new_center_pos, np.array([0, 0, 1.0]), 0, new_radius, 5)


#region Points
class Points:
    """
    Container for storing and manipulating ordered 3D positions.
    """

    def __init__(self) -> None:
        """Initialize an empty list of positions."""
        self.pos = []

    def update(self, new_pos) -> None:
        """
        Replace the entire list of stored positions.

        Args:
            new_pos: New list of positions.
        """
        self.pos = new_pos

    def append(self, new_pos) -> None:
        """
        Append a new position to the list.

        Args:
            new_pos: Position to append.
        """
        self.pos.append(new_pos)

    def get_last(self):
        """
        Get the last stored position.

        Returns:
            The last position, or None if empty.
        """
        if len(self.pos):
            return self.pos[-1]
        return None

    def get_first(self):
        """
        Get the first stored position.

        Returns:
            The first position, or None if empty.
        """
        if len(self.pos):
            return self.pos[0]
        return None

    def move_away_last(self, to_pos) -> None:
        """
        Move the last point slightly away from a reference position.

        Args:
            to_pos: Reference position to move away from.
        """
        self.pos[-1] = -0.05 * normalize(to_pos - self.pos[-1]) + self.pos[-1]

    def move_away_first(self, to_pos) -> None:
        """
        Move the first point slightly away from a reference position.

        Args:
            to_pos: Reference position to move away from.
        """
        self.pos[0] = -0.05 * normalize(to_pos - self.pos[0]) + self.pos[0]

    def set_last(self, new_pos) -> None:
        """
        Overwrite the last stored position.

        Args:
            new_pos: New value for the last position.
        """
        self.pos[-1] = new_pos

    def set_first(self, new_pos) -> None:
        """
        Overwrite the first stored position.

        Args:
            new_pos: New value for the first position.
        """
        self.pos[0] = new_pos


class EvasionPoints(Points):
    """
    Specialized Points container that tracks obstacle-related evasion positions.
    """

    def __init__(self) -> None:
        """Initialize evasion points and obstacle tracking."""
        super().__init__()
        self.obstacles_pos = []
        self.obstacles_id = []
        self.max_changeable_dist = 0.3

    def update(self, current_drone_position:np.ndarray, before:np.ndarray, after:np.ndarray, new_pos:np.ndarray) -> None:
        """
        Update evasion points while preserving stable positions when possible.

        Args:
            current_drone_position (np.ndarray): Current drone position (unused).
            before(np.ndarray): Previous reference point (unused).
            after(np.ndarray): Next reference point (unused).
            new_pos(np.ndarray): Newly computed evasion positions.
        """
        old_len = len(self.pos)
        new_len = len(new_pos)
        # super().update(new_pos)
        if old_len == 0:
            super().update(new_pos)
        else:
            if old_len == new_len:
                for i in range(old_len):
                    if length(self.pos[i] - new_pos[i]) < self.max_changeable_dist:
                        self.pos[i] = new_pos[i]
            elif old_len < new_len:
                previous_eva_pos_close_enough = True
                for pos in self.pos:
                    sorted_by_pos = sort_by_distance(new_pos, pos)[0]
                    if sorted_by_pos[2] > self.max_changeable_dist:
                        previous_eva_pos_close_enough = False
                        break
                if previous_eva_pos_close_enough:
                    super().update(new_pos)

    def get_obstacles_id(self):
        """
        Get the IDs of obstacles associated with these evasion points.

        Returns:
            List of obstacle IDs.
        """
        return self.obstacles_id

    def check(self, new_pos, obstacles_pos, obstacles_id) -> None:
        """
        Update evasion points if obstacle IDs match and movement is small.

        Args:
            new_pos: New candidate evasion positions.
            obstacles_pos: Current obstacle positions.
            obstacles_id: Current obstacle IDs.
        """
        if len(self.obstacles_pos) == len(obstacles_pos):
            for i in range(len(self.obstacles_pos)):
                if obstacles_id[i] == self.obstacles_id[i]:
                    if length(new_pos[i] - self.pos[i]) < 0.2:
                        self.pos[i] = new_pos[i]


class Path:
    """
    Manage the full flight path including start, gate, and evasion points.
    """

    def __init__(self) -> None:
        """Initialize path containers."""
        self.start_pos = Points()
        self.points = [Points(), Points(), Points(), Points()]
        self.evasion_points = [
            EvasionPoints(),
            EvasionPoints(),
            EvasionPoints(),
            EvasionPoints(),
        ]

    def update(self, new_pos, gate_index) -> None:
        """
        Replace the stored points for a specific gate.

        Args:
            new_pos: New positions for the gate.
            gate_index: Index of the gate to update.
        """
        self.points[gate_index].update(new_pos)

    def append(self, new_pos, gate_index) -> None:
        """
        Append a new point to a specific gate.

        Args:
            new_pos: Position to append.
            gate_index: Gate index to modify.
        """
        self.points[gate_index].append(new_pos)

    def get_pos(self, gate_index):
        """
        Get stored positions for a specific gate.

        Args:
            gate_index: Index of the gate.

        Returns:
            List of positions for the gate.
        """
        return self.points[gate_index].pos

    def _add_to_new_path(self, positions) -> None:
        """
        Add positions to the final path, inserting intermediate points when needed.

        Args:
            positions: Positions to add.
        """
        for pos in positions:
            last_pos = self.new_path[-1]
            new_pos = pos - np.array([0, 0, 0.08])
            distance = length(new_pos - last_pos)
            if distance > 1:
                direction = new_pos - last_pos
                self.new_path.append(0.1 * direction + last_pos)
                self.new_path.append(0.9 * direction + last_pos)
            if distance > 0.0001:
                self.new_path.append(new_pos)

    def get_path(self):
        """
        Build and return the full smoothed path.

        Returns:
            List of positions forming the full path.
        """
        self.new_path = []
        self.new_path.extend(self.start_pos.pos)
        for i in range(4):
            self._add_to_new_path(self.evasion_points[i].pos)
            self._add_to_new_path(self.points[i].pos)
        return self.new_path

    def check_evasion_points(self, obstacles) -> None:
        """
        Update evasion points based on obstacle movement.

        Args:
            obstacles: List of obstacle objects.
        """
        for evasion_points in self.evasion_points:
            new_obstacles_pos = []
            for i in evasion_points.get_obstacles_id():
                new_obstacles_pos.append(obstacles[i].center_pos)
            evasion_points.check(new_obstacles_pos)

    def adjust_gate_entry_exit(self) -> None:
        """
        Adjust entry and exit points around gates to avoid sharp transitions.
        """
        before = self.start_pos.get_last()
        for i in range(3):
            after = self.points[i + 1].get_first()
            self.points[i].move_away_first(before)
            self.points[i].move_away_last(after)
            before = self.points[i].get_last()            


# region Pathfinder
class Pathfinder:
    """Computes and manages a navigable path through gates and obstacles using evasion logic."""

    def __init__(self, obs: Dict[str, Any], trajectory_time: float) -> None:
        """Initializes the Pathfinder with environment data.

        Args:
            obs (Dict[str, Any]): Observation dictionary containing position, gates, and obstacles.
        """

        # Trajectory building parameter based on geometry of obstacles, gates
        self.gate_pos_offset = 0.3 # 0.3 is good
        self.update_path_normal_distance = 0.01
        self.radius_evasion_factor = 2
        self.gate_ri = 0.1
        self.gate_ra = 0.6
        self.gate_h = 0.3
        self.stab_ra = 0.2 #0.2 is good
        self.fly_offset = 0.15
        self.fly_speed = 2  # points per second
        self.cpffeb = 0.0 #0.1 # "correct path free for entry behaviour": this the geometric shift of entry, middle and exit position based on the place, where the drone is coming from
        self.obs_comb_dist = 0.4
        self.max_gate_angle = 50
        self.gate_offset_adjust = 0.5 # < 1. When the position of the gate_before or gate_after is inside a collision then self.gate_pos_offset will be multiplied with this until it fits

        # variables for obs and trajectory
        self.start_pos = obs['pos']
        self.current_pos = self.start_pos
        self.path_free = [np.array([100,100,100])]
        self.obstacles = [Pipe(np.array([100,100,100]),np.array([0,0,1]),0,0,0)]
        self.fly_end = self.start_pos
        self.trajectory_time = trajectory_time
        self.path_eva_i = 1
        self.path_free_target_i = 1
        self.t_offset = 0.0
        self.new_path = True
        self.path_initial_checked = False
        self.path = Path() # this stores now the different parts of the trajectory

    def update(self, obs: Dict[str, Any],t:float) -> None:
        """Updates the current position and recalculates the path.

        Args:
            obs (Dict[str, Any]): Updated observation data.
        """
        self.current_pos = obs['pos']
        self.set_obs(obs)

        if self.new_path:
            self.check_path()
            self.interpolate_path(t)
            self.path_initial_checked = True

    def is_point_safe(self, point) -> bool:
        """
        Check whether a point lies outside all obstacles.

        Args:
            point: The point to evaluate.

        Returns:
            bool: False if the point is inside any obstacle, otherwise True.
        """
        for obstacle in self.obstacles:
            if obstacle.contains_point(point):
                return False
        return True


    def adjust_gate_offset(self) -> None:
        """
        Adjust entry and exit offsets for each gate to ensure the path does not
        intersect obstacles. The method iteratively pulls the before/after points
        toward the gate center until they are safe.

        Updates:
            - Path segments for each gate.
            - Extends the final segment beyond the last gate.
        """
        gate_pos = None
        gate_after = None

        away = self.start_pos
        dist = [0.05,0.03,0.05,0.05] # slightly modifid to improve lvl 2

        for gate_i, gate_pos in enumerate(self.gate_pos):
            path_free = []
            gate_before = self.gate_before[gate_i]
            gate_after = self.gate_after[gate_i]

            gate_before = rotated_offset_choice(gate_pos,gate_before,away,dist[gate_i])

            # bring gate before closer to gate center if a obstacle is close
            while not self.is_point_safe(gate_before) and length(gate_before - gate_pos) > 0.1:
                gate_before = self.gate_offset_adjust * (gate_before - gate_pos) + gate_pos

            if gate_i < 3:
                away = self.gate_before[gate_i+1]
                gate_after = rotated_offset_choice(gate_pos,gate_after,away,dist[gate_i])

            # bring gate after closer to gate center if a obstacle is close
            while not self.is_point_safe(gate_after) and length(gate_after - gate_pos) > 0.1:
                gate_after = self.gate_offset_adjust * (gate_after - gate_pos) + gate_pos

            away = gate_after

            path_free.append(gate_before)
            path_free.append(gate_after)

            self.path.update(path_free, gate_i)

        # Extend the last point
        # self.path.append(gate_pos + 2 * (gate_after - gate_pos), 3)
        # self.path.adjust_gate_entry_exit()

    def set_obs(self, obs: Dict[str, Any]) -> None:
        """Sets up obstacles and gates based on the observation data.
 
        Args:
            obs (Dict[str, Any]): Observation data containing gates and obstacles.
        """
        last_obstacles = self.obstacles.copy()
        self.obstacles: List[Pipe] = []
        ra = [0.05,0.05,0,-0.05] # this improves level 2 slightly

        # Setup Start Condition 
        self.current_pos = obs['pos']
        path_free = [self.start_pos]
        path_free.append(self.start_pos + [0,0,0.2]) # This seems necessary for current Attitute_RL
        self.path.start_pos.update(path_free)

        self.gate_after = []
        self.gate_pos = []
        self.gate_before = []

        gate_after = None
        gate_pos = None

        # Setup the Gate Positions and add Gate to obstacles
        for gate_i, gate_pos in enumerate(obs['gates_pos']):
            gate_before, gate_after, gate_dir = self.get_gate_pos_and_dir(gate_pos, obs['gates_quat'][gate_i])
            self.gate_after.append(gate_after)
            self.gate_pos.append(gate_pos)
            self.gate_before.append(gate_before)
            self.obstacles.append(Pipe(gate_pos, gate_dir, self.gate_ri, self.gate_ra, self.gate_h))
        # Setup the Stab Obstacle
        for stab_i, stab_pos in enumerate(obs['obstacles_pos']):
            self.obstacles.append(Pipe(stab_pos, [0, 0, 1], 0, self.stab_ra+ra[stab_i], 4))
        
        self.adjust_gate_offset()
        
        # Check if reversing from gate is possible
        for i in range(len(self.gate_after)-1): #last obstacle should not have an obstacle behind the gate
            if abs(angle_between(self.gate_pos[i]-self.gate_after[i],self.gate_before[i+1]-self.gate_after[i])) < 60:
                path_kick_back = self.path.get_pos(i)
                path_kick_back.append(path_kick_back[1]-np.array([0,0,0.001])) # this is the gate exit 
                path_kick_back.append(path_kick_back[0])
                self.path.update(path_kick_back,i)

        # check if obstacles has changed
        self.new_path = False
        for obs_i in range(len(last_obstacles)):
            if length(last_obstacles[obs_i].center_pos - self.obstacles[obs_i].center_pos) > self.update_path_normal_distance:
                self.new_path = True
                break

        
    def check_path(self) -> None:
        """Constructs the path with evasion points between gates and obstacles."""
        before = self.path.start_pos.get_last()
        for i in range(4):
            points = self.path.points[i]
            after = points.get_first() # after evasion point is the first point of the next (current) gate
            path_eva= self.get_evasion_pos(before,after,i)
            before = points.get_last()
            self.path.evasion_points[i].update(self.current_pos,before, after, path_eva)
                
    def add_better_gate_angle_pos(self, gate_pos, gate_dir, point):
        """
        Check whether the angle between the gate direction and the point exceeds
        the allowed gate angle, and if so, compute a corrected gate entry position.

        Args:
            gate_pos: The gate's center position.
            gate_dir: The gate's forward direction vector.
            point: The point whose angle relative to the gate is evaluated.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - True and a corrected position if an adjustment is needed.
                - False and None if no correction is required.
        """
        alpha = angle_between(gate_dir, point - gate_pos)
        abs_alpha = abs(alpha)
        if abs_alpha > self.max_gate_angle:
            if abs_alpha <= 90:
                return True, gate_pos + self.gate_pos_offset * gate_dir
            elif abs_alpha < 130:
                return True, gate_pos - self.gate_pos_offset * gate_dir
        return False, None

    def get_colliding_obstacles(self, V1: np.ndarray, V2: np.ndarray,ignore_obs_i:List[int]):
        """
        Return the IDs of obstacles that collide with the segment from V1 to V2.

        Args:
            V1 (np.ndarray): Start point of the segment.
            V2 (np.ndarray): End point of the segment.

        Returns:
            List[int]: Sorted obstacle IDs that intersect the segment.
        """
        obstacles_pos = []
        obstacles_id = []
        for obs_i, obstacle in enumerate(self.obstacles):
            if obs_i not in ignore_obs_i and obstacle.is_colliding(V1, V2):
                obstacles_pos.append(obstacle.center_pos)
                obstacles_id.append(obs_i)

        obstacles_sorted_set = sort_by_distance(obstacles_pos, V1)

        sorted_obstacles_id = []
        for i, p, d in obstacles_sorted_set:
            sorted_obstacles_id.append(obstacles_id[i])

        return sorted_obstacles_id
    
    def get_evasion_pos(self, V1: np.ndarray, V2: np.ndarray, gate_i:int) -> List[np.ndarray]:
        """
        Compute a sequence of evasion points between V1 and V2 based on obstacle collisions.

        The method:
        - Finds the first obstacle intersecting the segment V1→V2.
        - Computes an initial evasion point (E0).
        - Optionally computes a secondary evasion point between V1→E0.
        - Optionally computes a secondary evasion point between E0→V2.
        - Returns all valid evasion points in order.

        Args:
            V1 (np.ndarray): Start point of the segment.
            V2 (np.ndarray): End point of the segment.
            gate_i (int): the gate_index used to ignore the gate_obstacles

        Returns:
            List[np.ndarray]: A list of valid evasion points. May be empty.
        """

        ignore_obs_i = [gate_i]
        if gate_i > 0:
            ignore_obs_i.append(gate_i-1)

        obstacles_id = self.get_colliding_obstacles(V1, V2, ignore_obs_i)

        new_evasion_pos = []
        if len(obstacles_id):
            obstacle = self.obstacles[obstacles_id[0]]
            eva_E0 = obstacle.get_safe_evasion_point(V1, V2, self.obstacles)

            if eva_E0 is not None:
                obstacles_id_V1_E0 = self.get_colliding_obstacles(V1, eva_E0, ignore_obs_i)
                if len(obstacles_id_V1_E0) == 1:
                    obstacle_to_id_V1_E0 = self.obstacles[obstacles_id_V1_E0[0]]
                    eva_V1_E0 = obstacle_to_id_V1_E0.get_safe_evasion_point(
                        V1, eva_E0, self.obstacles
                    )
                    if eva_V1_E0 is not None:
                        new_evasion_pos.append(eva_V1_E0)

                new_evasion_pos.append(eva_E0)

                obstacles_id_E0_V2 = self.get_colliding_obstacles(eva_E0, V2, ignore_obs_i)
                if len(obstacles_id_E0_V2) == 1:
                    obstacles_to_id_E0_V2 = self.obstacles[obstacles_id_E0_V2[0]]
                    eva_E0_V2 = obstacles_to_id_E0_V2.get_safe_evasion_point(
                        eva_E0, V2, self.obstacles
                    )
                    if eva_E0_V2 is not None:
                        new_evasion_pos.append(eva_E0_V2)

        return new_evasion_pos

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

    def interpolate_path(self, t:float) -> None:
        """Interpolates the path using linear interpolation for smooth navigation."""
        self.t_offset = 0
        self.new_path = np.asarray(self.path.get_path())
        self.N = len(self.new_path)
        # t_total = self.N / self.fly_speed # this could produce better results, when it comes to level 3
        t_values = np.linspace(0,self.trajectory_time - self.t_offset , self.N)
        self.pos_spline = interp1d(t_values, self.new_path, axis=0)
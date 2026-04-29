from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lsy_drone_racing.control.controller import Controller
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class MiControladorReto(Controller):

    def __init__(self, obs: dict[str, NDArray[np.floating]], info, config):
        super().__init__(obs, info, config)

        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self.gates =obs.get("gates_pos", [])

        self.gates_squat= obs.get("gates_quat", [])
      
        print("Inicial gatesquat:", self.gates_squat)

        self._t_total = 0.0
        pos_inicial = obs["pos"]

        t_dummy = [0.0, 1.0]
        pos_dummy = np.vstack([pos_inicial, pos_inicial])
        self._des_pos_spline = CubicSpline(t_dummy, pos_dummy)

       



        self.current_gate_index = 0
        self.target_yaw = 0.0


        self.update(obs, info, is_init=True)
        
        
        
        

      


    def compute_control(
            self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
        ) -> NDArray[np.floating]:

# --- --------------------
        
            self.update(obs,info)



            t = min(self._tick / self._freq, self._t_total)
           

            des_pos = self._des_pos_spline(t)
            velocity =self._des_pos_spline(t, 1)  # Velocidad deseada (derivada de la posición)
            
            action = np.zeros(13, dtype=np.float32)
            action[0:3] = des_pos[:3] # Posición x, y, z
            action[3:6] = velocity[:3] # Velocidad vx, vy, vz
            
            if np.linalg.norm(velocity[:2]) > 0.1:
                self.target_yaw = np.arctan2(velocity[1], velocity[0])
            
            # Aplica el yaw guardado (ya sea el recién calculado o el anterior)
            action[9] = self.target_yaw

            self._tick += 1


            return action

    def step_callback(self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool: 
     
     
        self._tick += 1
        return self._finished
        

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

    def quat_to_forward(self, q):
        x, y, z, w = q

        fx = 1 - 2*(y*y + z*z)
        fy = 2*(x*y + w*z)
        fz = 2*(x*z - w*y)

        fwd = np.array([fx, fy, fz])
        return fwd / (np.linalg.norm(fwd) + 1e-6)


    def update(self, obs, info, is_init=False):

        new_gates = obs.get("gates_pos", [])
        new_quats = obs.get("gates_quat", [])
        visited = np.array(obs.get("gates_visited", []), dtype=bool)

        if len(new_gates) == 0:
            return

        new_gates = np.array(new_gates)
        new_quats = np.array(new_quats)

        unvisited_idx = ~visited
        unvisited_gates = new_gates[unvisited_idx]
        unvisited_quats = new_quats[unvisited_idx]

        if len(unvisited_gates) == 0:
            return

        # only recompute if needed
        if (is_init or
            len(unvisited_gates) != len(self.gates) or
            not np.allclose(unvisited_gates, self.gates, atol=0.01)):

            self.gates = unvisited_gates
            self.gates_squat = unvisited_quats

            waypoints = []
            start_pos = obs["pos"]

            prev_pos = start_pos

            for i in range(len(unvisited_gates)):

                gate_pos = unvisited_gates[i]
                gate_quat = unvisited_quats[i]

                # --- direction from quaternion
                dir_vec = self.quat_to_forward(gate_quat)

                # --- shortest direction fix
                to_gate = gate_pos - prev_pos
                to_gate = to_gate / (np.linalg.norm(to_gate) + 1e-6)

                if np.dot(dir_vec, to_gate) < 0:
                    dir_vec = -dir_vec

                # --- adaptive offset
                if i < len(unvisited_gates) - 1:
                    next_gate = unvisited_gates[i + 1]
                    dist_next = np.linalg.norm(next_gate - gate_pos)
                else:
                    dist_next = np.linalg.norm(gate_pos - prev_pos)

                d = np.clip(0.25 * dist_next, 0.3, 1.0)

                # --- APPROACH POINT (key change)
                approach = gate_pos - dir_vec * d

                waypoints.append(approach)

                prev_pos = gate_pos

            # add final position (optional stabilizer)
            waypoints.append(unvisited_gates[-1])

            waypoints_xyz = np.vstack([obs["pos"], np.array(waypoints)])

            # --- spline timing (unchanged)
            diffs = np.diff(waypoints_xyz, axis=0)
            distances = np.linalg.norm(diffs, axis=1)

            Velocity = 1
            distances = np.maximum(distances, 0.01)
            times_segment = distances / Velocity

            t = np.zeros(len(waypoints_xyz))
            t[1:] = np.cumsum(times_segment)

            self._t_total = t[-1]

            current_vel = obs.get("vel", np.zeros(3))

            self._des_pos_spline = CubicSpline(
                t, waypoints_xyz,
                bc_type=((1, current_vel), 'not-a-knot')
            )

            self._tick = 0
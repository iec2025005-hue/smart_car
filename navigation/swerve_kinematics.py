"""
navigation/swerve_kinematics.py -- Inverse kinematics for a 4-wheel swerve drive.

Calculates the required steering angle and drive speed for each of the 4 wheel pods
based on a desired translational velocity (Vx, Vy) and rotational velocity (Omega).
"""

import math
from typing import Tuple, List

import config

class SwerveKinematics:
    def __init__(self):
        # L = distance between front and rear axles
        self.L = config.SWERVE_WHEELBASE_LENGTH
        # W = distance between left and right wheels
        self.W = config.SWERVE_TRACK_WIDTH
        self.max_speed = config.MAX_DRIVE_SPEED

    def calculate(self, vx: float, vy: float, omega: float) -> List[Tuple[float, float]]:
        """
        Calculate wheel angles and speeds.
        
        Parameters
        ----------
        vx : float
            Forward velocity (positive is forward).
        vy : float
            Strafe velocity (positive is right).
        omega : float
            Rotational velocity (positive is clockwise).
            
        Returns
        -------
        list of (angle_deg, speed) tuples for:
            0: Front-Right
            1: Front-Left
            2: Rear-Left
            3: Rear-Right
        Angles are in degrees (0 = forward, 90 = right).
        Speeds are scaled to config.MAX_DRIVE_SPEED.
        """
        # Intermediate variables
        A = vx - omega * (self.W / 2.0)
        B = vx + omega * (self.W / 2.0)
        C = vy - omega * (self.L / 2.0)
        D = vy + omega * (self.L / 2.0)
        
        # Calculate speeds and angles
        # Front-Right (x=L/2, y=W/2) -> B, C
        s_fr = math.hypot(B, C)
        a_fr = math.atan2(C, B) * 180.0 / math.pi
        
        # Front-Left (x=L/2, y=-W/2) -> B, D
        s_fl = math.hypot(B, D)
        a_fl = math.atan2(D, B) * 180.0 / math.pi
        
        # Rear-Left (x=-L/2, y=-W/2) -> A, D
        s_rl = math.hypot(A, D)
        a_rl = math.atan2(D, A) * 180.0 / math.pi
        
        # Rear-Right (x=-L/2, y=W/2) -> A, C
        s_rr = math.hypot(A, C)
        a_rr = math.atan2(C, A) * 180.0 / math.pi
        
        # Normalize speeds if any exceeds max_speed
        max_calculated = max(s_fr, s_fl, s_rl, s_rr)
        if max_calculated > self.max_speed:
            scale = self.max_speed / max_calculated
            s_fr *= scale
            s_fl *= scale
            s_rl *= scale
            s_rr *= scale
            
        # Normalize angles to [0, 360)
        a_fr = a_fr % 360.0
        a_fl = a_fl % 360.0
        a_rl = a_rl % 360.0
        a_rr = a_rr % 360.0
        
        return [
            (a_fr, s_fr),
            (a_fl, s_fl),
            (a_rl, s_rl),
            (a_rr, s_rr)
        ]

if __name__ == "__main__":
    print("Testing Swerve Kinematics")
    kinematics = SwerveKinematics()
    
    print("\n1. Forward Only (vx=100, vy=0, w=0)")
    res = kinematics.calculate(100, 0, 0)
    for i, (a, s) in enumerate(res):
        print(f"  Wheel {i}: Angle {a:5.1f}°, Speed {s:5.1f}")
        
    print("\n2. Strafe Right Only (vx=0, vy=100, w=0)")
    res = kinematics.calculate(0, 100, 0)
    for i, (a, s) in enumerate(res):
        print(f"  Wheel {i}: Angle {a:5.1f}°, Speed {s:5.1f}")
        
    print("\n3. Rotate Clockwise Only (vx=0, vy=0, w=10)")
    res = kinematics.calculate(0, 0, 10)
    for i, (a, s) in enumerate(res):
        print(f"  Wheel {i}: Angle {a:5.1f}°, Speed {s:5.1f}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Illuminance Simulator (Direct-only) for Rectangular Rooms using IES Photometry

Features
--------
- Parse IES LM-63 photometric files (commonly LM-63-2002/2019). Supports:
  * TILT=NONE
  * TILT with tabular data (applied via cosine interpolation)
  * Type C photometric web (C angles / vertical gamma angles)
- Place any number of luminaires (position + orientation in yaw/pitch/roll)
- Define calculation planes: work plane (z=const), walls (x=0/W, y=0/D), ceiling (z=H)
- Compute point-by-point illuminance (lux) on planes (direct light only; no inter-reflections)
- Export CSV and PNG heatmap for each plane
- Example usage provided at the bottom

Important Notes
---------------
- This is a *direct illuminance* calculator. It does **not** account for inter-reflections or obstructions.
  For high-accuracy design, use a validated lighting simulation suite (e.g., Radiance, DIALux, Relux).
- Coordinate System (Room): X to width (W), Y to depth (D), Z up. Floor at z=0.
- Luminaire Local Axes: By default the luminaire "down" axis is negative Z local axis.
  Type C web assumes gamma=0 is along the local -Z (down), gamma=90 horizontal, gamma=180 up.
- Orientations use ZYX intrinsic Euler angles: yaw (about Z), pitch (about Y), roll (about X).
- Units: IES intensities are in candela per lamp set. Illuminance results are in lux.

Author: ChatGPT
License: MIT
"""

from __future__ import annotations
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import csv
import matplotlib.pyplot as plt
import os

# ---------------------------- IES Parsing ------------------------------------

@dataclass
class IESData:
    manufacturer: str
    lumens: float
    candela: np.ndarray      # shape (nH, nV)
    h_angles: np.ndarray     # C angles in degrees (0..360 or 0..180)
    v_angles: np.ndarray     # Gamma angles in degrees (0..180)
    n_lamps: int
    lumens_per_lamp: float
    candela_multiplier: float
    ballast_factor: float
    input_watts: float
    tilt_vertical_angles: Optional[np.ndarray] = None
    tilt_multipliers: Optional[np.ndarray] = None

def _read_until_numbers(line: str) -> List[float]:
    # Extract floats from a line
    out = []
    for tok in line.replace(',', ' ').split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out

def parse_ies(path: str) -> IESData:
    """
    Minimal but practical IES LM-63 parser for common files.
    Focuses on Type C data and TILT parsing.
    """
    with open(path, 'r', encoding='latin-1') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() != '']

    # Header/meta collection (simplified)
    idx = 0
    manufacturer = ""
    while idx < len(lines) and not lines[idx].upper().startswith("TILT="):
        if lines[idx].startswith('[') or lines[idx].startswith('!'):
            # comment or keyword, optionally capture manufacturer
            if 'MANUFAC' in lines[idx].upper():
                manufacturer = lines[idx].split(':', 1)[-1].strip()
        idx += 1

    if idx >= len(lines):
        raise ValueError("Malformed IES: missing TILT line")

    tilt_line = lines[idx]
    idx += 1

    tilt_vertical_angles = None
    tilt_multipliers = None

    if tilt_line.upper().startswith("TILT=NONE"):
        pass
    elif tilt_line.upper().startswith("TILT="):
        # TILT=<filename> or TILT=INCLUDE
        # For simplicity, expect an inline table per LM-63 when not NONE.
        # Next lines: num_tilt_angles, then list of angles, then list of mults
        ntilt = int(float(lines[idx].split()[0])); idx += 1
        tilt_vertical_angles = []
        while len(tilt_vertical_angles) < ntilt:
            tilt_vertical_angles += _read_until_numbers(lines[idx])
            idx += 1
        tilt_vertical_angles = np.array(tilt_vertical_angles[:ntilt], dtype=float)

        tilt_multipliers = []
        while len(tilt_multipliers) < ntilt:
            tilt_multipliers += _read_until_numbers(lines[idx])
            idx += 1
        tilt_multipliers = np.array(tilt_multipliers[:ntilt], dtype=float)
    else:
        raise ValueError("Unsupported TILT specification")

    # Numeric block
    # Per LM-63: nlamps, lumens per lamp, candela multiplier, nV, nH, photometric type, units type,
    # width, length, height, ballast factor, future, input watts
    nums = []
    while len(nums) < 10:
        nums += _read_until_numbers(lines[idx])
        idx += 1
    n_lamps = int(nums[0])
    lumens_per_lamp = float(nums[1])
    candela_multiplier = float(nums[2])
    nV = int(nums[3])
    nH = int(nums[4])
    photometric_type = int(nums[5])  # 1=A,2=B,3=C (support only C)
    units_type = int(nums[6])        # 1=feet, 2=meters (we ignore; internal is meters)
    dims = nums[7:10]
    # ballast factor and watts may be on subsequent line(s)
    extra = []
    while len(extra) < 2:
        extra += _read_until_numbers(lines[idx])
        idx += 1
    ballast_factor = float(extra[0])
    input_watts = float(extra[1])

    if photometric_type != 3:
        raise ValueError("Only Type C photometry is supported in this simulator.")

    # Vertical (gamma) angles
    v_angles = []
    while len(v_angles) < nV:
        v_angles += _read_until_numbers(lines[idx])
        idx += 1
    v_angles = np.array(v_angles[:nV], dtype=float)

    # Horizontal (C) angles
    h_angles = []
    while len(h_angles) < nH:
        h_angles += _read_until_numbers(lines[idx])
        idx += 1
    h_angles = np.array(h_angles[:nH], dtype=float)

    # Candela values: nH blocks each with nV values (per LM-63)
    cd = []
    for _ in range(nH * nV):
        cd += _read_until_numbers(lines[idx])
        idx += 1
    cd = np.array(cd[:nH*nV], dtype=float).reshape(nH, nV) * candela_multiplier

    ies = IESData(
        manufacturer=manufacturer,
        lumens=n_lamps * lumens_per_lamp * ballast_factor,
        candela=cd,
        h_angles=h_angles,
        v_angles=v_angles,
        n_lamps=n_lamps,
        lumens_per_lamp=lumens_per_lamp,
        candela_multiplier=candela_multiplier,
        ballast_factor=ballast_factor,
        input_watts=input_watts,
        tilt_vertical_angles=tilt_vertical_angles,
        tilt_multipliers=tilt_multipliers
    )
    return ies

# -------------------- Interpolation over photometric web ---------------------

def _wrap_angle_deg(a):
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def _interp1d(x: np.ndarray, y: np.ndarray, xi: float) -> float:
    # simple linear interpolation with clamp
    if xi <= x[0]:
        return float(y[0])
    if xi >= x[-1]:
        return float(y[-1])
    j = np.searchsorted(x, xi) - 1
    x0, x1 = x[j], x[j+1]
    y0, y1 = y[j], y[j+1]
    t = (xi - x0) / (x1 - x0)
    return float(y0 * (1 - t) + y1 * t)

def _interp2d(h_angles: np.ndarray, v_angles: np.ndarray, candela: np.ndarray, C: float, G: float, wrap_horizontal: bool=True) -> float:
    """
    Bilinear interpolation of candela at (C, G).
    h_angles: (nH), v_angles: (nV), candela: (nH, nV)
    """
    # Handle wrap-around for horizontal (C) if covers 0..360
    if wrap_horizontal and h_angles[-1] - h_angles[0] >= 360 - 1e-3:
        C = _wrap_angle_deg(C)
        # ensure h_angles also includes 360 if starts at 0
        H = np.append(h_angles, h_angles[0] + 360.0)
        Cd = np.vstack([candela, candela[0:1, :]])
    else:
        H = h_angles.copy()
        Cd = candela

    # indices
    if C <= H[0]:
        i0 = 0
        i1 = 1
        th = 0.0
    elif C >= H[-1]:
        i0 = len(H) - 2
        i1 = len(H) - 1
        th = 1.0
    else:
        i1 = np.searchsorted(H, C)
        i0 = i1 - 1
        th = (C - H[i0]) / (H[i1] - H[i0])

    # vertical interpolation weights
    if G <= v_angles[0]:
        j0, j1, tv = 0, 1, 0.0
    elif G >= v_angles[-1]:
        j0, j1, tv = len(v_angles)-2, len(v_angles)-1, 1.0
    else:
        j1 = np.searchsorted(v_angles, G)
        j0 = j1 - 1
        tv = (G - v_angles[j0]) / (v_angles[j1] - v_angles[j0])

    c00 = Cd[i0, j0]
    c10 = Cd[i1, j0]
    c01 = Cd[i0, j1]
    c11 = Cd[i1, j1]
    c0 = c00 * (1 - th) + c10 * th
    c1 = c01 * (1 - th) + c11 * th
    c = c0 * (1 - tv) + c1 * tv
    return float(c)

# --------------------------- Geometry utilities ------------------------------

def rotation_matrix_from_euler(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return 3x3 rotation matrix for ZYX intrinsic Euler angles in degrees."""
    z = math.radians(yaw_deg)
    y = math.radians(pitch_deg)
    x = math.radians(roll_deg)
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])
    Rx = np.array([[1,  0,   0],
                   [0, cx, -sx],
                   [0, sx,  cx]])
    return Rz @ Ry @ Rx

# --------------------------- Scene definitions -------------------------------

@dataclass
class Luminaire:
    ies: IESData
    position: Tuple[float, float, float]  # (x, y, z) meters
    yaw: float = 0.0    # deg about +Z
    pitch: float = 0.0  # deg about +Y
    roll: float = 0.0   # deg about +X
    luminous_flux_scale: float = 1.0      # extra scaling if needed

    def direction_to_angles(self, vec_room: np.ndarray) -> Tuple[float, float]:
        """
        Convert a vector (from luminaire to point) in room coords to (C, G) in luminaire local coords.
        Returns (C, G) degrees for Type C web.
        """
        # Transform into luminaire local coordinates
        R = rotation_matrix_from_euler(self.yaw, self.pitch, self.roll)
        # from room to local: R^T
        v_local = R.T @ vec_room
        # In luminaire local coordinates: local -Z is gamma=0 (down)
        # Compute gamma as angle from -Z axis
        # Normalize
        n = np.linalg.norm(v_local)
        if n < 1e-9:
            return 0.0, 0.0
        v = v_local / n
        # gamma: angle between -Z and v
        cos_g = np.dot(-np.array([0,0,1.0]), v)
        cos_g = max(-1.0, min(1.0, cos_g))
        G = math.degrees(math.acos(cos_g))
        # C angle: azimuth around +Z, measured from +X towards +Y
        C = math.degrees(math.atan2(v[1], v[0]))
        if C < 0:
            C += 360.0
        return C, G

@dataclass
class Plane:
    name: str
    origin: Tuple[float, float, float]  # a point on plane
    normal: Tuple[float, float, float]  # plane normal (unit not required)
    u_dir: Tuple[float, float, float]   # axis along plane (first grid axis)
    v_dir: Tuple[float, float, float]   # axis along plane (second grid axis)
    u_len: float                        # length along u (m)
    v_len: float                        # length along v (m)
    u_steps: int                        # grid cells along u
    v_steps: int                        # grid cells along v

    def grid_points(self) -> np.ndarray:
        """Return array of shape (N, 3) of grid points centered in each cell."""
        u = np.array(self.u_dir, dtype=float)
        v = np.array(self.v_dir, dtype=float)
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        n = np.array(self.normal, dtype=float)
        # enforce v orthogonal to u? Assume provided consistent.

        du = self.u_len / self.u_steps
        dv = self.v_len / self.v_steps
        pts = []
        o = np.array(self.origin, dtype=float)
        for i in range(self.u_steps):
            for j in range(self.v_steps):
                # center point of each cell
                pu = (i + 0.5) * du
                pv = (j + 0.5) * dv
                p = o + u * pu + v * pv
                pts.append(p)
        return np.array(pts)

    def incidence_cosines(self) -> float:
        """Return the plane normal (unit) for incidence calculations."""
        n = np.array(self.normal, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            raise ValueError("Plane normal cannot be zero vector")
        return n / n_norm

@dataclass
class Room:
    width: float   # X (m)
    depth: float   # Y (m)
    height: float  # Z (m)
    luminaires: List[Luminaire] = field(default_factory=list)

    def add_luminaire(self, lum: Luminaire):
        self.luminaires.append(lum)

# ------------------------ Illuminance calculation ----------------------------

def compute_illuminance_on_plane(room: Room, plane: Plane) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute direct illuminance (lux) on the given plane.
    Returns (E_grid, mask) where E_grid has shape (u_steps, v_steps).
    """
    pts = plane.grid_points() # (N,3)
    N = pts.shape[0]
    E = np.zeros(N, dtype=float)
    n_hat = plane.incidence_cosines()

    # For each point, sum contributions from each luminaire
    for k, lum in enumerate(room.luminaires):
        ies = lum.ies
        for i in range(N):
            p = pts[i]
            src = np.array(lum.position, dtype=float)
            vec = p - src
            r2 = float(np.dot(vec, vec))
            if r2 < 1e-9:
                continue
            r = math.sqrt(r2)
            # direction angles in lum local
            C, G = lum.direction_to_angles(vec)
            # intensity
            I = _interp2d(ies.h_angles, ies.v_angles, ies.candela, C, G)
            # apply TILT if provided (approximate using gamma)
            if ies.tilt_vertical_angles is not None and ies.tilt_multipliers is not None:
                tilt_mult = _interp1d(ies.tilt_vertical_angles, ies.tilt_multipliers, G)
                I *= tilt_mult
            # cosine of incidence with plane normal (ray comes from src to p)
            ray_dir = vec / r
            cos_inc = np.dot(-ray_dir, n_hat)  # negative because light arrives onto plane
            if cos_inc <= 0:
                continue  # grazing from backside; no contribution
            E[i] += I * cos_inc / r2

    # reshape to grid
    E_grid = E.reshape(plane.u_steps, plane.v_steps)
    return E_grid, pts

# ----------------------------- Helpers ---------------------------------------

def save_csv(path: str, grid: np.ndarray, plane: Plane):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([plane.name])
        writer.writerow(["u_steps", plane.u_steps, "v_steps", plane.v_steps])
        for i in range(grid.shape[0]):
            writer.writerow(list(np.round(grid[i], 3)))

def save_heatmap_png(path: str, grid: np.ndarray, plane: Plane, vmax: Optional[float]=None):
    plt.figure()
    plt.imshow(grid.T, origin='lower', aspect='auto', vmax=vmax)
    plt.colorbar(label='Lux')
    plt.title(f"{plane.name} (lux)")
    plt.xlabel('u index')
    plt.ylabel('v index')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def make_workplane(room: Room, z: float, u_steps=40, v_steps=40) -> Plane:
    if z < 0 or z > room.height:
        raise ValueError("Workplane z must be within room height")
    return Plane(
        name=f"Workplane z={z} m",
        origin=(0.0, 0.0, z),
        normal=(0.0, 0.0, 1.0),
        u_dir=(room.width, 0.0, 0.0),
        v_dir=(0.0, room.depth, 0.0),
        u_len=room.width,
        v_len=room.depth,
        u_steps=u_steps,
        v_steps=v_steps
    )

def make_wall(room: Room, which: str, u_steps=40, v_steps=40) -> Plane:
    which = which.lower()
    if which == 'x0':
        return Plane(
            name="Wall x=0",
            origin=(0.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
            u_dir=(0.0, room.depth, 0.0),
            v_dir=(0.0, 0.0, room.height),
            u_len=room.depth,
            v_len=room.height,
            u_steps=u_steps,
            v_steps=v_steps
        )
    elif which == 'xw':
        return Plane(
            name=f"Wall x={room.width}",
            origin=(room.width, 0.0, 0.0),
            normal=(-1.0, 0.0, 0.0),
            u_dir=(0.0, room.depth, 0.0),
            v_dir=(0.0, 0.0, room.height),
            u_len=room.depth,
            v_len=room.height,
            u_steps=u_steps,
            v_steps=v_steps
        )
    elif which == 'y0':
        return Plane(
            name="Wall y=0",
            origin=(0.0, 0.0, 0.0),
            normal=(0.0, 1.0, 0.0),
            u_dir=(room.width, 0.0, 0.0),
            v_dir=(0.0, 0.0, room.height),
            u_len=room.width,
            v_len=room.height,
            u_steps=u_steps,
            v_steps=v_steps
        )
    elif which == 'yd':
        return Plane(
            name=f"Wall y={room.depth}",
            origin=(0.0, room.depth, 0.0),
            normal=(0.0, -1.0, 0.0),
            u_dir=(room.width, 0.0, 0.0),
            v_dir=(0.0, 0.0, room.height),
            u_len=room.width,
            v_len=room.height,
            u_steps=u_steps,
            v_steps=v_steps
        )
    else:
        raise ValueError("which must be one of: 'x0', 'xw', 'y0', 'yd'")

def make_ceiling(room: Room, u_steps=40, v_steps=40) -> Plane:
    return Plane(
        name=f"Ceiling z={room.height} m",
        origin=(0.0, 0.0, room.height),
        normal=(0.0, 0.0, -1.0),
        u_dir=(room.width, 0.0, 0.0),
        v_dir=(0.0, room.depth, 0.0),
        u_len=room.width,
        v_len=room.depth,
        u_steps=u_steps,
        v_steps=v_steps
    )

# ------------------------------- Runner --------------------------------------

def run_simulation(
    ies_path: str,
    room_size: Tuple[float, float, float],
    lum_positions: List[Tuple[float, float, float]],
    lum_orientations: Optional[List[Tuple[float,float,float]]] = None,
    planes_spec: Optional[List[Dict]] = None,
    out_dir: str = "outputs",
    heatmap_vmax: Optional[float] = None
):
    """
    High-level convenience function to run a case.
    - ies_path: path to IES file
    - room_size: (W, D, H) meters
    - lum_positions: list of (x,y,z) meters
    - lum_orientations: list of (yaw, pitch, roll) degrees for each luminaire (optional)
    - planes_spec: list of dicts with keys {"type": "workplane"|"wall"|"ceiling", ...}
      * workplane: {"z": 0.8, "u_steps": 40, "v_steps": 40}
      * wall: {"which": "x0"|"xw"|"y0"|"yd", "u_steps": .., "v_steps": ..}
      * ceiling: {"u_steps": .., "v_steps": ..}
    - out_dir: output directory (CSV + PNG per plane)
    - heatmap_vmax: optional fixed max for heatmap color scale (lux)
    """
    ies = parse_ies(ies_path)

    W, D, H = room_size
    room = Room(W, D, H)

    if lum_orientations is None:
        lum_orientations = [(0.0, 0.0, 0.0)] * len(lum_positions)

    for pos, ang in zip(lum_positions, lum_orientations):
        room.add_luminaire(Luminaire(ies=ies, position=pos, yaw=ang[0], pitch=ang[1], roll=ang[2]))

    if planes_spec is None:
        planes_spec = [
            {"type": "workplane", "z": 0.8, "u_steps": 40, "v_steps": 40},
            {"type": "wall", "which": "x0", "u_steps": 40, "v_steps": 40},
            {"type": "wall", "which": "y0", "u_steps": 40, "v_steps": 40},
            {"type": "ceiling", "u_steps": 40, "v_steps": 40},
        ]

    planes: List[Plane] = []
    for spec in planes_spec:
        t = spec["type"].lower()
        if t == "workplane":
            planes.append(make_workplane(room, z=float(spec["z"]), u_steps=spec.get("u_steps",40), v_steps=spec.get("v_steps",40)))
        elif t == "wall":
            planes.append(make_wall(room, which=spec["which"], u_steps=spec.get("u_steps",40), v_steps=spec.get("v_steps",40)))
        elif t == "ceiling":
            planes.append(make_ceiling(room, u_steps=spec.get("u_steps",40), v_steps=spec.get("v_steps",40)))
        else:
            raise ValueError(f"Unknown plane type: {t}")

    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for plane in planes:
        grid, pts = compute_illuminance_on_plane(room, plane)
        csv_path = os.path.join(out_dir, f"{plane.name.replace(' ','_').replace('=','_').replace('.','p')}.csv")
        png_path = os.path.join(out_dir, f"{plane.name.replace(' ','_').replace('=','_').replace('.','p')}.png")
        save_csv(csv_path, grid, plane)
        save_heatmap_png(png_path, grid, plane, vmax=heatmap_vmax)
        summary[plane.name] = {"csv": csv_path, "png": png_path, "min_lux": float(np.min(grid)), "avg_lux": float(np.mean(grid)), "max_lux": float(np.max(grid))}

    # Save JSON summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# -------------------------- Example template ---------------------------------

EXAMPLE = r"""
# Example usage (edit paths and positions, then run this file):
# python illuminance_sim.py --ies "/path/to/luminaire.ies" \
#   --room 6 10 3.0 \
#   --grid 3 2 --mount 2.8 \
#   --workplane 0.8 --walls x0 y0 xw yd --ceiling \
#   --out outputs_case1
#
# The --grid and --mount options create a rectangular array of luminaires:
#   --grid NX NY --mount Z   places NX x NY luminaires evenly over the room at height Z.

"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Direct Illuminance Simulator using IES photometry (Type C).")
    parser.add_argument("--ies", required=True, help="Path to IES file")
    parser.add_argument("--room", nargs=3, type=float, required=True, metavar=("W","D","H"), help="Room size in meters")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("NX","NY"), help="Grid of luminaires")
    parser.add_argument("--mount", type=float, help="Mounting height (Z) for grid luminaires (m)")
    parser.add_argument("--positions", nargs='*', type=float, help="Custom luminaire positions as flat list x1 y1 z1 x2 y2 z2 ...")
    parser.add_argument("--orient", nargs='*', type=float, help="Optional orientations per luminaire (yaw pitch roll repeated)")
    parser.add_argument("--workplane", type=float, help="Workplane height z (m)")
    parser.add_argument("--walls", nargs='*', choices=["x0","xw","y0","yd"], help="Which walls to compute")
    parser.add_argument("--ceiling", action="store_true", help="Include ceiling plane")
    parser.add_argument("--u_steps", type=int, default=40, help="Grid resolution along U")
    parser.add_argument("--v_steps", type=int, default=40, help="Grid resolution along V")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed max for heatmap color scale (lux)")

    args = parser.parse_args()

    W, D, H = args.room

    # Build luminaire position list
    lum_positions = []
    if args.positions:
        vals = args.positions
        if len(vals) % 3 != 0:
            raise SystemExit("positions must be in triples x y z")
        for i in range(0, len(vals), 3):
            lum_positions.append((vals[i], vals[i+1], vals[i+2]))
    elif args.grid and args.mount is not None:
        NX, NY = args.grid
        z = args.mount
        xs = np.linspace(W/(NX+1), W*(NX/(NX+1)), NX)
        ys = np.linspace(D/(NY+1), D*(NY/(NY+1)), NY)
        for x in xs:
            for y in ys:
                lum_positions.append((float(x), float(y), float(z)))
    else:
        raise SystemExit("Provide either --positions or both --grid and --mount")

    # Orientations
    lum_orients = None
    if args.orient:
        if len(args.orient) % 3 != 0:
            raise SystemExit("orient must be yaw pitch roll per luminaire")
        triples = []
        for i in range(0, len(args.orient), 3):
            triples.append((args.orient[i], args.orient[i+1], args.orient[i+2]))
        if len(triples) != len(lum_positions):
            raise SystemExit("number of orient triples must match number of luminaires")
        lum_orients = triples

    # Planes
    planes_spec = []
    if args.workplane is not None:
        planes_spec.append({"type": "workplane", "z": args.workplane, "u_steps": args.u_steps, "v_steps": args.v_steps})
    if args.walls:
        for w in args.walls:
            planes_spec.append({"type": "wall", "which": w, "u_steps": args.u_steps, "v_steps": args.v_steps})
    if args.ceiling:
        planes_spec.append({"type": "ceiling", "u_steps": args.u_steps, "v_steps": args.v_steps})

    if not planes_spec:
        # default to a workplane at 0.8 m
        planes_spec = [{"type": "workplane", "z": 0.8, "u_steps": args.u_steps, "v_steps": args.v_steps}]

    summary = run_simulation(
        ies_path=args.ies,
        room_size=(W, D, H),
        lum_positions=lum_positions,
        lum_orientations=lum_orients,
        planes_spec=planes_spec,
        out_dir=args.out,
        heatmap_vmax=args.vmax
    )

    print("Outputs:")
    for name, rec in summary.items():
        print(f"- {name}: CSV={rec['csv']}, PNG={rec['png']}, min/avg/max={rec['min_lux']:.2f}/{rec['avg_lux']:.2f}/{rec['max_lux']:.2f} lux")

# pacman_env.py
from __future__ import annotations
import math, random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# ───────── rendering constants ─────────
PIXELS_PER_CELL = 12          # tile size

# ───────── 7×7 boards (unchanged) ───────
tiny_walls = np.array(
    [[1,1,1,1,1,1,1],
     [1,0,0,0,0,0,1],
     [1,0,1,1,1,0,1],
     [1,0,1,0,1,0,1],
     [1,0,1,0,0,0,1],
     [1,0,0,0,1,0,1],
     [1,1,1,1,1,1,1]], dtype=int)

LAYOUTS: dict[str, np.ndarray] = {
    "empty": np.zeros((7, 7), dtype=int),
    "spiral": tiny_walls.copy(),
    "spiral_harder": tiny_walls.copy(),
}

# ───────── classic 19×15 maze with corridors ─────────
classic_board = np.array([
 [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
 [1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
 [1,0,1,1,1,1,0,1,0,1,1,0,1,0,1,1,1,1,0,0,1],
 [1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1],
 [1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,0,1],
 [1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
 [1,1,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1],
 [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],  # portals left/right (value 0)
 [1,1,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1],
 [1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
 [1,0,1,1,1,1,0,1,0,1,1,0,1,0,1,1,1,1,0,0,1],
 [1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1],
 [1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,0,1],
 [1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
 [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=int)

LAYOUTS["classic"] = classic_board

# ───────── colour table (indices used in board array) ─────────
# 0 empty, 1 wall, 2 pac‑man, 3 red ghost, 4 pellet, 5 blue ghost
COLOR_TABLE = np.array(
    [[0,0,0],   [80,80,80], [255,255,0],
     [255,0,0], [0,255,0],  [0,128,255]], dtype=np.uint8)

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

# ────────────────────────────────────────────────────────────────
class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # ───────────────────────────── init ─────────────────────────
    def __init__(self, layout: str = "empty"):
        super().__init__()
        if layout not in LAYOUTS:
            raise ValueError(layout)
        self.layout_name = layout
        self.floor = LAYOUTS[layout]          # 0/1 grid
        self.h, self.w = self.floor.shape     # board dims
        self.img_h = self.h * PIXELS_PER_CELL
        self.img_w = self.w * PIXELS_PER_CELL

        self._build_tiles()                   # pixel‑art sprites

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            0, 255, shape=(self.img_h, self.img_w, 3), dtype=np.uint8
        )

        self.rng = np.random.default_rng()
        self.reset()

    # ───────────────────────── helpers ──────────────────────────
    def _legal_neighbours(self, x: int, y: int):
        out = []
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.h and 0 <= ny < self.w and self.floor[nx, ny] == 0:
                out.append((nx, ny))
        return out

    # ───────────────────────── reset ────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.layout_name == "empty":
            self.pac_pos = (0, 0)
            self.ghost_pos = [(6, 6)]
            self.pellets = [(0, 6), (6, 0)]

        elif self.layout_name == "spiral":
            self.pac_pos = (1, 1)
            self.ghost_pos = [(5, 5)]
            self.pellets = [(1, 5), (5, 1)]

        elif self.layout_name == "spiral_harder":
            self.pac_pos = (1, 1)
            self.ghost_pos = [(3, 3)]
            self.pellets = [(1, 5), (5, 1)]

        else:  # classic
            self.pac_pos = (13, 10)           # lower middle corridor
            self.ghost_pos = [(7, 9), (7, 11)]  # red, blue
            # pellets everywhere except walls and starting positions
            self.pellets = [(i, j) for i in range(self.h) for j in range(self.w)
                            if self.floor[i, j] == 0 and
                               (i, j) not in self.ghost_pos + [self.pac_pos]]

        self.first_ghost_move = True                    # only used on "empty"
        self.ghost_dir: List[Tuple[int, int] | None] = [None] * len(self.ghost_pos)
        return self._render_board(), {}

    # ───────────────────────── step ────────────────────────────
    def step(self, action: int):
        # move Pac‑Man
        px, py = self.pac_pos
        if   action == 0: px = max(px-1, 0)
        elif action == 1: px = min(px+1, self.h-1)
        elif action == 2: py = max(py-1, 0)
        elif action == 3: py = min(py+1, self.w-1)
        if self.floor[px, py]: px, py = self.pac_pos
        self.pac_pos = (px, py)

        reward, terminated = -0.1, False
        
        # CHECK 1: Collision immediately after Pac-Man moves
        if self.pac_pos in self.ghost_pos:
            reward -= 50
            terminated = True
            return self._render_board(), reward, terminated, False, {}
        
        if self.pac_pos in self.pellets:
            self.pellets.remove(self.pac_pos); reward += 10
            if not self.pellets: reward += 50; terminated = True

        # move each ghost
        if not terminated:
            for g_idx, (gx, gy) in enumerate(self.ghost_pos):
                if self.layout_name == "empty":
                    if self.first_ghost_move:
                        gx, gy = self.rng.choice(self._legal_neighbours(gx, gy))
                        self.first_ghost_move = False
                    else:
                        dx, dy = px - gx, py - gy
                        if self.rng.random() < 0.7:
                            if abs(dx) > abs(dy): gx += int(math.copysign(1, dx))
                            elif dy:              gy += int(math.copysign(1, dy))
                        else:
                            gx += self.rng.choice([-1, 0, 1])
                            gy += self.rng.choice([-1, 0, 1])
                        gx = int(np.clip(gx, 0, self.h-1))
                        gy = int(np.clip(gy, 0, self.w-1))
                        if self.floor[gx, gy]:
                            gx, gy = self.ghost_pos[g_idx]

                else:  # corridor‑following ghost
                    dir_ = self.ghost_dir[g_idx]
                    if dir_ is None:
                        gx, gy = self.rng.choice(self._legal_neighbours(gx, gy))
                        self.ghost_dir[g_idx] = (gx - self.ghost_pos[g_idx][0],
                                                gy - self.ghost_pos[g_idx][1])
                    else:
                        dx, dy = dir_
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.h and 0 <= ny < self.w and self.floor[nx, ny] == 0:
                            legal = self._legal_neighbours(gx, gy)
                            corridor = len(legal) == 2 and (nx, ny) in legal
                            if corridor:
                                gx, gy = nx, ny
                            else:
                                candidates = [(lx - gx, ly - gy) for lx, ly in legal]
                                rev = (-dx, -dy)
                                if len(candidates) > 1 and rev in candidates:
                                    candidates.remove(rev)
                                dx, dy = self.rng.choice(candidates)
                                self.ghost_dir[g_idx] = (dx, dy)
                                gx, gy = gx + dx, gy + dy
                        else:
                            legal = self._legal_neighbours(gx, gy)
                            dx, dy = self.rng.choice([(lx - gx, ly - gy) for lx, ly in legal])
                            self.ghost_dir[g_idx] = (dx, dy)
                            gx, gy = gx + dx, gy + dy

                self.ghost_pos[g_idx] = (gx, gy)

            # CHECK 2: Final collision check after all ghosts move
            if self.pac_pos in self.ghost_pos:
                reward -= 50
                terminated = True

        return self._render_board(), reward, terminated, False, {}

    # ──────────────────────── tile builder ─────────────────────
    def _build_tiles(self):
        """Build 12×12 pixel‑art tiles: 0 empty, 1 wall, 2 pac, 3 red ghost, 4 pellet, 5 blue ghost."""
        s = PIXELS_PER_CELL
        cx = (s - 1) / 2
        yy, xx = np.mgrid[0:s, 0:s]
        tiles = np.zeros((6, s, s, 3), dtype=np.uint8)

        # wall
        tiles[1, :, :, :] = [80, 80, 80]

        # pac‑man
        circle = (xx - cx) ** 2 + (yy - cx) ** 2 <= (s * 0.48) ** 2
        mouth  = np.abs(np.arctan2(yy - cx, xx - cx)) < np.pi / 5
        tiles[2, circle & ~mouth] = [255, 255, 0]

        # helper to build ghost (red then blue)
        def ghost_tile(rgb):
            g = np.zeros((s, s), bool)
            g |= (yy - 4) ** 2 + (xx - cx) ** 2 <= 25  # round head
            g |= yy >= 4                               # body
            for col in range(0, s, 4):                 # legs
                g[s - 1, col + 2:col + 4] = False
            tile = np.zeros((s, s, 3), np.uint8)
            tile[g] = rgb
            for ex in (int(s * 0.28), int(s * 0.58)):
                ey = int(s * 0.33)
                tile[ey:ey + 3, ex:ex + 2] = [255, 255, 255]  # whites
                tile[ey + 1, ex + 1] = [0, 0, 0]              # pupil
            return tile

        tiles[3] = ghost_tile([255, 0, 0])     # red
        tiles[5] = ghost_tile([0, 128, 255])   # blue

        # pellet
        p0 = int(cx) - 1
        tiles[4, p0:p0 + 2, p0:p0 + 2] = [0, 255, 0]

        self.tiles = tiles

    # ──────────────────────── rendering ────────────────────────
    def _render_board(self) -> np.ndarray:
        board = np.zeros((self.h, self.w), np.uint8)
        board[self.floor == 1] = 1
        for (x, y) in self.pellets: board[x, y] = 4
        for idx, (gx, gy) in enumerate(self.ghost_pos):
            board[gx, gy] = 3 if idx == 0 else 5
        px, py = self.pac_pos
        board[px, py] = 2

        img = np.zeros((self.img_h, self.img_w, 3), np.uint8)
        s = PIXELS_PER_CELL
        for i in range(self.h):
            for j in range(self.w):
                img[i*s:(i+1)*s, j*s:(j+1)*s] = self.tiles[board[i, j]]
        return img

    def render(self, mode="human"):
        if mode == "rgb_array":
            return self._render_board()
        plt.imshow(self._render_board()); plt.axis("off"); plt.show()

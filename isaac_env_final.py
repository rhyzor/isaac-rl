import time
import re
import subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import mss
import numpy as np
from gymnasium import spaces
from pynput.keyboard import Controller, Key


@dataclass
class GameState:
    player_health: int
    player_max_health: int
    player_position: Tuple[float, float]
    player_velocity: Tuple[float, float]
    player_direction: str
    current_room_clear: bool
    enemies_in_room: List[Tuple[int, int]]
    enemy_projectiles: List[Tuple[int, int]]
    pickups_nearby: Dict[str, List[Tuple[int, int]]]
    doors_status: Dict[str, bool]
    bomb_count: int
    key_count: int
    coin_count: int
    tear_damage: float
    tear_range: float
    tear_delay: float
    shot_speed: float
    movement_speed: float
    room_cleared_advanced: bool
    player_dead: bool
    room_changed: bool
    new_room_unexplored: bool
    floor_changed: bool
    boss_killed: bool
    room_cleared: bool
    enemies_killed_this_frame: int
    item_picked_up: bool
    item_type: Optional[str]
    chest_opened: bool
    door_opened: bool
    secret_room_found: bool
    health_lost: int
    projectiles_dodged: int
    player_moving: bool


@dataclass
class SessionStats:
    started_at: float = field(default_factory=time.monotonic)
    rooms_cleared: int = 0
    repeated_same_room_actions: int = 0
    repeated_actions: int = 0
    room_changes: int = 0
    enemies_killed: int = 0
    total_room_time_s: float = 0.0
    room_clear_time: float = 0.0
    exploration_rate: float = 0.0
    kills_per_minute: float = 0.0
    avg_room_clear_time: float = 0.0

    def update(self, game_state: GameState) -> None:
        elapsed_min = max(1e-3, (time.monotonic() - self.started_at) / 60.0)
        self.enemies_killed += game_state.enemies_killed_this_frame
        self.kills_per_minute = self.enemies_killed / elapsed_min
        if game_state.room_cleared:
            self.rooms_cleared += 1
        if game_state.room_changed:
            self.room_changes += 1
        self.exploration_rate = self.room_changes / elapsed_min
        self.avg_room_clear_time = self.total_room_time_s / max(1, self.rooms_cleared)

    @property
    def rooms_cleared_per_minute(self) -> float:
        elapsed_min = max(1e-3, (time.monotonic() - self.started_at) / 60.0)
        return self.rooms_cleared / elapsed_min

    @property
    def average_clear_speed(self) -> float:
        return max(0.1, self.rooms_cleared_per_minute)


class GameMemory:
    def __init__(self):
        self.visited_rooms: List[str] = []
        self.secret_rooms_found: List[str] = []
        self.item_positions: Dict[str, List[Tuple[int, int]]] = {}
        self.boss_patterns: Dict[str, Dict[str, float]] = {}
        self.danger_zone_history: List[Tuple[int, int]] = []
        self.current_floor_layout: Dict[str, Dict[str, bool]] = {}

    def update_memory(self, game_state: GameState) -> None:
        room_key = str(game_state.player_position)
        if room_key not in self.visited_rooms:
            self.visited_rooms.append(room_key)
        if game_state.secret_room_found and room_key not in self.secret_rooms_found:
            self.secret_rooms_found.append(room_key)
        for item_type, points in game_state.pickups_nearby.items():
            if points:
                self.item_positions[item_type] = points
        self.danger_zone_history.extend(game_state.enemy_projectiles[:4])


class AntiFarmSystem:
    def __init__(self):
        self.room_entry_time: Dict[str, float] = {}

    def calculate_time_in_room_penalty(self, current_room: str, game_state: GameState) -> float:
        now = time.monotonic()
        if current_room not in self.room_entry_time:
            self.room_entry_time[current_room] = now
            return 0.0
        if game_state.room_cleared:
            time_in_room = now - self.room_entry_time[current_room]
            if time_in_room > 3.0:
                penalty = -0.5 * (time_in_room - 3.0)
                if not game_state.player_moving:
                    penalty *= 2
                if game_state.room_cleared and len(game_state.enemies_in_room) > 0:
                    penalty *= 10
                return penalty
        return 0.0

    def detect_farming_behavior(self, action_history: List[str], game_state: GameState) -> float:
        recent_actions = action_history[-60:]
        bomb_usage = recent_actions.count("use_bomb")
        if bomb_usage > 5 and not game_state.room_cleared_advanced:
            return -20.0
        shooting_actions = {"shoot_up", "shoot_down", "shoot_left", "shoot_right"}
        shooting_without_enemies = any(action in shooting_actions for action in recent_actions) and len(game_state.enemies_in_room) == 0
        if shooting_without_enemies:
            return -10.0
        return 0.0


def calculate_bounding_box_area(points: List[Tuple[float, float]]) -> float:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs or not ys:
        return 0.0
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


class StagnationDetector:
    def __init__(self):
        self.room_pickups: Dict[str, bool] = {}

    def analyze_movement_pattern(self, positions_last_30_seconds: List[Tuple[float, float]]) -> float:
        if len(positions_last_30_seconds) < 2:
            return 0.0
        total_displacement = 0.0
        path_points: List[Tuple[float, float]] = []
        for i in range(1, len(positions_last_30_seconds)):
            p0, p1 = positions_last_30_seconds[i - 1], positions_last_30_seconds[i]
            displacement = float(np.linalg.norm(np.array(p1) - np.array(p0)))
            if displacement > 10:
                total_displacement += displacement
                path_points.append(p1)
        if path_points:
            area = calculate_bounding_box_area(path_points)
            if area < 500 and total_displacement < 1000:
                return -30.0
        return 0.0

    def room_has_pickups(self, room_id: str) -> bool:
        return self.room_pickups.get(room_id, False)

    def detect_room_loitering(self, current_room: str, room_clear_time: Optional[float], current_time: float) -> float:
        if room_clear_time is None:
            return 0.0
        t = current_time - room_clear_time
        if t > 5.0:
            penalty = -10.0 * (t - 5.0)
            if not self.room_has_pickups(current_room):
                penalty *= 2
            return penalty
        return 0.0


class ProgressTracker:
    def __init__(self):
        self.room_entry_count: Dict[str, Dict[str, float]] = {}
        self.last_meaningful_action: float = time.monotonic()

    def is_making_progress(self, game_state: GameState, action: str) -> float:
        progressive_actions = {
            "killed_enemy",
            "found_secret",
            "opened_door",
            "picked_up_item",
            "entered_new_room",
            "damaged_boss",
            "used_item_effectively",
        }
        now = time.monotonic()
        if action in progressive_actions:
            self.last_meaningful_action = now
            return 50.0
        time_since_progress = now - self.last_meaningful_action
        if time_since_progress > 10:
            return -20.0 * (time_since_progress / 10.0)
        return 0.0

    def detect_room_reset_farming(self, room_id: str, action: str) -> float:
        now = time.monotonic()
        counter = self.room_entry_count.setdefault(room_id, {"count": 0.0, "last_entry": now})
        if action == "entered_new_room":
            counter["count"] += 1.0
            counter["last_entry"] = now
        if action == "exit_room" and counter["count"] > 3:
            if now - counter["last_entry"] < 10:
                return -100.0
        return 0.0


class AdaptiveRewardSystem:
    def __init__(self):
        self.baseline_metrics = {
            "avg_room_clear_time": 45.0,
            "avg_stagnation_time": 8.0,
            "avg_exploration_rate": 1.0,
            "avg_kill_rate": 5.0,
        }
        self.stagnation_penalty_multiplier = 1.0
        self.base_kill_reward = 50.0
        self.new_room_bonus = 50.0

    def adjust_rewards_based_on_performance(self, player_stats: SessionStats) -> float:
        bonus = 0.0
        if player_stats.avg_room_clear_time > self.baseline_metrics["avg_room_clear_time"] * 1.5:
            self.stagnation_penalty_multiplier = 2.0
            bonus -= 5.0
        if player_stats.kills_per_minute > self.baseline_metrics["avg_kill_rate"] * 2:
            self.base_kill_reward = max(10.0, self.base_kill_reward * 0.5)
            bonus -= 5.0
        if player_stats.exploration_rate < self.baseline_metrics["avg_exploration_rate"] * 0.5:
            self.new_room_bonus = 100.0
            bonus += 5.0
        return bonus


def calculate_progress_reward(game_state: GameState, session_stats: SessionStats) -> float:
    reward = 0.0
    if session_stats.rooms_cleared_per_minute > session_stats.average_clear_speed:
        reward += 15.0
    if session_stats.repeated_same_room_actions > 10:
        diminishing_factor = max(0.1, 1.0 - (session_stats.repeated_actions * 0.1))
        reward *= diminishing_factor
    if game_state.room_changed:
        reward += 50.0 if game_state.new_room_unexplored else 10.0
        session_stats.repeated_same_room_actions = 0
        session_stats.repeated_actions = 0
    return reward


def handle_boss_fight(boss_type: str, boss_phase: int) -> float:
    if boss_phase == 1:
        return 1.0
    if boss_phase == 2:
        return 1.5
    return 2.0


def evaluate_item_usage(item_type: str, game_situation: str) -> float:
    if game_situation == "danger" and item_type == "defensive":
        return 50.0
    if game_situation == "crowded_room" and item_type == "offensive":
        return 50.0
    return 0.0


def resource_management_reward(resources_before: Dict[str, int], resources_after: Dict[str, int], situation: str) -> float:
    spent_bombs = max(0, resources_before.get("bomb", 0) - resources_after.get("bomb", 0))
    if spent_bombs == 0:
        return 0.0
    if situation in {"secret_room", "boss", "blocked_path"}:
        return 20.0 * spent_bombs
    return -5.0 * spent_bombs


def detect_suspicious_patterns(action_history: List[str], position_history: List[Tuple[float, float]]) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []
    recent_actions = action_history[-100:]
    if recent_actions.count("shoot_up") + recent_actions.count("shoot_down") > 30 and len(set(position_history[-40:])) < 10:
        alerts.append({"type": "corner_camping", "severity": "high", "suggested_action": "teleport_aggressive_enemies"})
    if len(position_history) > 40 and calculate_bounding_box_area(position_history[-80:]) < 600:
        alerts.append({"type": "circular_farming", "severity": "medium", "suggested_action": "reduce_rewards"})
    exits = sum(1 for a in action_history[-50:] if a.startswith("move_"))
    if exits > 40:
        alerts.append({"type": "door_farming", "severity": "critical", "suggested_action": "block_door_interaction"})
    return alerts


class DynamicDifficulty:
    def __init__(self):
        self.stagnation_threshold = 30
        self.boredom_level = 0

    def adjust_difficulty(self, stagnation_frames: int, time_in_room: float, room_cleared: bool, respawn_farming_detected: bool) -> Dict[str, float]:
        changes = {"spawn_aggressive": 0.0, "spawn_secret_enemies": 0.0, "enemy_multiplier": 1.0}
        if stagnation_frames > self.stagnation_threshold:
            self.boredom_level += 1
            if self.boredom_level > 5:
                changes["spawn_aggressive"] = 1.0
        if time_in_room > 30 and room_cleared:
            changes["spawn_secret_enemies"] = 1.0
        if respawn_farming_detected:
            changes["enemy_multiplier"] = 1.5
        return changes


training_stages = [
    {"name": "stage1_movement", "focus": "исследование и движение", "enemies": False, "objective": "научиться двигаться и собирать предметы", "reward_focus": "движение и сбор"},
    {"name": "stage2_combat_basic", "focus": "бой с простыми врагами", "enemies": True, "bosses": False, "objective": "научиться атаковать и уклоняться", "reward_focus": "убийства и уклонение"},
    {"name": "stage3_resources", "focus": "управление ресурсами", "enemies": True, "secrets": True, "objective": "научиться находить секреты и использовать ресурсы", "reward_focus": "секреты и эффективность"},
    {"name": "stage4_bosses", "focus": "битвы с боссами", "bosses": True, "objective": "научиться побеждать боссов", "reward_focus": "паттерны боссов и выживание"},
    {"name": "stage5_full_game", "focus": "полная игра", "all_content": True, "objective": "комплексная стратегия", "reward_focus": "полный прогресс"},
]


metrics_template = {
    "survival_time": 0.0,
    "rooms_cleared": 0,
    "enemies_killed": 0,
    "bosses_killed": 0,
    "damage_taken": 0,
    "items_collected": 0,
    "secrets_found": 0,
    "accuracy": 0.0,
    "exploration_percentage": 0.0,
    "resource_efficiency": 0.0,
    "dodge_rate": 0.0,
    "kill_speed": 0.0,
}


class IsaacEnv(gym.Env):
    """Кастомное окружение для The Binding of Isaac с динамическим определением окна."""

    def __init__(
        self,
        window_w: int = 879,
        window_h: int = 489,
        manual_reset: bool = True,
        reset_wait_seconds: float = 2.0,
        window_refresh_interval_s: float = 0.5,
        hp_smoothing_window: int = 3,
        hp_delta_threshold: int = 5,
        max_room_time_s: float = 60.0,
    ):
        super().__init__()

        # Размеры окна (из настроек игры)
        self.window_w = window_w
        self.window_h = window_h

        # Настройки управления эпизодами
        self.manual_reset = manual_reset
        self.reset_wait_seconds = reset_wait_seconds

        # Производительность
        self.window_refresh_interval_s = window_refresh_interval_s
        self._last_window_update_ts = 0.0

        # Стабилизация оценки HP и reward
        self.hp_delta_threshold = hp_delta_threshold
        self.hp_history: Deque[int] = deque(maxlen=max(1, hp_smoothing_window))
        self.max_room_time_s = max_room_time_s

        # Находим ID окна Isaac
        self.window_id = self._find_isaac_window()
        if not self.window_id:
            raise RuntimeError("Окно Isaac не найдено. Запусти игру и повтори попытку.")
        print(f"✅ Найдено окно Isaac: {self.window_id}")

        # Получаем текущие координаты окна
        self._update_window_position(force=True)

        # Расширенное пространство действий.
        self.action_map = {
            0: "move_up",
            1: "move_down",
            2: "move_left",
            3: "move_right",
            4: "shoot_up",
            5: "shoot_down",
            6: "shoot_left",
            7: "shoot_right",
            8: "use_bomb",
            9: "use_bomb_special",
            10: "place_bomb_wall",
            11: "activate_item",
            12: "pickup_item",
            13: "open_chest",
            14: "wait",
            15: "move_diagonal_up_right",
            16: "move_diagonal_up_left",
            17: "move_diagonal_down_right",
            18: "move_diagonal_down_left",
        }
        self.action_space = spaces.Discrete(len(self.action_map))

        # Пространство наблюдений: сжатое изображение 84x84 в градациях серого
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8,
        )

        # Инициализация захвата экрана
        self.sct = mss.mss()
        self.keyboard = Controller()

        # Для отслеживания здоровья
        self.prev_hp = 0
        self.prev_state: Optional[GameState] = None
        self.action_history: Deque[str] = deque(maxlen=240)
        self.position_history: Deque[Tuple[float, float]] = deque(maxlen=900)
        self.room_id = "unknown"
        self.room_entry_time = time.monotonic()
        self.room_clear_time: Optional[float] = None
        self.session_stats = SessionStats()

        self.memory = GameMemory()
        self.anti_farm = AntiFarmSystem()
        self.stagnation_detector = StagnationDetector()
        self.progress_tracker = ProgressTracker()
        self.adaptive_rewards = AdaptiveRewardSystem()

        # Для отладки
        self.debug = False

    def _default_state(self) -> "GameState":
        return GameState(
            player_health=self.prev_hp,
            player_max_health=max(self.prev_hp, 1),
            player_position=(self.window_w / 2, self.window_h / 2),
            player_velocity=(0.0, 0.0),
            player_direction="up",
            current_room_clear=False,
            enemies_in_room=[],
            enemy_projectiles=[],
            pickups_nearby={},
            doors_status={"up": False, "down": False, "left": False, "right": False},
            bomb_count=0,
            key_count=0,
            coin_count=0,
            tear_damage=3.5,
            tear_range=6.5,
            tear_delay=10.0,
            shot_speed=1.0,
            movement_speed=1.0,
            room_cleared_advanced=False,
            player_dead=False,
            room_changed=False,
            new_room_unexplored=False,
            floor_changed=False,
            boss_killed=False,
            room_cleared=False,
            enemies_killed_this_frame=0,
            item_picked_up=False,
            item_type=None,
            chest_opened=False,
            door_opened=False,
            secret_room_found=False,
            health_lost=0,
            projectiles_dodged=0,
            player_moving=False,
        )

    def _find_isaac_window(self) -> Optional[str]:
        """Находит ID окна Isaac через xdotool."""
        commands = [
            ["xdotool", "search", "--name", "(?i)isaac|binding"],
            ["xdotool", "search", "--class", "isaac"],
        ]

        try:
            for cmd in commands:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.stdout:
                    return result.stdout.strip().split("\n")[0]
        except FileNotFoundError as exc:
            raise RuntimeError("xdotool не установлен. Установи его через apt.") from exc

        return None

    def _update_window_position(self, force: bool = False) -> None:
        """Обновляет координаты окна Isaac (с ограничением частоты вызовов)."""
        now = time.monotonic()
        if not force and (now - self._last_window_update_ts) < self.window_refresh_interval_s:
            return

        try:
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", self.window_id],
                capture_output=True,
                text=True,
                check=False,
            )
            x = y = 0
            for line in result.stdout.split("\n"):
                if "Position:" in line:
                    coords = re.findall(r"\d+", line)
                    if len(coords) >= 2:
                        x, y = int(coords[0]), int(coords[1])
            self.window_x = x
            self.window_y = y
            self.monitor = {
                "left": self.window_x,
                "top": self.window_y,
                "width": self.window_w,
                "height": self.window_h,
            }
            self._last_window_update_ts = now
            if self.debug:
                print(f"Окно Isaac на позиции: ({self.window_x}, {self.window_y})")
        except Exception as exc:
            raise RuntimeError(f"Ошибка при получении позиции окна: {exc}") from exc

    def _capture(self):
        """Захватывает область с игрой."""
        self._update_window_position()
        img = self.sct.grab(self.monitor)
        return np.array(img)

    def _get_observation(self):
        """Преобразует скриншот в наблюдение для нейросети."""
        img = self._capture()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (84, 84))
        img = np.expand_dims(img, axis=-1)

        if self.debug:
            cv2.imwrite("debug_obs.png", img.squeeze())

        return img

    def _get_hp_raw(self) -> int:
        """Определяет сырое здоровье по области сердец (кол-во красных пикселей)."""
        hearts_dx = 74
        hearts_dy = 3
        hearts_w = 148
        hearts_h = 48

        img = self._capture()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hearts = img[hearts_dy : hearts_dy + hearts_h, hearts_dx : hearts_dx + hearts_w]

        if hearts.size == 0:
            if self.debug:
                print("⚠️ Область сердец пуста! Проверь координаты.")
            return 0

        red_lower = np.array([0, 0, 150])
        red_upper = np.array([100, 100, 255])
        mask = cv2.inRange(hearts, red_lower, red_upper)

        if self.debug:
            cv2.imwrite("debug_hearts.png", hearts)
            cv2.imwrite("debug_mask.png", mask)
            hearts_colored = hearts.copy()
            hearts_colored[mask > 0] = [0, 255, 0]
            cv2.imwrite("debug_hearts_colored.png", hearts_colored)

        return int(cv2.countNonZero(mask))

    def _get_hp(self) -> int:
        """Возвращает сглаженную оценку HP, устойчивую к шуму кадра."""
        hp_raw = self._get_hp_raw()
        self.hp_history.append(hp_raw)
        return int(np.median(np.array(self.hp_history, dtype=np.int32)))

    def detect_enemies(self, screen: np.ndarray) -> List[Tuple[int, int]]:
        """Быстрый детектор врагов по темным blob-объектам (черновой baseline)."""
        gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
        mask = cv2.inRange(gray, 10, 70)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enemies: List[Tuple[int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 40 <= area <= 1200:
                x, y, w, h = cv2.boundingRect(contour)
                enemies.append((x + w // 2, y + h // 2))
        return enemies[:32]

    def detect_secret_room_indicators(self, screen: np.ndarray) -> float:
        """Ищет трещины на стенах через edge-density в пограничной зоне комнаты."""
        bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        h, w = bgr.shape[:2]
        border = np.zeros((h, w), dtype=np.uint8)
        thickness = max(8, int(min(h, w) * 0.05))
        border[:thickness, :] = 255
        border[-thickness:, :] = 255
        border[:, :thickness] = 255
        border[:, -thickness:] = 255
        edges = cv2.Canny(bgr, 60, 120)
        edge_density = float(np.count_nonzero(cv2.bitwise_and(edges, border))) / float(np.count_nonzero(border) + 1)
        return float(np.clip(edge_density * 4.0, 0.0, 1.0))

    def detect_pickups(self, screen: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """Находит предметы по цветовым маскам (монеты/бомбы/сердца/ключи/сундуки)."""
        bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        def centers(mask: np.ndarray, area_min: int = 10, area_max: int = 500) -> List[Tuple[int, int]]:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts: List[Tuple[int, int]] = []
            for c in cnts:
                a = cv2.contourArea(c)
                if area_min <= a <= area_max:
                    x, y, w, h = cv2.boundingRect(c)
                    pts.append((x + w // 2, y + h // 2))
            return pts[:32]

        masks = {
            "coins": cv2.inRange(hsv, np.array([15, 90, 130]), np.array([35, 255, 255])),
            "bombs": cv2.inRange(hsv, np.array([0, 0, 25]), np.array([180, 60, 90])),
            "keys": cv2.inRange(hsv, np.array([20, 30, 170]), np.array([40, 140, 255])),
            "hearts": cv2.inRange(hsv, np.array([0, 120, 120]), np.array([12, 255, 255])),
            "items": cv2.inRange(hsv, np.array([80, 40, 110]), np.array([130, 255, 255])),
            "chests": cv2.inRange(hsv, np.array([8, 90, 90]), np.array([20, 255, 200])),
        }
        return {key: centers(mask) for key, mask in masks.items()}

    def detect_threats(self, screen: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """Детектор угроз: вражеские пули/враги/ловушки (эвристика)."""
        bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        bullets = cv2.inRange(hsv, np.array([0, 100, 170]), np.array([179, 255, 255]))
        enemy_positions = self.detect_enemies(screen)
        bullet_points = np.argwhere(bullets > 0)
        bullet_centers = [(int(p[1]), int(p[0])) for p in bullet_points[:: max(1, len(bullet_points) // 48 + 1)]]
        return {
            "enemy_bullets": bullet_centers,
            "enemy_positions": enemy_positions,
            "spikes": [],
            "fire_places": [],
        }

    def analyze_room_clear_status(self, enemies: List[Tuple[int, int]], doors_status: Dict[str, bool]) -> float:
        """Оценка прогресса зачистки комнаты."""
        if not enemies:
            return 1.0 if any(doors_status.values()) else 0.9
        return max(0.0, 1.0 - min(1.0, len(enemies) / 12.0))

    def _read_game_state(self, action_name: str) -> "GameState":
        frame = self._capture()
        hp = self._get_hp()
        enemies = self.detect_enemies(frame)
        pickups = self.detect_pickups(frame)
        threats = self.detect_threats(frame)
        secret_prob = self.detect_secret_room_indicators(frame)
        player_pos = (self.window_w / 2, self.window_h / 2)

        doors_status = {
            "up": True,
            "down": True,
            "left": True,
            "right": True,
        }
        clear_pct = self.analyze_room_clear_status(enemies, doors_status)
        room_cleared = clear_pct >= 0.98

        health_lost = max(0, self.prev_hp - hp)
        room_changed = action_name.startswith("move") and len(self.position_history) > 10 and np.random.rand() < 0.01
        item_type = "coin" if pickups.get("coins") else None

        return GameState(
            player_health=hp,
            player_max_health=max(hp, 1),
            player_position=player_pos,
            player_velocity=(0.0, 0.0),
            player_direction=action_name,
            current_room_clear=room_cleared,
            enemies_in_room=enemies,
            enemy_projectiles=threats["enemy_bullets"],
            pickups_nearby=pickups,
            doors_status=doors_status,
            bomb_count=0,
            key_count=0,
            coin_count=len(pickups.get("coins", [])),
            tear_damage=3.5,
            tear_range=6.5,
            tear_delay=10.0,
            shot_speed=1.0,
            movement_speed=1.0,
            room_cleared_advanced=room_cleared and len(threats["enemy_bullets"]) == 0,
            player_dead=hp < 20,
            room_changed=room_changed,
            new_room_unexplored=room_changed,
            floor_changed=False,
            boss_killed=False,
            room_cleared=room_cleared,
            enemies_killed_this_frame=1 if self.prev_state and len(enemies) < len(self.prev_state.enemies_in_room) else 0,
            item_picked_up=action_name == "pickup_item" and item_type is not None,
            item_type=item_type,
            chest_opened=action_name == "open_chest" and len(pickups.get("chests", [])) > 0,
            door_opened=action_name.startswith("move") and room_changed,
            secret_room_found=secret_prob > 0.75 and action_name == "place_bomb_wall",
            health_lost=health_lost,
            projectiles_dodged=max(0, 2 - len(threats["enemy_bullets"]) // 20),
            player_moving=action_name.startswith("move"),
        )

    def _take_action(self, action):
        """Выполняет действие."""
        for key in ["w", "a", "s", "d", Key.up, Key.down, Key.left, Key.right]:
            try:
                self.keyboard.release(key)
            except ValueError:
                continue
            except OSError:
                continue

        action_name = self.action_map.get(action, "wait")

        if action_name == "move_up":
            self.keyboard.press("w")
        elif action_name == "move_down":
            self.keyboard.press("s")
        elif action_name == "move_left":
            self.keyboard.press("a")
        elif action_name == "move_right":
            self.keyboard.press("d")
        elif action_name == "shoot_up":
            self.keyboard.press(Key.up)
        elif action_name == "shoot_down":
            self.keyboard.press(Key.down)
        elif action_name == "shoot_left":
            self.keyboard.press(Key.left)
        elif action_name == "shoot_right":
            self.keyboard.press(Key.right)
        elif action_name in {"use_bomb", "use_bomb_special", "place_bomb_wall"}:
            self.keyboard.press("e")
            self.keyboard.release("e")
        elif action_name == "activate_item":
            self.keyboard.press("q")
            self.keyboard.release("q")
        elif action_name in {"pickup_item", "open_chest"}:
            self.keyboard.press(Key.space)
            self.keyboard.release(Key.space)
        elif action_name == "move_diagonal_up_right":
            self.keyboard.press("w")
            self.keyboard.press("d")
        elif action_name == "move_diagonal_up_left":
            self.keyboard.press("w")
            self.keyboard.press("a")
        elif action_name == "move_diagonal_down_right":
            self.keyboard.press("s")
            self.keyboard.press("d")
        elif action_name == "move_diagonal_down_left":
            self.keyboard.press("s")
            self.keyboard.press("a")

        return action_name

    def reset(self, seed=None, options=None):
        """Сброс окружения."""
        super().reset(seed=seed)

        if self.manual_reset:
            print("\n🔄 Перезапусти игру вручную и нажми Enter...")
            input("⏎ Нажми Enter, когда будешь готов...")
        else:
            if self.debug:
                print("🔁 Автоматический reset без ожидания ввода пользователя.")

        time.sleep(self.reset_wait_seconds)
        self.hp_history.clear()
        self.prev_hp = self._get_hp()
        self.prev_state = self._default_state()
        self.action_history.clear()
        self.position_history.clear()
        self.session_stats = SessionStats()
        self.room_entry_time = time.monotonic()
        self.room_clear_time = None

        if self.debug:
            print(f"❤️ Начальное здоровье (сглаж.): {self.prev_hp}")

        return self._get_observation(), {}

    def step(self, action):
        """Шаг окружения."""
        action_name = self._take_action(action)
        time.sleep(0.1)

        obs = self._get_observation()
        game_state = self._read_game_state(action_name)
        reward = self.calculate_reward(game_state, self.prev_state)

        self.prev_hp = game_state.player_health
        self.prev_state = game_state
        self.action_history.append(action_name)
        self.position_history.append(game_state.player_position)
        self.memory.update_memory(game_state)
        self.session_stats.update(game_state)

        terminated = game_state.player_dead
        truncated = (time.monotonic() - self.room_entry_time) > self.max_room_time_s

        return obs, reward, terminated, truncated, {"room_id": self.room_id}

    def calculate_reward(self, game_state: "GameState", previous_state: Optional["GameState"]) -> float:
        reward = 0.01

        if game_state.player_moving:
            reward += 0.1
        if not game_state.player_moving and len(self.action_history) > 30:
            reward -= 0.5

        reward += 50.0 * game_state.enemies_killed_this_frame
        if game_state.boss_killed:
            reward += 500
        if game_state.room_cleared and (not previous_state or not previous_state.room_cleared):
            reward += 100

        if game_state.item_picked_up:
            reward += {
                "health": 30,
                "key": 40,
                "bomb": 40,
                "coin": 20,
                "treasure": 150,
            }.get(game_state.item_type or "", 10)

        if game_state.chest_opened:
            reward += 80
        if game_state.door_opened:
            reward += 30
        if game_state.secret_room_found:
            reward += 200
        if game_state.health_lost > 0:
            reward -= 100 * game_state.health_lost
        if game_state.player_dead:
            reward -= 1000
        reward += 5 * game_state.projectiles_dodged
        if game_state.floor_changed:
            reward += 300

        reward += self.anti_farm.calculate_time_in_room_penalty(self.room_id, game_state)
        reward += self.anti_farm.detect_farming_behavior(list(self.action_history), game_state)
        reward += self.stagnation_detector.analyze_movement_pattern(list(self.position_history))
        reward += self.stagnation_detector.detect_room_loitering(self.room_id, self.room_clear_time, time.monotonic())
        reward += self.progress_tracker.is_making_progress(game_state, self._infer_progress_action(game_state))
        reward += self.progress_tracker.detect_room_reset_farming(self.room_id, self._infer_progress_action(game_state))
        reward += calculate_progress_reward(game_state, self.session_stats)
        reward += self.adaptive_rewards.adjust_rewards_based_on_performance(self.session_stats)

        return float(reward)

    def _infer_progress_action(self, game_state: "GameState") -> str:
        if game_state.enemies_killed_this_frame > 0:
            return "killed_enemy"
        if game_state.secret_room_found:
            return "found_secret"
        if game_state.door_opened:
            return "opened_door"
        if game_state.item_picked_up:
            return "picked_up_item"
        if game_state.room_changed and game_state.new_room_unexplored:
            return "entered_new_room"
        if game_state.boss_killed:
            return "damaged_boss"
        return "noop"

    def render(self):
        pass

    def close(self):
        self.sct.close()


if __name__ == "__main__":
    print("🎮 Тестирование Isaac Environment (адаптивное окно)")
    print("=" * 50)

    env = IsaacEnv(manual_reset=True)
    env.debug = True

    obs, _ = env.reset()

    action_names = env.action_map

    print("\n📝 Будет выполнено 30 случайных действий")
    print("Наблюдай за персонажем и смотри на вывод\n")

    try:
        for i in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(action)
            print(
                f"Шаг {i:2d} | {action_names.get(action, 'unknown')} | Награда: {reward:6.2f} | "
                f"Здоровье: {env.prev_hp:4d}"
            )
            if terminated:
                print("\n💀 ПЕРСОНАЖ УМЕР! Тест прерван.")
                break
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n\n⏹️ Тест прерван пользователем")
    finally:
        env.close()

    print("\n✅ Тест завершён")

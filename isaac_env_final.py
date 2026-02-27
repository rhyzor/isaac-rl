import time
import re
import subprocess
from collections import deque
from typing import Deque, Optional

import cv2
import gymnasium as gym
import mss
import numpy as np
from gymnasium import spaces
from pynput.keyboard import Controller, Key


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

        # Находим ID окна Isaac
        self.window_id = self._find_isaac_window()
        if not self.window_id:
            raise RuntimeError("Окно Isaac не найдено. Запусти игру и повтори попытку.")
        print(f"✅ Найдено окно Isaac: {self.window_id}")

        # Получаем текущие координаты окна
        self._update_window_position(force=True)

        # Пространство действий: 0 - ничего, 1-4 движение, 5-8 стрельба
        self.action_space = spaces.Discrete(9)

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

        # Для отладки
        self.debug = False

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

    def _take_action(self, action):
        """Выполняет действие."""
        for key in ["w", "a", "s", "d", Key.up, Key.down, Key.left, Key.right]:
            try:
                self.keyboard.release(key)
            except ValueError:
                continue
            except OSError:
                continue

        if action == 1:
            self.keyboard.press("w")
        elif action == 2:
            self.keyboard.press("s")
        elif action == 3:
            self.keyboard.press("a")
        elif action == 4:
            self.keyboard.press("d")
        elif action == 5:
            self.keyboard.press(Key.up)
        elif action == 6:
            self.keyboard.press(Key.down)
        elif action == 7:
            self.keyboard.press(Key.left)
        elif action == 8:
            self.keyboard.press(Key.right)

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

        if self.debug:
            print(f"❤️ Начальное здоровье (сглаж.): {self.prev_hp}")

        return self._get_observation(), {}

    def step(self, action):
        """Шаг окружения."""
        self._take_action(action)
        time.sleep(0.1)

        obs = self._get_observation()
        current_hp = self._get_hp()

        reward = 0.1
        hp_delta = current_hp - self.prev_hp

        # Игнорируем мелкие колебания детектора (шум)
        if abs(hp_delta) <= self.hp_delta_threshold:
            hp_delta = 0

        if hp_delta < 0:
            damage = abs(hp_delta)
            reward = -0.5 * damage
            if self.debug:
                print(f"💔 УРОН! Было: {self.prev_hp}, стало: {current_hp}, штраф: {reward:.2f}")
        elif hp_delta > 0:
            heal = hp_delta
            reward = 0.2 * heal
            if self.debug:
                print(f"💚 ЛЕЧЕНИЕ! +{heal}, бонус: {reward:.2f}")

        self.prev_hp = current_hp

        terminated = current_hp < 20

        return obs, reward, terminated, False, {}

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

    action_names = [
        "🕐 НИЧЕГО",
        "⬆️ ДВИЖЕНИЕ вверх",
        "⬇️ ДВИЖЕНИЕ вниз",
        "⬅️ ДВИЖЕНИЕ влево",
        "➡️ ДВИЖЕНИЕ вправо",
        "⬆️🔫 СТРЕЛЬБА вверх",
        "⬇️🔫 СТРЕЛЬБА вниз",
        "⬅️🔫 СТРЕЛЬБА влево",
        "➡️🔫 СТРЕЛЬБА вправо",
    ]

    print("\n📝 Будет выполнено 30 случайных действий")
    print("Наблюдай за персонажем и смотри на вывод\n")

    try:
        for i in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(action)
            print(
                f"Шаг {i:2d} | {action_names[action]} | Награда: {reward:6.2f} | "
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

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mss
import cv2
from pynput.keyboard import Controller, Key
import time
import subprocess
import re

class IsaacEnv(gym.Env):
    """Кастомное окружение для The Binding of Isaac с динамическим определением окна"""
    
    def __init__(self):
        super().__init__()
        
        # Размеры окна (постоянные, из настроек игры)
        self.window_w = 879
        self.window_h = 489
        
        # Находим ID окна Isaac
        self.window_id = self._find_isaac_window()
        if not self.window_id:
            print("❌ Окно Isaac не найдено! Запусти игру.")
            exit(1)
        else:
            print(f"✅ Найдено окно Isaac: {self.window_id}")
        
        # Получаем текущие координаты окна
        self._update_window_position()
        
        # Пространство действий: 0 - ничего, 1-4 движение, 5-8 стрельба
        self.action_space = spaces.Discrete(9)
        
        # Пространство наблюдений: сжатое изображение 84x84 в градациях серого
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 1),
            dtype=np.uint8
        )
        
        # Инициализация захвата экрана
        self.sct = mss.mss()
        self.keyboard = Controller()
        
        # Для отслеживания здоровья
        self.prev_hp = 0
        
        # Для отладки
        self.debug = False
    
    def _find_isaac_window(self):
        """Находит ID окна Isaac через xdotool"""
        try:
            # Поиск по имени (регистронезависимо)
            result = subprocess.run(
                ['xdotool', 'search', '--name', '(?i)isaac|binding'],
                capture_output=True, text=True
            )
            if result.stdout:
                return result.stdout.strip().split('\n')[0]
            
            # Поиск по классу
            result = subprocess.run(
                ['xdotool', 'search', '--class', 'isaac'],
                capture_output=True, text=True
            )
            if result.stdout:
                return result.stdout.strip().split('\n')[0]
        except Exception as e:
            print(f"Ошибка при поиске окна: {e}")
        return None
    
    def _update_window_position(self):
        """Обновляет координаты окна Isaac"""
        try:
            result = subprocess.run(
                ['xdotool', 'getwindowgeometry', self.window_id],
                capture_output=True, text=True
            )
            x = y = 0
            for line in result.stdout.split('\n'):
                if 'Position:' in line:
                    coords = re.findall(r'\d+', line)
                    if len(coords) >= 2:
                        x, y = int(coords[0]), int(coords[1])
            self.window_x = x
            self.window_y = y
            self.monitor = {
                'left': self.window_x,
                'top': self.window_y,
                'width': self.window_w,
                'height': self.window_h
            }
            if self.debug:
                print(f"Окно Isaac на позиции: ({self.window_x}, {self.window_y})")
        except Exception as e:
            print(f"Ошибка при получении позиции окна: {e}")
    
    def _capture(self):
        """Захватывает область с игрой (с предварительным обновлением позиции)"""
        self._update_window_position()
        img = self.sct.grab(self.monitor)
        return np.array(img)
    
    def _get_observation(self):
        """Преобразует скриншот в наблюдение для нейросети"""
        img = self._capture()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (84, 84))
        img = np.expand_dims(img, axis=-1)
        
        if self.debug:
            cv2.imwrite('debug_obs.png', img.squeeze())
        
        return img
    
    def _get_hp(self):
        """
        Определяет здоровье по области сердец.
        Координаты задаются ОТНОСИТЕЛЬНО окна игры (смещения от левого верхнего угла окна).
        Эти значения нужно подобрать экспериментально.
        """
        # --- ОТНОСИТЕЛЬНЫЕ КООРДИНАТЫ ОБЛАСТИ СЕРДЕЦ (подбери их!) ---
        hearts_dx = 74    # смещение по X от левого края окна
        hearts_dy = 3     # смещение по Y от верхнего края окна
        hearts_w = 148
        hearts_h = 48
        # -------------------------------------------------------------
        
        img = self._capture()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Обрезаем область сердец (используем относительные координаты)
        hearts = img[hearts_dy:hearts_dy+hearts_h, hearts_dx:hearts_dx+hearts_w]
        
        if hearts.size == 0:
            print("⚠️ Область сердец пуста! Проверь координаты.")
            return 0
        
        # Ищем красные пиксели (сердца)
        red_lower = np.array([0, 0, 150])
        red_upper = np.array([100, 100, 255])
        mask = cv2.inRange(hearts, red_lower, red_upper)
        
        if self.debug:
            cv2.imwrite('debug_hearts.png', hearts)
            cv2.imwrite('debug_mask.png', mask)
            hearts_colored = hearts.copy()
            hearts_colored[mask > 0] = [0, 255, 0]
            cv2.imwrite('debug_hearts_colored.png', hearts_colored)
        
        return cv2.countNonZero(mask)
    
    def _take_action(self, action):
        """Выполняет действие"""
        # Отпускаем все клавиши
        for key in ['w', 'a', 's', 'd', Key.up, Key.down, Key.left, Key.right]:
            try:
                self.keyboard.release(key)
            except:
                pass
        
        # Движение
        if action == 1:
            self.keyboard.press('w')
        elif action == 2:
            self.keyboard.press('s')
        elif action == 3:
            self.keyboard.press('a')
        elif action == 4:
            self.keyboard.press('d')
        # Стрельба
        elif action == 5:
            self.keyboard.press(Key.up)
        elif action == 6:
            self.keyboard.press(Key.down)
        elif action == 7:
            self.keyboard.press(Key.left)
        elif action == 8:
            self.keyboard.press(Key.right)
        # action 0 - ничего не делаем
    
    def reset(self, seed=None):
        """Сброс окружения (новая игра)"""
        super().reset(seed=seed)
        
        print("\n🔄 Перезапусти игру вручную и нажми Enter...")
        input("⏎ Нажми Enter, когда будешь готов...")
        
        time.sleep(2)
        self.prev_hp = self._get_hp()
        print(f"❤️ Начальное здоровье: {self.prev_hp}")
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Шаг окружения"""
        self._take_action(action)
        time.sleep(0.1)
        
        obs = self._get_observation()
        current_hp = self._get_hp()
        
        reward = 0.1  # базовая награда за шаг
        
        if current_hp < self.prev_hp:
            damage = self.prev_hp - current_hp
            reward = -10 * damage
            print(f"💔 УРОН! Было: {self.prev_hp}, стало: {current_hp}, штраф: {reward}")
        elif current_hp > self.prev_hp:
            heal = current_hp - self.prev_hp
            reward = 5 * heal
            print(f"💚 ЛЕЧЕНИЕ! +{heal}")
        
        self.prev_hp = current_hp
        
        # Проверка смерти (порог подбери экспериментально)
        terminated = current_hp < 20
        
        return obs, reward, terminated, False, {}
    
    def render(self):
        pass
    
    def close(self):
        self.sct.close()

# Тест окружения
if __name__ == "__main__":
    print("🎮 Тестирование Isaac Environment (адаптивное окно)")
    print("=" * 50)
    
    env = IsaacEnv()
    env.debug = True  # включим отладку для калибровки
    
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
        "➡️🔫 СТРЕЛЬБА вправо"
    ]
    
    print("\n📝 Будет выполнено 30 случайных действий")
    print("Наблюдай за персонажем и смотри на вывод\n")
    
    try:
        for i in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, _, _ = env.step(action)
            print(f"Шаг {i:2d} | {action_names[action]} | Награда: {reward:6.2f} | Здоровье: {env.prev_hp:4d}")
            if terminated:
                print("\n💀 ПЕРСОНАЖ УМЕР! Тест прерван.")
                break
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n\n⏹️ Тест прерван пользователем")
    finally:
        env.close()
    
    print("\n✅ Тест завершён")
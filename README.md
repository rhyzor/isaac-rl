# 🎮 Training a Neural Network to Play The Binding of Isaac

This project is a reinforcement learning environment for **The Binding of Isaac: Repentance+** running on Linux. It connects Python with the game via screen capture and keyboard emulation, allowing you to train an AI agent using algorithms like PPO.

![Isaac Environment Demo](docs/demo.gif) *(Add gif later)*

---

# ❗ This projet now in development phase 



## ✨ Features

- **Real-time screen capture** – uses `mss` for fast, efficient grabbing of the game window.
- **Keyboard control** – emulates key presses with `pynput` (or `xdotool` as fallback).
- **Dynamic window tracking** – locates the Isaac window via `xdotool` and adapts to its position automatically.
- **Health detection** – isolates the red heart containers using OpenCV and calculates a pixel-based health value.
- **Custom Gymnasium environment** – implements the `gym.Env` interface, ready for use with any RL library (Stable-Baselines3, Ray RLlib, etc.).
- **Calibration tools** – scripts to fine‑tune the heart detection region for your screen resolution.

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/rhyzor/isaac-rl.git
cd isaac-rl
```

### 2. Set up a Python virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install torch torchvision gymnasium opencv-python mss pynput numpy matplotlib stable-baselines3
```

### 4. Install system tools (for window tracking)
```bash
sudo apt update
sudo apt install xdotool
```

---

## 🚀 Usage

### 🔍 Calibrate the heart detection region
Before training, you must tell the program exactly where the health hearts appear inside the game window.
1. Launch **The Binding of Isaac** and start a run.
2. Run the calibration script:
   ```bash
   python calibrate_hearts_dynamic.py
   ```
3. Follow the on‑screen instructions. You will see the current detection rectangle and can adjust its position and size.
4. Once the rectangle perfectly encloses the hearts, note the final `dx, dy, w, h` values and update them in `isaac_env_final.py` inside the `_get_hp()` method.

### ▶️ Test the environment
After calibration, run a short test with random actions:
```bash
python isaac_env_final.py
```
You should see the character move and shoot randomly, while the terminal prints health values and rewards.

### 🧠 Train an agent
*(Placeholder – you will implement actual RL training later)*
Example with Stable‑Baselines3:
```python
from stable_baselines3 import PPO
from isaac_env_final import IsaacEnv

env = IsaacEnv()
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("isaac_ppo")
```

---

## 📁 Project Structure

```
isaac-rl/
├── isaac_env_final.py          # Main Gym environment with dynamic window tracking
├── calibrate_hearts_dynamic.py # Interactive tool to fine‑tune heart ROI
├── calibrate_hp.py             # Simple script to monitor health values in real time
├── train_simple.py             # (Optional) rudimentary training example
├── README.md
└── docs/                        # Images / gifs for documentation
```

---

## ⚙️ Configuration

### Window size
The environment assumes the game runs at **879×489** (you can change `self.window_w`, `self.window_h` in `__init__` if your game uses a different resolution).

### Heart ROI (region of interest)
Inside `_get_hp()` you will find four variables:
```python
hearts_dx = 74    # horizontal offset from the left edge of the window
hearts_dy = 3     # vertical offset from the top edge of the window
hearts_w = 148    # width of the heart region
hearts_h = 48     # height of the heart region
```
These **must** be set according to your screen. Use the calibration script to find the correct values.

### Action space
The environment defines 9 discrete actions:
- `0` : do nothing
- `1` : move up
- `2` : move down
- `3` : move left
- `4` : move right
- `5` : shoot up
- `6` : shoot down
- `7` : shoot left
- `8` : shoot right

You can easily extend it to include bombs, item usage, etc.

---

## 🧪 Results

*(This section will be updated as training progresses.)*

Preliminary tests show that the environment correctly captures screen, simulates key presses, and detects health changes. A simple random agent survives only a few seconds – perfect starting point for reinforcement learning.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [OpenAI Gymnasium](https://gymnasium.farama.org/) for the RL interface.
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for state‑of‑the‑art RL algorithms.
- [mss](https://github.com/BoboTiG/python-mss) for ultra‑fast screen capture.
- [pynput](https://pynput.readthedocs.io/) for cross‑platform keyboard control.
- The Binding of Isaac community for keeping this awesome game alive.

---

**Happy training!** If you have any questions, feel free to reach out.

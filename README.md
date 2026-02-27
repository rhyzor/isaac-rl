# 🎮 Training a Neural Network to Play The Binding of Isaac

This project is a reinforcement learning environment for **The Binding of Isaac: Repentance+** on Linux. It connects Python with the game via screen capture and keyboard emulation, allowing you to train an agent with PPO and other RL algorithms.

## ❗ Status
Project is under active development.

## ✨ Features

- Real-time screen capture via `mss`
- Keyboard control via `pynput`
- Dynamic game window tracking using `xdotool`
- HP detection from heart UI region via OpenCV
- Gymnasium-compatible environment (`IsaacEnv`)
- Optional manual or non-interactive reset mode
- Smoothed HP signal for more stable rewards

## 📦 Installation

```bash
git clone https://github.com/rhyzor/isaac-rl.git
cd isaac-rl
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision gymnasium opencv-python mss pynput numpy matplotlib stable-baselines3
sudo apt update
sudo apt install xdotool
```

## 🚀 Usage

### 1) Calibrate / inspect HP detector

```bash
python calibrate_hearts_dynamic.py
# or
python calibrate_hp.py
```

Tune ROI constants in `isaac_env_final.py` inside `_get_hp_raw()`:

```python
hearts_dx = 74
hearts_dy = 3
hearts_w = 148
hearts_h = 48
```

### 2) Test environment

```bash
python isaac_env_final.py
```

### 3) Train baseline PPO

```bash
python train_simple.py
```

## 📁 Project Structure

```text
isaac-rl/
├── isaac_env_final.py          # Main Gym environment
├── calibrate_hearts_dynamic.py # HP/ROI calibration helper
├── calibrate_hp.py             # Simple HP monitor
├── train_simple.py             # Minimal PPO training example
├── README.md
├── LICENSE
└── docs/
```

## ⚙️ Key Environment Settings

`IsaacEnv` supports key runtime parameters:

- `manual_reset` — if `True`, waits for Enter during `reset()`
- `window_refresh_interval_s` — how often to refresh window geometry
- `hp_smoothing_window` — median smoothing window for HP signal
- `hp_delta_threshold` — ignores small HP detector fluctuations

## 📄 License

MIT — see [LICENSE](LICENSE).

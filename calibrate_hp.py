"""Простой монитор значения HP-детектора в реальном времени."""

import time

from isaac_env_final import IsaacEnv


def main() -> None:
    env = IsaacEnv(manual_reset=False)
    env.debug = True

    try:
        print("Мониторинг HP. Нажми Ctrl+C для выхода.")
        while True:
            hp = env._get_hp()
            print(f"HP(detector): {hp}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

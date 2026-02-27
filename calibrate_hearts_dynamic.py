"""Интерактивная заготовка калибровки области сердец.

Показывает текущее значение HP и напоминает, где менять ROI.
"""

import time

from isaac_env_final import IsaacEnv


def main() -> None:
    env = IsaacEnv(manual_reset=False)
    env.debug = True
    print("Текущий ROI для сердец задаётся в isaac_env_final.py -> _get_hp_raw().")
    print("Измени hearts_dx/hearts_dy/hearts_w/hearts_h и перезапусти скрипт.")

    try:
        while True:
            hp = env._get_hp()
            print(f"HP(detector, smoothed): {hp}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nКалибровка завершена.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

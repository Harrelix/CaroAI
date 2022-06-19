import numpy as np

if __name__ == "__main__":
    with open("training_data/game_state_data.npy", "wb") as f:
        np.save(f, np.empty((0, 13, 13, 9), dtype=bool))
    with open("training_data/pi_data.npy", "wb") as f:
        np.save(f, np.empty((0, 13, 13, 1), dtype=np.float32))
    with open("training_data/result_data.npy", "wb") as f:
        np.save(f, np.empty((0,), dtype=np.float32))

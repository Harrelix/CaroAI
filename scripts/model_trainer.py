import numpy as np
from load_and_save_model import load, save
import json
import re
import os
import pickle

if __name__ == "__main__":
    print("loading constants")
    with open("constants.jsonc", "r") as f:
        constants = json.loads(re.sub("//.*", "", f.read(), flags=re.MULTILINE))
    print("loading data")
    with open(constants["TRAINING_DATA_PATH"] + "game_state_data.npy", "rb") as f:
        game_state_data = np.load(f)
    with open(constants["TRAINING_DATA_PATH"] + "pi_data.npy", "rb") as f:
        pi_data = np.load(f)
    with open(constants["TRAINING_DATA_PATH"] + "result_data.npy", "rb") as f:
        result_data = np.load(f)
    print("LOADED DATA")
    if not (game_state_data.shape[0] == pi_data.shape[0] == result_data.shape[0]):
        print("DATAS HAVE DIFFERENT LENGTHS, QUITTING")
        exit()
    else:
        print("NUM DATA: ", result_data.shape[0])

    print("augmenting data")
    symmetries = [
        lambda x: x,
        lambda x: np.rot90(x, 1, (0, 1)),
        lambda x: np.rot90(x, 2, (0, 1)),
        lambda x: np.rot90(x, 3, (0, 1)),
        np.fliplr,
        lambda x: np.rot90(np.fliplr(x), 1, (0, 1)),
        lambda x: np.rot90(np.fliplr(x), 2, (0, 1)),
        lambda x: np.rot90(np.fliplr(x), 3, (0, 1)),
    ]

    aug_game_state_data = np.empty((0, 13, 13, 9))
    aug_pi_data = np.empty((0, 13, 13, 1))
    for symmetry in symmetries:
        aug_game_state_data = np.concatenate(
            (aug_game_state_data, np.array(list(map(symmetry, game_state_data)))),
            axis=0,
        )
        aug_pi_data = np.concatenate(
            (aug_pi_data, np.array(list(map(symmetry, pi_data)))), axis=0
        )
    aug_result_data = np.tile(result_data, len(symmetries))

    print("loading model")
    net = load(constants["NET_PATH"])

    print("training_model")
    history = net.fit(
        {"main_input": aug_game_state_data},
        {"value_head": aug_result_data, "policy_head": aug_pi_data},
        epochs=constants["NUM_EPOCH"],
        batch_size=constants["MINI_BATCH"],
        shuffle=True,
    )

    print("saving model")
    save(net, constants["NET_PATH"])

    print("SAVING HISTORY")
    counter = 1
    filename = "models/history#{}.pkl"
    while os.path.isfile(filename.format(counter)):
        counter += 1
    filename = filename.format(counter)

    with open(filename, "wb") as file_pi:
        pickle.dump(history.history, file_pi)

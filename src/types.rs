use std::{error::Error, path::PathBuf};

use ndarray::{concatenate, Array1, Array3, Array4, Axis};
use ndarray_npy::{read_npy, write_npy};

use crate::{
    constants::{self, sizes},
    rules::types::*,
};

pub struct TrainingData {
    num_turns: usize,
    game_state_data: Vec<bool>,
    pi_data: Vec<f32>,
    outcomes: Option<Vec<f32>>,
    sides: Vec<Side>,
}

impl TrainingData {
    pub fn new() -> Self {
        Self {
            num_turns: 0,
            game_state_data: Vec::new(),
            pi_data: Vec::new(),
            outcomes: None,
            sides: Vec::new(),
        }
    }
    pub fn append_turn(&mut self, gs: &GameState, p: &Array3<f32>) {
        self.num_turns += 1;
        self.sides.push(gs.get_side());
        self.game_state_data
            .extend(gs.get_contents_clone().into_iter());
        self.pi_data.extend(p.iter());
    }
    pub fn set_result(&mut self, result: GameResult) {
        self.outcomes = Some(
            self.sides
                .iter()
                .map(|side| result.outcome_for_side(*side))
                .collect(),
        );
    }
    pub fn dump(self) -> Result<(), Box<dyn Error>> {
        let game_state_data_path: PathBuf = [constants::TRAINING_DATA_PATH, "game_state_data.npy"]
            .iter()
            .collect();
        let pi_data_path: PathBuf = [constants::TRAINING_DATA_PATH, "pi_data.npy"]
            .iter()
            .collect();
        let result_data_path: PathBuf = [constants::TRAINING_DATA_PATH, "result_data.npy"]
            .iter()
            .collect();
        let game_state_data_path = game_state_data_path.to_str().unwrap();
        let pi_data_path = pi_data_path.to_str().unwrap();
        let result_data_path = result_data_path.to_str().unwrap();

        {
            let old_game_state_data: Array4<bool> =
                read_npy(game_state_data_path).expect("Can't read previous game data");
            let new_game_state_data = Array4::from_shape_vec(
                (
                    self.num_turns,
                    sizes::GAME_STATE_HEIGHT,
                    sizes::GAME_STATE_WIDTH,
                    sizes::GAME_STATE_PLANES,
                ),
                self.game_state_data,
            )
            .unwrap();
            let game_state_data = concatenate![Axis(0), old_game_state_data, new_game_state_data];
            write_npy(game_state_data_path, &game_state_data)?;
        }
        {
            let old_pi_data: Array4<f32> =
                read_npy(pi_data_path).expect("Can't read previous pi data");
            let new_pi_data = Array4::from_shape_vec(
                (
                    self.num_turns,
                    sizes::MOVE_WIDTH,
                    sizes::MOVE_HEIGHT,
                    sizes::MOVE_PLANES,
                ),
                self.pi_data,
            )
            .unwrap();
            let pi_data = concatenate![Axis(0), old_pi_data, new_pi_data];
            write_npy(pi_data_path, &pi_data)?;
        }
        {
            let old_result_data: Array1<f32> =
                read_npy(result_data_path).expect("Can't read previous result data");
            let new_result_data = Array1::from(
                self.outcomes
                    .expect("Can't dump because result was not set"),
            );
            let result_data = concatenate![Axis(0), old_result_data, new_result_data];
            write_npy(result_data_path, &result_data)?;
        }

        Ok(())
    }
}

use lib::constants;
use lib::monte_carlo_tree_search;
use lib::rules;
use lib::rules::types::GameState;
use lib::rules::types::NeuralNet;

use std::error::Error;
use std::path::PathBuf;

use tensorflow;
use tensorflow::Code;
use tensorflow::Status;

use crate::types::TrainingData;

mod types {
    use lib::{
        constants::{self, sizes},
        rules::types::{GameResult, GameState, Side},
    };
    use ndarray::{Array1, Array3, Array4};
    use ndarray_npy::{write_npy, WriteNpyError};

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
        pub fn append_turn(&mut self, gs: GameState, p: Array3<f32>) {
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
        pub fn dump(self) -> Result<(), WriteNpyError> {
            write_npy(
                constants::TRAINING_DATA_PATH.to_owned() + "game_state_data.npy",
                &Array4::from_shape_vec(
                    (
                        self.num_turns,
                        sizes::GAME_STATE_HEIGHT,
                        sizes::GAME_STATE_WIDTH,
                        sizes::GAME_STATE_HEIGHT,
                    ),
                    self.game_state_data,
                )
                .unwrap(),
            )?;

            write_npy(
                constants::TRAINING_DATA_PATH.to_owned() + "pi_data.npy",
                &Array1::from(self.pi_data),
            )?;

            write_npy(
                constants::TRAINING_DATA_PATH.to_owned() + "",
                &Array1::from(self.outcomes.expect("Can't dump because result was not set")),
            )?;

            Ok(())
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    rules::vaildate_consts()?;
    constants::write_constants_to_file()?;

    let model_file: PathBuf = [constants::model::NET_PATH, "saved_model.pb"]
        .iter()
        .collect();
    if !model_file.exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run '/init_models.py' to generate {} and try again.",
                    model_file.display()
                ),
            )
            .unwrap(),
        ));
    }

    let net = NeuralNet::new();
    for _ in 0..constants::NUM_GAME_PER_STEP {
        let mut game_state = GameState::init_game_state();
        let mut tree_search = monte_carlo_tree_search::TreeSearch::new(game_state.clone());
        let mut res = game_state.evaluate();
        let mut training_data = TrainingData::new();

        let mut turn_number: usize = 1;
        while !res.has_ended() {
            let tree_search_output = tree_search.search(&net, true);
            training_data.append_turn(game_state.clone(), tree_search_output.pi.clone());

            let best_move = tree_search_output.best_move;
            game_state.move_game(best_move, None);

            res = game_state.evaluate();
            turn_number += 1;
        }
        println!("{}", game_state.get_board_view());
        println!("# of moves: {}", &turn_number);
        println!("{}", res);

        training_data.set_result(res);
        training_data.dump()?;
    }
    // let r = net.run(&game_state);
    // println!("Pi: {:?}", r.pi);

    Ok(())
}

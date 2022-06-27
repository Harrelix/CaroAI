use crossterm::execute;
use lib::constants;
use lib::monte_carlo_tree_search;
use lib::rules;
use lib::rules::types::GameState;
use lib::rules::types::NeuralNet;
use lib::types::TrainingData;

use crossterm::{cursor, terminal};
use std::error::Error;
use std::fs::OpenOptions;
use std::io::{stdout, Write};
use std::path::PathBuf;

use tensorflow;
use tensorflow::Code;
use tensorflow::Status;

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
                    "Run 'python scripts/init_models.py' to generate {} and try again.",
                    model_file.display()
                ),
            )
            .unwrap(),
        ));
    }

    let mut stdout = stdout();
    let mut log = OpenOptions::new()
        .append(true)
        .create(true)
        .open(constants::LOG_PATH)
        .unwrap();
    writeln!(log, "--New session starts--")?;

    let net = NeuralNet::new();

    for g in 0..constants::NUM_GAME_PER_STEP {
        writeln!(log, "Game #{}", g)?;
        writeln!(
            stdout,
            "Generating game number {}/{}",
            g,
            constants::NUM_GAME_PER_STEP
        )?;
        let mut game_state = GameState::init_game_state();
        let mut tree_search = monte_carlo_tree_search::TreeSearch::new(game_state.clone());
        let mut res = game_state.evaluate();
        let mut training_data = TrainingData::new();

        let mut turn_number: usize = 1;
        while !res.has_ended() {
            write!(stdout, "Move #{}\r", turn_number)?;
            stdout.flush()?;
            let tree_search_output = tree_search.search(&net, true);
            training_data.append_turn(&game_state, &tree_search_output.pi);

            let best_move = tree_search_output.best_move;
            game_state.move_game(best_move, None);
            writeln!(log, "{:?}", best_move)?;

            res = game_state.evaluate();
            turn_number += 1;
        }
        writeln!(log)?;
        execute!(
            stdout,
            cursor::MoveToPreviousLine(1),
            terminal::Clear(terminal::ClearType::FromCursorDown)
        )?;
        writeln!(
            stdout,
            "GAME #{} RESULT: {}, in {} moves",
            g, res, &turn_number
        )?;

        write!(stdout, "Saving data...")?;
        training_data.set_result(res);
        training_data.dump()?;
        write!(stdout, "Done\r")?;
    }
    writeln!(log, "--Session Ended--")?;

    Ok(())
}

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use lib::constants;
use lib::monte_carlo_tree_search::TreeSearch;
use lib::rules;
use lib::rules::types::GameState;
use lib::rules::types::NeuralNet;
use lib::types::TrainingData;
use ndarray_npy::WriteNpyError;

use std::error::Error;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use tensorflow;
use tensorflow::Code;
use tensorflow::Status;

/// Struct that hold information of the log to send to the logging thread (logger),
/// which will log the text into LOG_PATH
///
/// `text`: info to add to the log file
/// `channel`: which file to log to
///
struct LogText {
    text: String,
    channel: usize,
}

/// Process that recieve LogText and write it to the log files at LOG_PATH
fn logger(log_rx: Receiver<LogText>) -> io::Result<()> {
    // vector to hold all log files
    let mut log_files = Vec::with_capacity(constants::NUM_THREADS);
    for i in 0..constants::NUM_THREADS {
        let path: PathBuf = [constants::LOG_PATH.to_owned(), format!("thread #{i}.txt")]
            .iter()
            .collect();
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .unwrap();
        writeln!(file, "--New session starts--")?;
        log_files.push(file);
    }

    loop {
        match log_rx.recv() {
            // write to log file
            Ok(log) => write!(log_files[log.channel], "{}", log.text)?,
            // end process if disconnected
            Err(_) => {
                for mut file in log_files {
                    writeln!(file, "--Session Ended--")?;
                }
                return Ok(());
            }
        }
    }
}

/// Indicate the progress made in generating game.
/// The `usize` inside indicates the process (thread) that just had this progress
/// Used to display the progress in progress_printer
enum ProgressSignal {
    Step(usize),
    New(usize),
    End(usize),
}

/// Process that prints the progress. Uses `Indicatif::MultiProgress`
fn progress_printer(
    progress_rx: Receiver<ProgressSignal>,
    num_games: [usize; constants::NUM_THREADS],
) -> Result<(), io::Error> {
    // initializes the progress bars
    let mp = MultiProgress::new();
    let mut pbars = Vec::with_capacity(constants::NUM_THREADS);
    for i in 0..constants::NUM_THREADS {
        let bar = mp.add(ProgressBar::new(num_games[i] as u64));
        let style = ProgressStyle::default_bar()
        .template(&format!("Thread #{i} [{{elapsed_precise}}] {{bar:40.cyan/blue}} Game #{{pos:4}}/{{len:5}} {{msg}}")[..]);
        bar.set_style(style);
        pbars.push(bar);
    }

    let barrier = Arc::new(Barrier::new(2));
    let bc = Arc::clone(&barrier);
    // spawns a new thread that updates all the progress bars
    let _ = thread::spawn(move || {
        let mut move_numbers = vec![0; constants::NUM_THREADS];
        let mut flag = false; // indicate we waited for barrier
        loop {
            match progress_rx.recv() {
                Ok(ProgressSignal::New(i)) => {
                    move_numbers[i] = 0;
                    pbars[i].inc(1);
                }
                Ok(ProgressSignal::Step(i)) => {
                    move_numbers[i] += 1;
                    pbars[i].set_message(format!("Move #{}", move_numbers[i]));
                }
                Ok(ProgressSignal::End(i)) => pbars[i].finish_with_message("done"),
                Err(_) => {
                    for bar in pbars {
                        bar.finish_with_message("done")
                    }
                    break;
                }
            }
            // prevents mp joining prematurely
            if !flag && move_numbers.iter().all(|&x| x > 1) {
                flag = true;
                bc.wait();
            }
        }
    });

    // wait until the bars starts rolling
    barrier.wait();
    // Wait for the progress bars to report that they are finished
    mp.join()?;
    Ok(())
}

// Dumps data when recieved one
fn dumper(data_rx: Receiver<TrainingData>) -> Result<(), WriteNpyError> {
    loop {
        match data_rx.recv() {
            Ok(data) => data.dump()?,
            Err(_) => return Ok(()),
        }
    }
}

/// Generates the games for the training datas
/// `thread_number` is the thread "id" this function is in
///
fn generate_games(
    num_game: usize,
    net: Arc<NeuralNet>,
    log_tx: Sender<LogText>,
    data_tx: Sender<TrainingData>,
    progress_tx: Sender<ProgressSignal>,
    thread_number: usize,
) {
    for g in 0..num_game {
        // log start of game
        log_tx
            .send(LogText {
                text: format!(
                    "Generating game number {g}/{}\n",
                    constants::NUM_GAME_PER_STEP
                ),
                channel: thread_number,
            })
            .unwrap();
        // initialize stuffs
        let mut game_state = GameState::init_game_state();
        let mut tree_search = TreeSearch::new(game_state.clone());
        let mut res = game_state.evaluate();
        let mut training_data = TrainingData::new();

        // game loop
        while !res.has_ended() {
            // update progress bar
            progress_tx
                .send(ProgressSignal::Step(thread_number))
                .unwrap();

            // get output from tree search
            let tree_search_output = tree_search.search(&net, true);

            // add this turn to the training data
            training_data.append_turn(&game_state, &tree_search_output.pi);

            // move the game based on teh tree search output
            let best_move = tree_search_output.best_move;
            game_state.move_game(best_move, None);
            // log the move
            log_tx
                .send(LogText {
                    text: format!("{:?}\n", best_move),
                    channel: thread_number,
                })
                .unwrap();

            res = game_state.evaluate();
        }
        // uodate the game result to all the moves data after finishing the game
        training_data.set_result(res);

        // log that we finish the game
        log_tx
            .send(LogText {
                text: format!("{}\n\n", res),
                channel: thread_number,
            })
            .unwrap();

        // send the data to be dumped
        data_tx.send(training_data).unwrap();
        // resets progress bar
        progress_tx
            .send(ProgressSignal::New(thread_number))
            .unwrap();
    }
    // finish the progress bar after finished generating games
    progress_tx
        .send(ProgressSignal::End(thread_number))
        .unwrap();
}

fn main() -> Result<(), Box<dyn Error>> {
    // checks if constants are valid
    rules::vaildate_consts()?;
    // update constants.jsonc for scripts
    constants::write_constants_to_file()?;

    let model_file: PathBuf = [constants::model::NET_PATH, "saved_model.pb"]
        .iter()
        .collect();
    // heck if the model is there
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

    // load the network
    let net = Arc::new(NeuralNet::new());

    // signal transmiters and receivers
    let (log_tx, log_rx) = mpsc::channel();
    let (data_tx, data_rx) = mpsc::channel();
    let (progress_tx, progress_rx) = mpsc::channel();

    // ==spawning the threads==
    let mut handles = Vec::with_capacity(constants::NUM_THREADS);
    // distribute load among threads evenly
    let mut num_games =
        [constants::NUM_GAME_PER_STEP / constants::NUM_THREADS; constants::NUM_THREADS];
    // add remainder games
    num_games[..constants::NUM_GAME_PER_STEP % constants::NUM_THREADS]
        .iter_mut()
        .for_each(|x| *x += 1);
    // spawn the threads
    for i in 0..constants::NUM_THREADS {
        let ltx = log_tx.clone();
        let dtx = data_tx.clone();
        let ptx = progress_tx.clone();
        let net_ref = Arc::clone(&net);

        let handle = thread::spawn(move || generate_games(num_games[i], net_ref, ltx, dtx, ptx, i));
        handles.push(handle);
    }
    // we don't ned the transmitter anymore in this thread (because we cloned it above)
    drop(log_tx);
    drop(data_tx);
    drop(progress_tx);
    
    // receiver threads
    let logger_handle = thread::spawn(move || logger(log_rx));
    let dumper_handle = thread::spawn(move || dumper(data_rx));
    let progress_handle = thread::spawn(move || progress_printer(progress_rx, num_games));

    // wait for everything to finish
    for handle in handles {
        handle.join().unwrap();
    }
    logger_handle.join().unwrap()?;
    dumper_handle.join().unwrap()?;
    progress_handle.join().unwrap()?;
    
    println!("DONE!");
    Ok(())
}

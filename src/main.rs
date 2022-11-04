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

struct LogText {
    text: String,
    channel: usize,
}

enum ProgressSignal {
    Step(usize),
    New(usize),
    End(usize),
}

fn generate_games(
    num_game: usize,
    net: Arc<NeuralNet>,
    log_tx: Sender<LogText>,
    data_tx: Sender<TrainingData>,
    progress_tx: Sender<ProgressSignal>,
    thread_number: usize,
) {
    for g in 0..num_game {
        log_tx
            .send(LogText {
                text: format!(
                    "Generating game number {g}/{}\n",
                    constants::NUM_GAME_PER_STEP
                ),
                channel: thread_number,
            })
            .unwrap();
        let mut game_state = GameState::init_game_state();
        let mut tree_search = TreeSearch::new(game_state.clone());
        let mut res = game_state.evaluate();
        let mut training_data = TrainingData::new();

        while !res.has_ended() {
            progress_tx
                .send(ProgressSignal::Step(thread_number))
                .unwrap();
            let tree_search_output = tree_search.search(&net, true);
            training_data.append_turn(&game_state, &tree_search_output.pi);

            let best_move = tree_search_output.best_move;
            game_state.move_game(best_move, None);
            log_tx
                .send(LogText {
                    text: format!("{:?}\n", best_move),
                    channel: thread_number,
                })
                .unwrap();

            res = game_state.evaluate();
        }
        training_data.set_result(res);

        log_tx
            .send(LogText {
                text: format!("{}\n\n", res),
                channel: thread_number,
            })
            .unwrap();
        data_tx.send(training_data).unwrap();
        progress_tx
            .send(ProgressSignal::New(thread_number))
            .unwrap();
    }
    progress_tx
        .send(ProgressSignal::End(thread_number))
        .unwrap();
}

fn logger(log_rx: Receiver<LogText>) -> io::Result<()> {
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
            Ok(log) => write!(log_files[log.channel], "{}", log.text)?,
            Err(_) => {
                for mut file in log_files {
                    writeln!(file, "--Session Ended--")?;
                }
                return Ok(());
            }
        }
    }
}

fn dumper(data_rx: Receiver<TrainingData>) -> Result<(), WriteNpyError> {
    loop {
        match data_rx.recv() {
            Ok(data) => data.dump()?,
            Err(_) => return Ok(()),
        }
    }
}

fn progress_printer(
    progress_rx: Receiver<ProgressSignal>,
    num_games: [usize; constants::NUM_THREADS],
) -> Result<(), io::Error> {
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
    let _ = thread::spawn(move || {
        let mut move_numbers = vec![0; constants::NUM_THREADS];
        let mut flag = true;
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
            if flag && move_numbers.iter().all(|&x| x > 1) {
                flag = false;
                bc.wait();
            }
        }
    });
    barrier.wait();
    mp.join()?;
    Ok(())
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
                    "Run 'python scripts/init_models.py' to generate {} and try again.",
                    model_file.display()
                ),
            )
            .unwrap(),
        ));
    }
    let net = Arc::new(NeuralNet::new());

    let (log_tx, log_rx) = mpsc::channel();
    let (data_tx, data_rx) = mpsc::channel();
    let (progress_tx, progress_rx) = mpsc::channel();

    let mut handles = Vec::with_capacity(constants::NUM_THREADS);
    let mut num_games =
        [constants::NUM_GAME_PER_STEP / constants::NUM_THREADS; constants::NUM_THREADS];
    num_games[..constants::NUM_GAME_PER_STEP % constants::NUM_THREADS]
        .iter_mut()
        .for_each(|x| *x += 1);
    for i in 0..constants::NUM_THREADS {
        let ltx = log_tx.clone();
        let dtx = data_tx.clone();
        let ptx = progress_tx.clone();
        let net_ref = Arc::clone(&net);

        let handle = thread::spawn(move || generate_games(num_games[i], net_ref, ltx, dtx, ptx, i));
        handles.push(handle);
    }
    drop(log_tx);
    drop(data_tx);
    drop(progress_tx);

    let logger_handle = thread::spawn(move || logger(log_rx));
    let dumper_handle = thread::spawn(move || dumper(data_rx));
    let progress_handle = thread::spawn(move || progress_printer(progress_rx, num_games));

    for handle in handles {
        handle.join().unwrap();
    }
    logger_handle.join().unwrap()?;
    dumper_handle.join().unwrap()?;
    progress_handle.join().unwrap()?;
    println!("DONE!");
    Ok(())
}

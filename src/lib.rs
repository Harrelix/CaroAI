pub mod monte_carlo_tree_search;

pub mod constants;

pub mod rules;

#[cfg(test)]
mod test {
    use crate::rules::types::*;
    use std::matches;

    #[test]
    fn play_10_boards_randomly() {
        for _ in 0..10 {
            let mut board = Board::init_board();
            println!("{}", board);
            while matches!(board.evaluate(), GameResult::NotFinished) {
                board.move_board_randomly();
                println!("{}", board);
                println!();
            }
            println!("{}", board);
        }
    }

    #[test]
    fn play_10_games_randomly() {
        for _ in 0..10 {
            let mut game_state = GameState::init_game_state();
            let mut board = game_state.get_board_view();
            while matches!(board.evaluate(), GameResult::NotFinished) {
                game_state.move_game_randomly();
                board = game_state.get_board_view();

                println!("{}", board);
            }
            println!("{}", board.evaluate());
        }
    }
}

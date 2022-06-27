use crate::constants::{self, sizes};

use ndarray::{ArrayBase, Data, Dim, RawData};

pub mod types;

pub fn vaildate_consts() -> Result<(), Box<dyn std::error::Error>> {
    if sizes::NUM_IN_A_ROW_FOR_WIN < 127
        && sizes::GAME_STATE_SHAPE.0 < 127
        && sizes::GAME_STATE_SHAPE.1 < 127
        && sizes::GAME_STATE_SHAPE.2 < 127
        // && constants::GAME_LENGTH_CAP < 162
        && sizes::PLAYER_TO_MOVE_INDEX_IN_BOARD <= sizes::BOARD_PLANES
    {
        Ok(())
    } else {
        Err("Bad constants :(".into())
    }
}

const DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

fn has_n_in_a_row_in_dir<T: Data + RawData<Elem = bool>>(
    plane: &ArrayBase<T, Dim<[usize; 2]>>,
    x: usize,
    y: usize,
    dx: i8,
    dy: i8,
    n: usize,
) -> bool {
    if !plane[[y, x]] {
        return false;
    }
    if n == 1 {
        return true;
    }
    let px = x as i8 + dx;
    let py = y as i8 + dy;
    if 0 <= px && px < plane.shape()[1] as i8 && 0 <= py && py < plane.shape()[0] as i8 {
        return has_n_in_a_row_in_dir(plane, px as usize, py as usize, dx, dy, n - 1);
    }
    false
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::constants;
    use types::*;
    #[test]
    fn board_evaluation_test() {
        let mut board = Board::init_board();
        assert!(
            matches!(board.evaluate(), GameResult::NotFinished),
            "empty board is finished??"
        );
        board.set_grid(0, 0, 0, true);
        board.set_grid(0, 1, 0, true);
        board.set_grid(0, 2, 0, true);
        assert!(matches!(board.evaluate(), GameResult::NotFinished));
        board.set_grid(0, 3, 0, true);
        board.set_grid(0, 4, 0, true);
        println!("{}", board);
        println!("{}", board.evaluate());
        assert!(
            matches!(board.evaluate(), GameResult::XWins),
            "can't detect wins"
        );
        board.set_grid(0, 3, 0, false);
        assert!(matches!(board.evaluate(), GameResult::NotFinished));
        board.set_grid(2, 2, 1, true);
        board.set_grid(3, 3, 1, true);
        board.set_grid(4, 4, 1, true);
        assert!(matches!(board.evaluate(), GameResult::NotFinished));
        board.set_grid(5, 5, 1, true);
        board.set_grid(6, 6, 1, true);
        assert!(matches!(board.evaluate(), GameResult::OWins));
        board.set_grid(7, 7, 0, true);
        assert!(matches!(board.evaluate(), GameResult::OWins));
        board.set_grid(1, 1, 0, true);
        if constants::BLOCKED_HEADS_RULE {
            assert!(
                matches!(board.evaluate(), GameResult::NotFinished),
                "can't detect blocked"
            );
        } else {
            assert!(matches!(board.evaluate(), GameResult::OWins));
        }
    }

    #[test]
    fn legal_moves_test() {
        let mut board = Board::init_board();
        assert_eq!(
            board.get_legal_moves(None).len(),
            sizes::BOARD_HEIGHT * sizes::BOARD_WIDTH
        );

        board.move_board_randomly();
        board.move_board_randomly();
        board.move_board_randomly();
        board.move_board_randomly();
        println!("{}", board);
        assert_eq!(
            board.get_legal_moves(None).len(),
            sizes::BOARD_HEIGHT * sizes::BOARD_WIDTH - 4
        );
    }

    #[test]
    fn move_game_test() {
        let mut game = GameState::init_game_state();
        game.move_game(Move::new(0, 0), None);
        assert!(game.get_grid(0, 0, 6));
        assert!(game.get_grid(0, 0, 8));
        game.move_game(Move::new(0, 1), None);
        assert!(game.get_grid(0, 0, 6));
        assert!(game.get_grid(0, 1, 7));
        assert!(game.get_grid(0, 0, 4));
        assert!(!game.get_grid(0, 1, 5));

        assert!(!game.get_grid(0, 0, 8));
        game.move_game(Move::new(0, 2), None);
        assert!(game.get_grid(0, 2, 6));

        assert!(game.get_grid(0, 0, 4));
        assert!(game.get_grid(0, 0, 2));

        assert!(!game.get_grid(0, 2, 4));

        assert!(game.get_grid(0, 1, 5));

        assert!(game.get_grid(0, 0, 8));
    }
}

use std::{collections::HashMap, fmt::Debug};

pub const GAME_LENGTH_CAP: usize = 81;
pub const BLOCKED_HEADS_RULE: bool = false;
pub const MASKING_VALUE: f32 = -100.0;
pub const NUM_GAME_PER_STEP: usize = 3;

pub const TRAINING_DATA_PATH: &str = "training_data/";
pub const LOG_PATH: &str = "log.txt";

pub mod sizes {
    pub const NUM_IN_A_ROW_FOR_WIN: usize = 5;
    // Plane size: 13 x 13; Number of planes: (1 + 1) * 3 + (1 + 1 + 1) = 9
    // Feature              Planes
    // X                    1       |
    // O                    1       | -> 1 + 3 previous board
    // Player to move       1   (0 = X to move, 1 = O to move)
    pub const BOARD_WIDTH: usize = 13;
    pub const BOARD_HEIGHT: usize = 13;
    pub const BOARD_PLANES: usize = 3;
    pub const BOARD_SHAPE: (usize, usize, usize) = (BOARD_HEIGHT, BOARD_WIDTH, BOARD_PLANES);
    pub const PLANES_PER_PREV_BOARD: usize = 2;
    pub const NUM_PREV_BOARDS: usize = 3;
    pub const PLAYER_TO_MOVE_INDEX_IN_BOARD: usize = 2;

    pub const GAME_STATE_WIDTH: usize = BOARD_WIDTH;
    pub const GAME_STATE_HEIGHT: usize = BOARD_HEIGHT;
    pub const GAME_STATE_PLANES: usize = PLANES_PER_PREV_BOARD * NUM_PREV_BOARDS + BOARD_PLANES;
    pub const GAME_STATE_SHAPE: (usize, usize, usize) =
        (GAME_STATE_HEIGHT, GAME_STATE_WIDTH, GAME_STATE_PLANES);
    pub const BOARD_STATE_START: usize = 6;
    pub const PLAYER_TO_MOVE_INDEX_IN_STATE: usize = 8;

    pub const MOVE_WIDTH: usize = 13;
    pub const MOVE_HEIGHT: usize = 13;
    pub const MOVE_PLANES: usize = 1;
    pub const MOVE_SHAPE: (usize, usize, usize) = (MOVE_WIDTH, MOVE_HEIGHT, MOVE_PLANES);
}
pub mod mcts {
    pub const NUM_SEARCH: usize = 64;
    pub const C_PUCT: f32 = 1.0;
    pub const EXPLORATION: f32 = 1.0;
    pub const DIRICHLET_ALPHA: f32 = 0.25;
    pub const DIRICHLET_WEIGHT: f32 = 0.25;
    pub const SECONDARY_DIRICHLET_WEIGHT: f32 = 0.01;
}
pub mod model {
    pub const NET_PATH: &str = "models/CaroZero";
    pub const NUM_HIDDEN_RES_BLOCK: usize = 3;
    pub const NUM_FILTERS: usize = 48;
    pub const KERNEL_SIZE: (usize, usize) = (3, 3);
    pub const LEARNING_RATE: f32 = 0.01;
    pub const MOMENTUM: f32 = 0.9;
    pub const REG_CONST: f32 = 0.0001;
}

pub fn move_to_index(pos: (usize, usize)) -> usize {
    let (x, y) = pos;
    x + y * (sizes::MOVE_SHAPE.0)
}

pub fn index_to_move(index: usize) -> (usize, usize) {
    (index % sizes::MOVE_SHAPE.0, index / sizes::MOVE_SHAPE.1)
}

pub fn write_constants_to_file() -> Result<(), Box<dyn std::error::Error>> {
    use model::*;
    use sizes::*;
    use std::any::Any;
    use std::fs;

    macro_rules! hashmap {
        ($($key: expr), *) => {{
            let mut map: HashMap<&str, Box<dyn Any>> = HashMap::new();
            $(map.insert(stringify!($key), Box::new($key)); )*
            map
        }};
    }
    fn push<T: Debug>(s: &mut String, key: &str, val: T) {
        s.push_str(
            format!("\t\"{}\": {:?},\n", key, val)
                .replace("(", "[")
                .replace(")", "]")
                .as_str(),
        );
    }

    let cm: HashMap<&str, Box<dyn Any>> = hashmap![
        GAME_STATE_SHAPE,
        NET_PATH,
        NUM_HIDDEN_RES_BLOCK,
        NUM_FILTERS,
        KERNEL_SIZE,
        LEARNING_RATE,
        MOMENTUM,
        REG_CONST
    ];

    let mut s =
        String::from("// File written by src/constants.rs, intended for init_models.py\n{\n");
    for (k, v) in cm.iter() {
        if let Some(unsigned) = v.downcast_ref::<usize>() {
            push(&mut s, k, unsigned);
        } else if let Some(float) = v.downcast_ref::<f32>() {
            push(&mut s, k, float);
        } else if let Some(boolean) = v.downcast_ref::<bool>() {
            push(&mut s, k, boolean);
        } else if let Some(tuple2) = v.downcast_ref::<(usize, usize)>() {
            push(&mut s, k, tuple2);
        } else if let Some(tuple3) = v.downcast_ref::<(usize, usize, usize)>() {
            push(&mut s, k, tuple3);
        } else if let Some(string) = v.downcast_ref::<&str>() {
            push(&mut s, k, string);
        } else {
            Err("Can't parse constants to json :(")?;
        }
    }
    s.pop();
    s.pop();
    s.push_str("\n}");

    fs::write("constants.jsonc", s)?;

    Ok(())
}

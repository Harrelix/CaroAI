use rand::Rng;
use std::fmt::{self, Display, Formatter};

use ndarray::{s, Array3, ArrayBase, Axis, Data, DataMut, Dim, OwnedRepr, RawData, ViewRepr};
use tensorflow::{
    eager::{self, raw_ops, Context, ReadonlyTensor, ToTensorHandle},
    Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};

use crate::constants::{self, sizes};

use super::{has_n_in_a_row_in_dir, DIRECTIONS};

pub struct Coord3D {
    x: usize,
    y: usize,
    t: usize,
}
impl Coord3D {
    pub fn new(x: usize, y: usize, t: usize) -> Self {
        Coord3D { x, y, t }
    }
    pub fn x(&self) -> usize {
        self.x
    }
    pub fn y(&self) -> usize {
        self.y
    }
    pub fn t(&self) -> usize {
        self.t
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Move {
    pub x: usize,
    pub y: usize,
    pub p: usize,
}

impl Move {
    pub fn new(x: usize, y: usize) -> Self {
        if x < sizes::MOVE_SHAPE.0 {
            if y < sizes::MOVE_SHAPE.1 {
                Self { x, y, p: 0 }
            } else {
                panic!(
                    "y is {}, but has to be smaller that {}",
                    y,
                    sizes::MOVE_SHAPE.1
                );
            }
        } else {
            panic!(
                "x is {}, but has to be smaller that {}",
                x,
                sizes::MOVE_SHAPE.0
            );
        }
    }
    pub fn get_move_arr(&self) -> [usize; 3] {
        [self.y, self.x, self.p]
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}, {}", self.x, self.y, self.p)
    }
}

#[derive(Clone, Copy)]
pub enum Side {
    X,
    O,
}
impl Side {
    pub fn to_bool(&self) -> bool {
        match self {
            Side::X => true,
            Side::O => false,
        }
    }
}
pub enum GameResult {
    XWins,
    OWins,
    Draws,
    NotFinished,
}

impl GameResult {
    pub fn has_ended(&self) -> bool {
        match *self {
            GameResult::XWins => true,
            GameResult::OWins => true,
            GameResult::Draws => true,
            GameResult::NotFinished => false,
        }
    }
    pub fn outcome_for_side(&self, side: Side) -> f32 {
        match *self {
            GameResult::XWins => {
                if matches!(side, Side::X) {
                    1.0
                } else {
                    -1.0
                }
            }
            GameResult::OWins => {
                if matches!(side, Side::O) {
                    1.0
                } else {
                    -1.0
                }
            }
            GameResult::Draws => 0.0,
            GameResult::NotFinished => 0.0,
        }
    }
}

impl fmt::Display for GameResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GameResult::XWins => write!(f, "X WINS!!"),
            GameResult::OWins => write!(f, "O WINS!!"),
            GameResult::Draws => write!(f, "DRAWS!!"),
            GameResult::NotFinished => write!(f, "NOT FINISHED!"),
        }
    }
}

pub struct Board<T: Data + RawData<Elem = bool>> {
    contents: ArrayBase<T, Dim<[usize; 3]>>,
}
impl Board<OwnedRepr<bool>> {
    pub fn init_board() -> Self {
        Board {
            contents: Array3::from_elem(constants::sizes::BOARD_SHAPE, false),
        }
    }
}
impl<T: Data + RawData<Elem = bool>> Board<T> {
    pub fn get_side(&self) -> Side {
        if self.contents[[0, 0, sizes::PLAYER_TO_MOVE_INDEX_IN_BOARD]] {
            Side::O
        } else {
            Side::X
        }
    }
    pub fn get_grid(&self, x: usize, y: usize, p: usize) -> bool {
        self.contents[[y, x, p]]
    }
    pub fn get_plane(&self, plane_index: usize) -> ArrayBase<ViewRepr<&bool>, Dim<[usize; 2]>> {
        self.contents.index_axis(Axis(2), plane_index)
    }

    pub fn get_legal_moves(&self, _side: Option<Side>) -> Vec<Move> {
        let mut result: Vec<Move> = Vec::new();
        for x in 0..sizes::BOARD_WIDTH {
            for y in 0..sizes::BOARD_HEIGHT {
                if !self
                    .contents
                    .slice(s![y, x, ..sizes::PLAYER_TO_MOVE_INDEX_IN_BOARD])
                    .iter()
                    .any(|x| *x)
                {
                    result.push(Move::new(x, y));
                }
            }
        }
        result
    }

    pub fn evaluate(&self) -> GameResult {
        for p in 0..=1 {
            for x in 0..sizes::BOARD_WIDTH {
                for y in 0..sizes::BOARD_HEIGHT {
                    for (dx, dy) in DIRECTIONS {
                        if has_n_in_a_row_in_dir(
                            &self.get_plane(p),
                            x,
                            y,
                            dx,
                            dy,
                            sizes::NUM_IN_A_ROW_FOR_WIN,
                        ) {
                            match p {
                                0 => return GameResult::XWins,
                                1 => return GameResult::OWins,
                                _ => panic!("Super weird error happened!"),
                            }
                        }
                    }
                }
            }
        }
        GameResult::NotFinished
    }
    pub fn get_contents_clone(&self) -> ArrayBase<OwnedRepr<bool>, Dim<[usize; 3]>> {
        self.contents.to_owned()
    }
    pub fn legal_moves_onehot(&self, side: Option<Side>) -> Array3<bool> {
        let mut res = Array3::from_elem(constants::sizes::MOVE_SHAPE, false);
        for mv in self.get_legal_moves(side) {
            res[[mv.y, mv.x, 1]] = true;
        }
        res
    }
}
impl<T: DataMut + RawData<Elem = bool>> Board<T> {
    pub fn set_grid(&mut self, x: usize, y: usize, p: usize, val: bool) {
        self.contents[[y, x, p]] = val;
    }
    pub fn toggle_side(&mut self) {
        let s = self.contents[[0, 0, sizes::PLAYER_TO_MOVE_INDEX_IN_BOARD]];
        self.contents
            .index_axis_mut(Axis(2), sizes::PLAYER_TO_MOVE_INDEX_IN_BOARD)
            .fill(!s);
    }
    pub fn move_board(&mut self, mv: Move, side: Option<Side>) {
        let plane_index = match side {
            Some(Side::X) => 0,
            Some(Side::O) => 1,
            None => match self.get_side() {
                Side::X => 0,
                Side::O => 1,
            },
        };
        self.set_grid(mv.x, mv.y, plane_index, true);
        self.toggle_side();
    }
    pub fn move_board_randomly(&mut self) {
        let mut rng = rand::thread_rng();
        let side = self.get_side();

        let legal_moves = self.get_legal_moves(Some(side));
        let mv = legal_moves[rng.gen_range(0..legal_moves.len())];
        self.move_board(mv, Some(side));
    }
}

impl<T: RawData<Elem = bool> + Data> Display for Board<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut drawing = String::from("");
        for y in 0..sizes::BOARD_HEIGHT {
            for x in 0..sizes::BOARD_WIDTH {
                if self.contents[[y, x, 0]] {
                    drawing.push('X');
                } else if self.contents[[y, x, 1]] {
                    drawing.push('O');
                } else {
                    drawing.push('-');
                }
            }
            drawing.push('\n');
        }
        write!(f, "{}", drawing)
    }
}

#[derive(Clone)]
pub struct GameState {
    contents: Array3<bool>,
}

impl GameState {
    pub fn init_game_state() -> Self {
        GameState {
            contents: Array3::from_elem(sizes::GAME_STATE_SHAPE, false),
        }
    }

    pub fn get_board_view(&self) -> Board<ViewRepr<&bool>> {
        Board {
            contents: self.contents.slice(s![.., .., sizes::BOARD_STATE_START..]),
        }
    }
    pub fn get_board_view_mut(&mut self) -> Board<ViewRepr<&mut bool>> {
        Board {
            contents: self
                .contents
                .slice_mut(s![.., .., sizes::BOARD_STATE_START..]),
        }
    }

    pub fn update_game<T: Data + RawData<Elem = bool>>(&mut self, board: &Board<T>) {
        let temp = self
            .contents
            .slice(s![
                ..,
                ..,
                sizes::PLANES_PER_PREV_BOARD
                    ..sizes::PLANES_PER_PREV_BOARD * (sizes::NUM_PREV_BOARDS + 1)
            ])
            .to_owned();
        self.contents
            .slice_mut(s![
                ..,
                ..,
                ..sizes::PLANES_PER_PREV_BOARD * sizes::NUM_PREV_BOARDS
            ])
            .assign(&temp);
        self.contents
            .slice_mut(s![
                ..,
                ..,
                sizes::PLANES_PER_PREV_BOARD * sizes::NUM_PREV_BOARDS..
            ])
            .assign(&board.get_contents_clone());
    }

    pub fn move_game(&mut self, mv: Move, side: Option<Side>) {
        let temp = self
            .contents
            .slice(s![
                ..,
                ..,
                sizes::PLANES_PER_PREV_BOARD
                    ..sizes::PLANES_PER_PREV_BOARD * (sizes::NUM_PREV_BOARDS + 1)
            ])
            .to_owned();
        self.contents
            .slice_mut(s![
                ..,
                ..,
                ..sizes::PLANES_PER_PREV_BOARD * sizes::NUM_PREV_BOARDS
            ])
            .assign(&temp);
        self.get_board_view_mut().move_board(mv, side);
    }
    pub fn get_contents_clone(&self) -> Array3<bool> {
        self.contents.to_owned()
    }
    pub fn get_grid(&self, x: usize, y: usize, p: usize) -> bool {
        self.contents[[y, x, p]]
    }
    pub fn move_game_randomly(&mut self) {
        let board = self.get_board_view();
        let side = board.get_side();
        let legal_moves = board.get_legal_moves(Some(side));

        let mut rng = rand::thread_rng();
        let mv = legal_moves[rng.gen_range(0..legal_moves.len())];

        self.move_game(mv, Some(side));
    }
    pub fn get_side(&self) -> Side {
        if self.contents[[0, 0, sizes::PLAYER_TO_MOVE_INDEX_IN_STATE]] {
            Side::O
        } else {
            Side::X
        }
    }
    pub fn evaluate(&self) -> GameResult {
        self.get_board_view().evaluate()
    }
    pub fn get_legal_moves(&self, side: Option<Side>) -> Vec<Move> {
        self.get_board_view().get_legal_moves(side)
    }
    pub fn legal_moves_onehot(&self, side: Option<Side>) -> Array3<bool> {
        self.get_board_view().legal_moves_onehot(side)
    }
}
pub struct NeuralNetOutput {
    pub value_head: f32,
    pub policy_head: ReadonlyTensor<f32>,
}

pub struct NeuralNet {
    pub ctx: Context,
    bundle: SavedModelBundle,
    x_op: Operation,
    value_head_op: Operation,
    policy_head_op: Operation,
}
impl NeuralNet {
    pub fn new() -> Self {
        // Create an eager execution context
        let opts = eager::ContextOptions::new();
        let ctx = eager::Context::new(opts).unwrap();

        // Load the model.
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(),
            &["serve"],
            &mut graph,
            constants::model::NET_PATH,
        )
        .expect("Can't load saved model");

        // get in/out operations
        let signature = bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            .unwrap();

        let x_info = signature.get_input("main_input").unwrap();
        let x_op = graph
            .operation_by_name_required(&x_info.name().name)
            .unwrap();

        let policy_head_info = signature
            .get_output("policy_head")
            .expect("Can't get policy head info");
        let value_head_info = signature
            .get_output("value_head")
            .expect("Can't get value head info");

        let policy_head_op = graph
            .operation_by_name_required(&policy_head_info.name().name)
            .expect("Can't get policy head op");
        let value_head_op = graph
            .operation_by_name_required(&value_head_info.name().name)
            .expect("Can't get value head op");
        NeuralNet {
            bundle,
            ctx,
            x_op,
            value_head_op,
            policy_head_op,
        }
    }

    pub fn run(&self, game_state: &GameState) -> NeuralNetOutput {
        let bool_x = Tensor::from(game_state.get_contents_clone()).freeze();
        let cast2float = raw_ops::Cast::new().DstT(tensorflow::DataType::Float);
        let float_x = cast2float.call(&self.ctx, &bool_x).unwrap();
        let batched_x = raw_ops::expand_dims(&self.ctx, &float_x, &0).unwrap();
        let readonly_x = batched_x.resolve().unwrap();
        let x: Tensor<f32> = unsafe { readonly_x.into_tensor() };

        // Run the graph.
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.x_op, 0, &x);

        let value_head_token = args.request_fetch(&self.value_head_op, 0);
        let policy_head_token = args.request_fetch(&self.policy_head_op, 0);

        let result = self.bundle.session.run(&mut args);
        if result.is_err() {
            panic!("Error occured during calculations: {:?}", result);
        }
        let value_head_output: Tensor<f32> = args.fetch(value_head_token).unwrap();
        let policy_head_output: Tensor<f32> = args.fetch(policy_head_token).unwrap();
        let policy_head_output = raw_ops::reshape(
            &self.ctx,
            &policy_head_output.to_handle(&self.ctx).unwrap(),
            &[
                sizes::MOVE_SHAPE.0 as i32,
                sizes::MOVE_SHAPE.1 as i32,
                sizes::MOVE_SHAPE.2 as i32,
            ],
        )
        .unwrap()
        .resolve()
        .unwrap();
        NeuralNetOutput {
            value_head: value_head_output[0],
            policy_head: policy_head_output,
        }
    }
}

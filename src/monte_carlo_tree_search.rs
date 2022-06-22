use crate::constants::{self, mcts, sizes};
use crate::rules::types::{GameState, Move, NeuralNet};
use ndarray::{Array, Array3};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use std::{cell::RefCell, rc::Rc};

use tensorflow::eager::raw_ops;

#[derive(Clone)]
struct Node {
    game_state: GameState,
    n: usize,
    w: f32,
    p: f32,
    q: f32,
    m: Option<Move>,
    children: Vec<Rc<RefCell<Node>>>,
}

impl Node {
    pub fn init_node(game_state: GameState, p: f32, m: Option<Move>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            game_state,
            n: 0,
            w: 0.0,
            p,
            q: 0.0,
            m,
            children: Vec::new(),
        }))
    }

    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    pub fn iter_children(&self) -> impl Iterator<Item = Rc<RefCell<Node>>> + '_ {
        self.children.iter().map(|c| Rc::clone(c))
    }

    pub fn get_child(&self, ind: usize) -> Rc<RefCell<Node>> {
        Rc::clone(&self.children[ind])
    }
}

type Path = Vec<Rc<RefCell<Node>>>;

pub struct MCTSOutput {
    pub best_move: Move,
    pub pi: Array3<f32>,
}

pub struct TreeSearch {
    root_node: Rc<RefCell<Node>>,
}

impl TreeSearch {
    pub fn new(game_state: GameState) -> TreeSearch {
        TreeSearch {
            root_node: Node::init_node(game_state, 0.0, None),
        }
    }
    fn backup(path: Path, value: f32) {
        for node in path.iter() {
            let mut mut_node = node.borrow_mut();
            mut_node.w += value;
            mut_node.n += 1;
            mut_node.q = mut_node.w / mut_node.n as f32;
        }
    }

    pub fn search(&mut self, net: &NeuralNet, play_stochastically: bool) -> MCTSOutput {
        for _ in 0..mcts::NUM_SEARCH {
            let mut path: Path = vec![Rc::clone(&self.root_node)];
            let mut last_node = Rc::clone(&path[0]);
            while last_node.borrow().has_children() {
                let n: f32 = last_node
                    .borrow()
                    .iter_children()
                    .map(|child| child.borrow().n as f32)
                    .sum();
                let ucb: Vec<f32> = last_node
                    .borrow()
                    .iter_children()
                    .map(|child| {
                        let child = child.borrow();
                        child.q + mcts::C_PUCT * child.p * (n.sqrt() / ((1 + child.n) as f32))
                    })
                    .collect();
                let max_ind = TreeSearch::argmax(ucb.iter()).unwrap();
                path.push(last_node.borrow().get_child(max_ind));
                last_node = Rc::clone(path.last().unwrap());
            }

            let backup_val: f32;
            {
                let mut last_node = last_node.borrow_mut();

                let res = last_node.game_state.evaluate();
                backup_val = if res.has_ended() {
                    res.outcome_for_side(last_node.game_state.get_side())
                } else {
                    let legal_move_pool = last_node.game_state.get_legal_moves(None);

                    let net_output = net.run(&last_node.game_state);

                    let mut masked_policy =
                        Array3::from_elem(sizes::MOVE_SHAPE, constants::MASKING_VALUE);
                    for mv in legal_move_pool.iter() {
                        masked_policy[[mv.y, mv.x, mv.p]] =
                            net_output
                                .policy_head
                                .get(&[mv.y as u64, mv.x as u64, mv.p as u64]);
                    }
                    let policy = raw_ops::reshape(
                        &net.ctx,
                        &raw_ops::softmax(&net.ctx, &Array::from_iter(masked_policy.into_iter()))
                            .unwrap(),
                        &[
                            sizes::MOVE_SHAPE.0 as i32,
                            sizes::MOVE_SHAPE.1 as i32,
                            sizes::MOVE_SHAPE.2 as i32,
                        ],
                    )
                    .unwrap()
                    .resolve()
                    .unwrap();

                    for mv in legal_move_pool {
                        let mut new_game_state = last_node.game_state.clone();
                        new_game_state.move_game(mv, None);

                        last_node.children.push(Node::init_node(
                            new_game_state,
                            policy.get(&[mv.y as u64, mv.x as u64, mv.p as u64]),
                            Some(mv),
                        ));
                    }
                    net_output.value_head
                };
            }
            TreeSearch::backup(path, backup_val);
        }

        let exp = if play_stochastically {
            1.0 / mcts::EXPLORATION
        } else {
            10.0
        };
        let mut pi = Array3::zeros(sizes::MOVE_SHAPE);
        let sum_n: usize = self
            .root_node
            .borrow()
            .iter_children()
            .map(|child| child.borrow().n)
            .sum();

        let mut weights = Vec::new();

        for child in self.root_node.borrow().iter_children() {
            let child_pi = ((child.borrow().n as f32) / (sum_n as f32)).powf(exp);
            let child_move = child.borrow().m.expect("Node has no prior move");
            weights.push(child_pi);
            pi[child_move.get_move_arr()] = child_pi;
        }

        let dist = WeightedIndex::new(weights).expect("Root node is leaf node");
        let mut rng = rand::thread_rng();
        self.root_node = Rc::clone(
            &Rc::clone(&self.root_node)
                .borrow()
                .get_child(dist.sample(&mut rng)),
        );

        MCTSOutput {
            best_move: self.root_node.borrow().m.unwrap(),
            pi,
        }
    }

    fn argmax<T, Iter>(xs: Iter) -> Option<usize>
    where
        T: Copy + PartialOrd,
        Iter: Iterator<Item = T>,
    {
        let mut argmax: Option<(usize, T)> = None;
        for (i, x) in xs.enumerate() {
            match argmax {
                None => {
                    argmax = Some((i, x));
                }
                Some((_, y)) => {
                    if y < x {
                        argmax = Some((i, x));
                    }
                }
            }
        }
        argmax.map(|(i, _)| i)
    }
}

# CaroAI (WIP)
Program based on Google DeepMind’s AlphaZero papers, written in Rust, used to play the board game [Caro](https://en.wikipedia.org/wiki/Gomoku#Caro) (a gomoku variant popular in Vietnam).


# How it works
To find the best move (move that maximizes winning probability), given a game state, the AI will expand and explore a tree of possible moves/scenarios originated from that original game state.
Paths (list of moves from the starting state to a different state) can be evaluated using a deep residual network, which will get better over time.
When done evaluating, the path value (winning probability/how favourable the path is) will be updated. Paths with higher values will be prioritized and consequently be explored more often. Therefore, the best move can be determined as it is the most explore one.
## The learning process:
The AI will play a game with itself, from start to finish. At the end of several games, the neural network will be trained, which encourages it to play the winning side's moves and avoid playing the losing side's moves.
After some time of self-playing, the neural network will get better and eventually play the best moves.
## Game states and moves representation
A single board state are represented using a stack of 3 planes with dimension 13 x 13 (the size the board). 3 planes are used to store information about the black and white pieces, and whose turn is it.<br/>
Game states are represented using a stack 9 planes to encode the current and 3 previous boards.<br/>
A move is represented by a single 13x13 plane.
## Network configuration 
*(subject to change)*<br/>
Input -> Convolution Block -> 4 Residual Blocks -> (Policy head, Value head)<br/>
Input: shape = (9, 13, 13), the game state to evaluate<br/>
Convolution Block: Conv layer (128 filters, 3x3 kernel, stride 1)-> batch normalisation -> swish<br/>
Residual Block: Conv block -> conv w/o swish -> skip connection -> swish<br/>
Value head: Conv layer (filter 1x1) -> batch normalisation -> ReLU -> Dense 128 -> swish -> Dense 1; output shape = (1, 1)<br/>
Policy head: Conv block -> Conv layer (50 filter 1x1) -> batch normalisation; output shape = (1, 13, 13)<br/>
## Tree search method (Monte Carlo Tree Search)
Each node will hold these information:
- Game State
- *n* - number of visits to the node
- *w* - total value of the node
- *q* - mean value of the node (= w/n)
- *p* - prior probability (probability of selecting this node earlier)
- *m* - prior move to get to this node<br/><br/>

**The explore loop:**
- Start with a single node for the current game state (root node)
- **Evaluation:** Use the neural network to get value of the game as well as probabilities of playing moves from that game state
- **Expansion:** From the selected node, create child nodes for all the possible move, then update their prior probability using the output of the neural network
- **Backup** Update *n* and *w* of all node in the path
- **Selection** Create a new path (list of nodes) that starts from the root node and ends on a leaf (unexpanded) node. When selecting a single node from many child nodes, it will take into consideration *n* and *w* of all those child nodes. When trying to explore new and uncommon moves (like in the early game), it'll choose nodes that has relatively low *n* (not visited often).And when playing accurately (end game), it'll prioritze *w*. After it has finish selecting a path, the leaf node will be expanded in Expansion and begin a new cycle.

After running a certain number of cycles, the node that was in a selected path most often will be choose and the AI will play the corresponding move.

# Reference
Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2017). Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815.  
Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017). Mastering the game of go without human knowledge. nature, 550(7676), 354-359.

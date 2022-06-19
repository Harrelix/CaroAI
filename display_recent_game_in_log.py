from ast import Lambda
from blessed import Terminal
from copy import deepcopy


def board_display(term, board, prev_move=None):
    res = ""
    for y, row in enumerate(board):
        for x, grid in enumerate(row):
            char = ""
            if grid == " ":
                char += term.white + "-"
            elif grid == "X":
                char += term.red + "X"
            elif grid == "O":
                char += term.green + "O"

            if prev_move is not None:
                if (x, y) == prev_move:
                    res += term.on_white(char)
                else:
                    res += char
            else:
                res += char
        res += "\n"
    return res


if __name__ == "__main__":
    with open("log.txt", "r") as f:
        lines = f.readlines()
        prev = curr = -1
        for i, line in enumerate(lines):
            if line == "\n":
                prev = curr
                curr = i
        game_data = lines[prev + 2 : curr]
    boards = [[[" " for _ in range(13)] for __ in range(13)]]
    turn = "X"
    moves = list(map(lambda x: tuple(map(int, x.split(" "))), game_data))
    moves = [(0, 0)] + moves
    for x, y in moves:
        board = deepcopy(boards[-1])
        board[y][x] = turn
        boards.append(board)
        turn = "X" if turn == "O" else "O"

    term = Terminal()
    board_index = 0

    def print_board():
        print(term.home + term.clear)
        print(board_display(term, boards[board_index], moves[board_index - 1]))
        print()
        print("< > to play, q to exit")

    print_board()
    with term.cbreak():
        val = ""
        while val.lower() != "q":
            val = term.inkey(timeout=None)
            if val in {",", "<"}:
                board_index = max(board_index - 1, 0)
                print_board()
            elif val in {".", ">"}:
                board_index = min(board_index + 1, len(boards) - 1)
                print_board()
        print("exited...", term.normal)

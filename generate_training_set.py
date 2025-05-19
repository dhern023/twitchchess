#!/usr/bin/env python3
import chess.pgn
import numpy as np
import pathlib
from state import State

def get_dataset(num_samples=None):
def extract_boards(instance_game):
    """
    Extracts serialized board from game
    """
    board = instance_game.board()
    list_out = []
    for move in instance_game.mainline_moves():
        board.push(move)
        board_serialized = State(board).serialize()
        list_out.append(board_serialized)

    return list_out

  X,Y = [], []
  gn = 0
  dict_map_result = {'1/2-1/2':0, '0-1':-1, '1-0':1}

  for fname in path_dir_pgn.glob("*.pgn"):
    pgn = open(fname)
    while 1:
      game = chess.pgn.read_game(pgn)
      if game is None:
        break
      result = game.headers['Result']
      if result not in dict_map_result:
        continue
      value = dict_map_result[result]

      # extract
      list_boards_serialized = extract_boards(game)
      X.extend(list_boards_serialized)
      Y.extend([value] * len(list_data))
      print("parsing game %d, got %d examples" % (gn, len(X)))

      # break early
      if num_samples is not None and len(X) > num_samples:
        return X,Y
      gn += 1
  X = np.array(X)
  Y = np.array(Y)
  return X,Y

if __name__ == "__main__":
  X,Y = get_dataset(25000000)
  DIR_DATA = pathtlib.Path(__file__).parent / "data"
  X,Y = get_dataset_via_pgn(DIR_DATA, 25e6)
  np.savez("processed/dataset_25M.npz", X, Y)


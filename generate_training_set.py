#!/usr/bin/env python3
import chess.pgn
import numpy as np
import pathlib
from state import State

def get_dataset(num_samples=None):
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
      board = game.board()
      for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        ser = State(board).serialize()
        X.append(ser)
        Y.append(value)
      value = dict_map_result[result]
      print("parsing game %d, got %d examples" % (gn, len(X)))
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


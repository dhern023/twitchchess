#!/usr/bin/env python3
import chess.pgn
import numpy as np
import pathlib
from state import State

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

def get_dataset_via_pgn(path_dir_pgn, num_samples=None):
  """
  Reads and iterates over pgn files
  chess.pgn.read_game works as a generator
  no way to tell how large the file is aside from counting Results in file.
  For 1M games: 
    1:47:27
  NOTE: A lot of games are ended by giving up

  NOTE: Hard-coded labels
      1/2-1/2 : tie
      0-1     : R player wins
      1-0     : L player wins

  We read the games and process one at a time

  TODO: Periodically writes to out file to avoid out-of-memory (OOM)
  """
  X,Y = [], []
  gn = 0
  dict_map_result = {'1/2-1/2':0, '0-1':-1, '1-0':1}

  for fname in path_dir_pgn.glob("*.pgn"):
    pgn = open(fname)
    while 1:
      # read & parse
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
  DIR_DATA = pathtlib.Path(__file__).parent / "data"
  X,Y = get_dataset_via_pgn(DIR_DATA, 25e6)
  np.savez("processed/dataset_25M.npz", X, Y)


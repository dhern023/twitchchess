#!/usr/bin/env python3
import chess
import numpy as np

def construct_dict_map_pieces(value_shift = 8):
    """
    https://python-chess.readthedocs.io/en/latest/core.html#chess.Piece
    gives the default values
        Gets the symbol P, N, B, R, Q or K for white pieces or the lower-case variants for the black pieces.
    We'll assign values
        f(piece_white) = 1 through N for the white pieces, and
        f(piece_black) = f(piece_white) + 8 for the black pieces

    Returns
        P = 1, p = 1+8
        N = 2, n = 2+8
        ...
        K = 6, k = 6+8

    which leaves (7,13), (15, infty) available
    """
    list_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
    dict_out = {}
    for i, piece in enumerate(list_pieces):
        value = i+1
        dict_out[piece] = value
        dict_out[piece.lower()] = value+value_shift

    return dict_out

class State(object):
  def __init__(self, board=None):
    if board is None:
      self.board = chess.Board()
    else:
      self.board = board

    self.dict_map_pieces = construct_dict_map_pieces()

  def key(self):
    return (self.board.board_fen(), self.board.turn, self.board.castling_rights, self.board.ep_square)

  def serialize(self):
    """
    Store the board piece value in an 8x8 matrix with hard-coded values.
    Convert the board to a bit matrix s.t.,
        state[i,j] are the extracted bits containing four bits per integer
        then keep the rightmost bit

    State is 256 bits according to readme
    """
    assert self.board.is_valid()

    bstate = np.zeros(64, np.uint8) # board state
    for i in range(64):
      pp = self.board.piece_at(i)
      if pp is not None:
        #print(i, pp.symbol())
        bstate[i] = self.dict_map_pieces[pp.symbol()]

    piece_rook = "R"
    value_rook_castled = 7 # hard-coded
    if self.board.has_queenside_castling_rights(chess.WHITE):
      assert bstate[0] == self.dict_map_pieces[piece_rook]
      bstate[0] = value_rook_castled
    if self.board.has_kingside_castling_rights(chess.WHITE):
      assert bstate[7] == self.dict_map_pieces[piece_rook]
      bstate[7] = value_rook_castled
    if self.board.has_queenside_castling_rights(chess.BLACK):
      assert bstate[56] == self.dict_map_pieces[piece_rook.lower()]
      bstate[56] = value_rook_castled+8
    if self.board.has_kingside_castling_rights(chess.BLACK):
      assert bstate[63] == self.dict_map_pieces[piece_rook.lower()]
      bstate[63] = value_rook_castled+8

    value_ep_square = 8 # hard-coded
    if self.board.ep_square is not None:
      assert bstate[self.board.ep_square] == 0
      bstate[self.board.ep_square] = value_ep_square
    bstate = bstate.reshape(8,8)

    # binary state
    state = np.zeros((5,8,8), np.uint8)

    # 0-3 columns to binary
    for i in range(4):
      state[i] = (bstate>>4-1-i)&1 # convert to binary, then keep rightmost bit

    # 4th column is who's turn it is
    state[4] = self.board.turn*1.0

    # 257 bits according to readme
    return state

  def edges(self):
    return list(self.board.legal_moves)

if __name__ == "__main__":
  s = State()


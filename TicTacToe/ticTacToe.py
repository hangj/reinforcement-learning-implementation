#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os, sys
import math

BOARD_ROWS = 3
BOARD_COLS = 3


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros(( BOARD_ROWS, BOARD_COLS ))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first, 1 for p1, -1 for p2, fill playerSymbol in board
        self.playerSymbol = 1

    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def getHash(self):
        self.boardHash = str(self.board.reshape( BOARD_ROWS*BOARD_COLS ))
        return self.boardHash

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            s = sum(self.board[i, :])
            if s == 3:
                self.isEnd = True
                return 1
            if s == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            s = sum(self.board[:, i])
            if s == 3:
                self.isEnd = True
                return 1
            if s == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS-1-i] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available position
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        return availablePositionsFuck(self.board)
        # positions = []
        # for i in range(self.board.shape[0]):
        #     for j in range(self.board.shape[1]):
        #         if self.board[i, j] == 0:
        #             positions.append((i, j))
        # return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    def play(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and update board state
                self.updateState(p1_action)
                # board_hash = self.getHash()
                self.p1.addState(self.board.copy())

                # check board status if it is end
                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    # board_hash = self.getHash()
                    self.p2.addState(self.board.copy())

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

        self.p1.savePolicy()
        self.p2.savePolicy()

    # play with human
    def play2(self):
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and update board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break
            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.exp_rate = exp_rate # possibility that take random action
        self.states = [] # record all positions taken
        self.learningRate = 0.2
        self.decay_gamma = 0.9
        self.states_value = {} # state -> value

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS*BOARD_ROWS))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        action = None
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = self.states_value.get(next_boardHash)
                value = 0 if value is None else value
                # print("value:", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    def genAllSimilarStatesHash(self, state):
        st = state
        arr = [self.getHash(np.rot90(st, i)) for i in range(4)] # 正面 4 个角度
        arr.extend([self.getHash(np.fliplr(np.rot90(st, i))) for i in range(4)]) # 背面 4 个角度

        return set(arr)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            hsh = self.getHash(st)
            if self.states_value.get(hsh) is None:
                self.states_value[hsh] = 0
            self.states_value[hsh] += self.learningRate * (self.decay_gamma * reward - self.states_value[hsh])
            reward = self.states_value[hsh]

            # 同一形状的棋盘，奖励也一样
            for hsh in self.genAllSimilarStatesHash(st):
                self.states_value[hsh] = reward

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()
    def loadPolicy(self, file=None):
        file = file or 'policy_' + str(self.name)
        if not os.path.exists(file):
            return

        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions, current_board, symbol):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update state value
    def feedReward(self, reward):
        pass
    def reset(self):
        pass


class Color:
    color_ = 2
    def __init__(self):
        Color.color_ = Color.color_ + 1
        self.color = Color.color_

def colorTheBoard(b):
    # b == b.T                      => b[x][y] == b[y][x]                                     # transpose 也就是 「 \ 」轴对称
    # b == np.fliplr(np.rot90(b))   => b[x][y] == b[b.shape[1] - 1 - y][b.shape[0] - 1 - x]   # 「 / 」轴对称
    # b == np.flipud(b)             => b[x][y] == b[b.shape[0] - 1 - x][y]                    # up down 翻转，也就是「 — 」轴对称
    # b == np.fliplr(b)             => b[x][y] == b[x][b.shape[1] - 1 - y]                    # up down 翻转，也就是「 | 」轴对称
    # b == np.rot90(np.rot90(b))    => b[x][y] == b[b.shape[0] - 1 - x][b.shape[1] - 1 - y]   # 中心对称

    colorB = b.tolist() # get a copy of the array data as a (nested) Python list

    if (b == b.T).all():
        for x in range(b.shape[0]):
            for y in range(x, b.shape[1]):
                if colorB[x][y] == 0:
                    colorB[x][y] = colorB[y][x] = Color()
                elif type(colorB[x][y]) == Color:
                    colorB[x][y].color = colorB[y][x].color

    if (b == np.fliplr(np.rot90(b))).all():
        for x in range(b.shape[0]):
            for y in range(b.shape[1] - x):
                if colorB[x][y] == 0:
                    colorB[x][y] = colorB[b.shape[1] - 1 - y][b.shape[0] - 1 - x] = Color()
                elif type(colorB[x][y]) == Color:
                    colorB[x][y].color = colorB[b.shape[1] - 1 - y][b.shape[0] - 1 - x].color

    if (b == np.flipud(b)).all():
        for x in range(math.ceil(b.shape[0]/2)):
            for y in range(b.shape[1]):
                if colorB[x][y] == 0:
                    colorB[x][y] = colorB[b.shape[0] - 1 - x][y] = Color()
                elif type(colorB[x][y]) == Color:
                    colorB[x][y].color = colorB[b.shape[0] - 1 - x][y].color

    if (b == np.fliplr(b)).all():
        for x in range(b.shape[0]):
            for y in range(math.ceil(b.shape[1]/2)):
                if colorB[x][y] == 0:
                    colorB[x][y] = colorB[x][b.shape[1] - 1 - y] = Color()
                elif type(colorB[x][y]) == Color:
                    colorB[x][y].color = colorB[x][b.shape[1] - 1 - y].color

    if (b == np.rot90(np.rot90(b))).all():
        for x in range(math.ceil(b.shape[0]/2)):
            for y in range(b.shape[1]):
                if colorB[x][y] == 0:
                    colorB[x][y] = colorB[b.shape[0] - 1 - x][b.shape[1] - 1 - y] = Color()
                elif type(colorB[x][y]) == Color:
                    colorB[x][y].color = colorB[b.shape[0] - 1 - x][b.shape[1] - 1 - y].color

    # 完全不对称的情况
    for x in range(b.shape[0]):
        for y in range(b.shape[1]):
            if colorB[x][y] == 0:
                colorB[x][y] = Color()

    # for x in range(b.shape[0]):
    #     for y in range(b.shape[1]):
    #         if type(colorB[x][y]) == Color:
    #             colorB[x][y] = colorB[x][y].color
    # print(np.array(colorB))
    # [[ b[x][y] for y in range(b.shape[1])] for x in range(b.shape[0])]

    return colorB

def availablePositionsFuck(board):
    b = colorTheBoard(board)
    # print('colored:\n', b)
    kv = {}
    for x in range(len(b)):
        for y in range(len(b[0])):
            if type(b[x][y]) == Color:
                if kv.get( b[x][y].color ) == None:
                    kv[ b[x][y].color ] = (x,y)
    return list(kv.values())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {0} test|train|play".format(sys.argv[0]))
        exit(1)

    if sys.argv[1] == 'test':
        b = np.zeros((3,3))
        b[0][0] = 1
        b[0][1] = -1
        print(b)
        # print(b[0][2] == 0)
        c = availablePositionsFuck(b)
        print(c)


    if sys.argv[1] == 'train':
        # training
        p1 = Player("p1")
        p2 = Player("p2")

        # p1.loadPolicy()
        # p2.loadPolicy()

        st = State(p1, p2)
        print("training...")
        st.play(int(sys.argv[2]))

        # print(p1.states_value)
        # print(p2.states_value)

    if sys.argv[1] == 'play':
        # play with human
        p1 = Player("computer", exp_rate=0)
        p1.loadPolicy("policy_p1")

        p2 = HumanPlayer("human")

        st = State(p1, p2)
        st.play2()




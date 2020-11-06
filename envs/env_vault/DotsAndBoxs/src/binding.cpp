#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dotsAndBoxes.h"

GameBoard * board = NULL;
int boardSize = 3;
int numInitMoves = 5;

void setParams(int boardDimension, int numberOfInitialMoves) {
    boardSize = boardDimension;
    numInitMoves = numberOfInitialMoves;
}

void reset() {
    if(!board) {
        printf("New\n");
        board = new GameBoard(boardSize);
    }
    board->initRandom(numInitMoves);
}

double step(int move) {
    if(!board)
        reset();
    
    bool endTurn = false;
    double score = board->scoreMove(move, endTurn);
    //If you make an illegal move turn will end and you get worst score.
    if(endTurn)
        board->OpponentMove();
    return score;
}

std::vector<int> state() {
    if(!board)
        reset();
    return board->serializeBoard();
}

bool done() {
    if(board->terminal()) {
        delete board;
        return true;
    }
    return false;
}

void print() {
    std::cout << "Dots and Boxes" << std::endl;
    if(board) {
        board->printBoard();
        int player1, player2;
        board->getScores(player1, player2);
        std::cout << "Player1: " << player1 << " Player2: " << player2 << std::endl;
    }
}

namespace py = pybind11;

PYBIND11_MODULE(dotsandboxes, m) {
    m.doc() = "Dots and Boxes game for Exalearn!";

    m.def("setParams", &setParams, "Sets the board size and number of initial moves.");
    m.def("reset", &reset, "Reset");
    m.def("step", &step, "Step");
    m.def("state", &state, "State");
    m.def("print", &print, "Print");
    m.def("done", &done, "Done");
    
    m.attr("__version__") = "dev";
}
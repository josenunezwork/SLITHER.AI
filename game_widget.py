from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QBrush, QColor
from PyQt5.QtCore import Qt

class GameWidget(QWidget):
    def __init__(self, game):
        super().__init__()
        self.game = game

    def paintEvent(self, event):
        painter = QPainter(self)
        self.draw_game(painter)

    def draw_game(self, painter):
        for snake in self.game.snakes:
            if not snake.is_alive:
                continue
            painter.setBrush(QBrush(snake.color))
            painter.setPen(Qt.NoPen)
            for segment in snake.body:
                painter.drawEllipse(segment[0] - snake.segment_size // 2, segment[1] - snake.segment_size // 2, snake.segment_size, snake.segment_size)
        painter.setBrush(QBrush(Qt.green))
        for food in self.game.food:
            painter.drawEllipse(food[0] - 3, food[1] - 3, 6, 6)

import sys
from PyQt5.QtWidgets import QApplication
from slitherio import SlitherIOGame

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = SlitherIOGame()
    sys.exit(app.exec_())
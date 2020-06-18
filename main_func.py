from testUI import Ui_MainWindow
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from generate_caption import *
from encode_Image import *
from peom_generate.generate import *


def prediction_img(img):
    encoded_dic = encode_images(img)
    txt = generate(encoded_dic)
    return txt


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyApp, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.show_vis)
        self.get_capt.clicked.connect(self.show_cap)
        self.prefix.clicked.connect(self.prefix_)
        self.img_name = ''
        self.generate_poem_state = 1   # 1：输入首句得到古诗   2： 藏头

    def show_vis(self):
        self.img_name, _ = QFileDialog.getOpenFileName(self, "show picture", "", "*.jpg;;*.png;;All Files(*)")
        self.Pic_show.setPixmap(QPixmap(self.img_name))
        self.Pic_show.setScaledContents(True)

    def show_cap(self):
        txt = prediction_img(self.img_name)
        self.Caption.setText(txt)
        self.Caption.adjustSize()

    def prefix_(self):
        self.label_3.setText(str(self.prefix.isChecked()))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
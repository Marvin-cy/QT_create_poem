# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(60, 390, 131, 31))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 60, 121, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(370, 70, 121, 31))
        self.label_2.setObjectName("label_2")
        self.Pic_show = QtWidgets.QLabel(self.centralwidget)
        self.Pic_show.setGeometry(QtCore.QRect(30, 110, 231, 211))
        self.Pic_show.setAutoFillBackground(False)
        self.Pic_show.setText("")
        self.Pic_show.setObjectName("Pic_show")
        self.get_capt = QtWidgets.QPushButton(self.centralwidget)
        self.get_capt.setGeometry(QtCore.QRect(320, 390, 101, 31))
        self.get_capt.setObjectName("get_capt")
        self.Caption = QtWidgets.QLabel(self.centralwidget)
        self.Caption.setGeometry(QtCore.QRect(300, 160, 211, 161))
        self.Caption.setText("")
        self.Caption.setObjectName("Caption")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 450, 371, 51))
        self.label_3.setObjectName("label_3")
        self.first_sen = QtWidgets.QPushButton(self.centralwidget)
        self.first_sen.setGeometry(QtCore.QRect(80, 530, 80, 23))
        self.first_sen.setObjectName("first_sen")
        self.prefix = QtWidgets.QPushButton(self.centralwidget)
        self.prefix.setGeometry(QtCore.QRect(320, 530, 80, 23))
        self.prefix.setObjectName("prefix")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "show picture"))
        self.label.setText(_translate("MainWindow", "Original Picture"))
        self.label_2.setText(_translate("MainWindow", "Caption"))
        self.get_capt.setText(_translate("MainWindow", "get caption"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.first_sen.setText(_translate("MainWindow", "shouju"))
        self.prefix.setText(_translate("MainWindow", "cangtou"))

from bs4 import BeautifulSoup

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout,QLineEdit,QTextBrowser


# File load
"""file_name = "C:/Users/samsung/Desktop/ioc/iocbucket_2bc439723ce857f7039c66e1f679bcd2d0cc6e3a_platinum apt (family).ioc"
docs = BeautifulSoup(open(file_name),"html.parser")
print(docs)"""

# Make window
class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.lbl1 = QLabel('Enter File path')

        self.le = QLineEdit()
        self.le.returnPressed.connect(self.ioc_finder)

        self.tb = QTextBrowser()
        self.tb.setAcceptRichText(True)
        self.tb.setOpenExternalLinks(True)

        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl1)
        vbox.addWidget(self.le)
        vbox.addWidget(self.tb)
        vbox.addStretch()

        self.setLayout(vbox)
        self.setWindowTitle('IOC')
        self.setGeometry(600, 600, 600, 400)
        self.show()

    def ioc_finder(self):
        text = self.le.text()
        docs = BeautifulSoup(open(text), "html.parser")
        print(docs)
        #self.tb.append(docs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
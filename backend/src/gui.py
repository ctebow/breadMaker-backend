import sys
from PySide6.QtWidgets import QApplication, QPushButton, QInputDialog, QLineEdit, QVBoxLayout, QLabel, QWidget, QDialog
from PySide6.QtCore import Qt

class Form(QDialog):

    def __init__(self, parent=None):
        # initialize parent and set window title
        super(Form, self).__init__(parent)
        self.setWindowTitle("My Form")
        # create widgets
        self.edit = QLineEdit("Write my name here..")
        self.button = QPushButton("Show Greetings")

        # Define the layout and add widgets
        layout = QVBoxLayout(self)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)

        # connect to greetings function
        self.button.clicked.connect(self.greetings)

    def greetings(self):
        print(f'Hello {self.edit.text()}')



class ClickableLabel(QLabel):

    def __init__(self, value="100"):

        super().__init__(value)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid black; padding: 10px")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            new_val, ok = QInputDialog.getText(self, "Edit Resistor", "Enter new resistance:")
            if ok and new_val:
                self.setText(new_val)

class CircuitEditor(QWidget):
    
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Resistor Editor")
        self.resistor = ClickableLabel("220")

        layout = QVBoxLayout()
        layout.addWidget(self.resistor)
        self.setLayout(layout)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = CircuitEditor()

    window.show()

    sys.exit(app.exec())
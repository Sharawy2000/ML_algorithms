import sys
import matplotlib.pyplot as plt
# import GUI library
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
import qdarkstyle
import KNN
# import SVM
# import Linear_regression


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        # Remove the window frame
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Set the dark mode style
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # Set window title and size
        self.setWindowTitle('Machine Learning')
        self.setGeometry(0, 0, 1340, 800)

        # Set window icon
        self.setWindowIcon(QIcon('Images/logo.png'))

        # Center window on screen
        self.center()

        # create widgets
        self.create_widgets()

    def center(self):
        # Get the screen geometry
        screen = QDesktopWidget().screenGeometry()

        # Calculate the center point
        center_x = (screen.width() - self.width()) // 2
        center_y = (screen.height() - self.height()) // 2

        # Move the window to the center
        self.move(center_x, center_y)

    def create_widgets(self):
        # layout for the main window
        layout = QVBoxLayout(self)

        # Add a label
        self.label = QLabel("Machine Learning Algorithms", self)
        self.label.setStyleSheet("font-size: 20pt; font-weight: bold;margin-top:50pt")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label, alignment=Qt.AlignTop)

        self.combo_algorithms = QComboBox(self)
        self.combo_algorithms.resize(500, 50)
        self.combo_algorithms.move(430, 300)
        self.combo_algorithms.addItems(['KNN Regression', 'SVM','Linear Regression','Decision Tree Classification'
                                           ,'Decision Tree Regression'])
        default_index = self.combo_algorithms.findText('KNN Regression')
        if default_index != -1:
            self.combo_algorithms.setCurrentIndex(default_index)

        self.apply_Algo_button = QPushButton('Apply Algorithm ', self)
        self.apply_Algo_button.setStyleSheet("font-size: 12pt;")
        self.apply_Algo_button.resize(175, 50)
        self.apply_Algo_button.move(580, 450)
        self.apply_Algo_button.clicked.connect(self.apply_algorithm)


        self.close_button = QPushButton('Quit', self)
        self.close_button.setStyleSheet("font-size: 12pt;")
        self.close_button.resize(175, 50)
        self.close_button.move(580, 600)
        self.close_button.clicked.connect(self.close)

        self.dark = QRadioButton("Dark mode", self)
        self.dark.move(1200, 700)
        self.dark.setChecked((True))
        self.dark.toggled.connect(self.setDark)

        self.light = QRadioButton("Light mode", self)
        self.light.move(1200, 730)
        self.light.toggled.connect(self.setLight)

    # Define a method to select an image

    def enhance_algorithms(self):

        if selected_item == 'KNN Regression':
            # Apply blur algorithm
            print("Applying KNN Regression Algorithm")
            KNN_algo()
        elif selected_item == 'SVM':
            # Apply edge enhance algorithm
            print("Applying SVM Algorithm")
            # contrast_algo(self.filename)

        elif selected_item == 'Linear Regression':
            # Apply brightness algorithm
            print("Applying Linear Regression Algorithm")
            # brightness_algo(self.filename)

        elif selected_item == 'Decision Tree Classification':
            # Apply brightness algorithm
            print("Applying Decision Tree Classification Algorithm")
            # brightness_algo(self.filename)

        elif selected_item == 'Decision Tree Regression':
            # Apply brightness algorithm
            print("Applying Decision Tree Regression Algorithm")
            # brightness_algo(self.filename)

    def apply_algorithm(self):
        global selected_item
        selected_item = self.combo_algorithms.currentText()
        self.enhance_algorithms()

    # Define a function for setting the dark theme
    def setDark(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Define a function for setting the light theme
    def setLight(self):

        self.setStyleSheet('')

def KNN_algo():
    KNN.main()


if __name__ == '__main__':
    # Create a QApplication instance
    app = QApplication(sys.argv)
    # Create an instance of our window
    window = MyWindow()

    # Show the window
    window.show()
    # Start the event loop and exit the application when the loop is finished
    sys.exit(app.exec_())


import sys

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QAction,
    QIcon,
    QImage,
    QMouseEvent,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
)


class ImageProcessor:
    def __init__(self):
        self.image = None
        self.history = []
        self.history_index = -1

    def load_image(self, file_path):
        self.image = cv2.imread(file_path)
        if self.image is None:
            return False
        self.add_to_history()
        return True

    def save_image(self, file_path):
        if self.image is None:
            return
        cv2.imwrite(file_path, self.image)

    def add_to_history(self):
        self.history = self.history[: self.history_index + 1]
        self.history.append(self.image.copy())
        self.history_index += 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.image = self.history[self.history_index].copy()
            return self.image
        return None

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.image = self.history[self.history_index].copy()
            return self.image
        return None

    def apply_grayscale(self, scale=1):
        if self.image is None:
            return None
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.merge((gray_image,) * 3)
        self.add_to_history()
        return self.image

    def apply_blur(self, kernel_size):
        if self.image is None:
            return None
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        self.add_to_history()
        return self.image

    def apply_canny(self, threshold1, threshold2):
        if self.image is None:
            return None
        self.image = cv2.Canny(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY),
            threshold1,
            threshold2,
        )
        self.add_to_history()
        return self.image

    def rotate_image(self, angle):
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.image = cv2.warpAffine(self.image, M, (w, h))
        self.add_to_history()
        return self.image

    def resize_image(self, width, height):
        self.image = cv2.resize(self.image, (width, height))
        self.add_to_history()
        return self.image

    def change_brightness_contrast(self, brightness=0, contrast=0):
        self.image = cv2.convertScaleAbs(
            self.image,
            alpha=(contrast / 127 + 1),
            beta=brightness,
        )
        self.add_to_history()
        return self.image

    def draw_text(self, text, x, y, font_scale, color):
        cv2.putText(
            self.image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
        )
        self.add_to_history()
        return self.image

    def draw_rectangle(self, x, y, w, h, color):
        cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
        self.add_to_history()
        return self.image

    def draw_line(self, x1, y1, x2, y2, color):
        cv2.line(self.image, (x1, y1), (x2, y2), color, 2)
        self.add_to_history()
        return self.image

    def draw_circle(self, center_x, center_y, radius, color):
        cv2.circle(self.image, (center_x, center_y), radius, color, 2)
        self.add_to_history()
        return self.image

    def detect_face(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        )

        angles = [0, 15, -15, 30, -30, 45, -45]
        faces = []

        for angle in angles:
            rotated_gray = gray
            if angle != 0:
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_gray = cv2.warpAffine(gray, M, (w, h))

            detected_faces = face_cascade.detectMultiScale(
                rotated_gray,
                scaleFactor=1.3,
                minNeighbors=5,
            )
            if len(detected_faces) > 0:
                faces.extend(detected_faces)

        if len(faces) > 0:
            return "��������� �������"
        else:
            return "������� �� ���������"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Обработка изображения OpenCV")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("ico/ico.png"))

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.setCentralWidget(self.image_label)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.createActions()
        self.createMenus()
        self.createToolBars()

        self.setStyleSheet(
            """
                    QMainWindow {
                        background-color: #f0f0f0;
                    }
                    QLabel {
                        background-color: #ffffff;
                        border: 1px solid #cccccc;
                    }
                    QMenuBar {
                        background-color: #666666;
                        color: white;
                    }
                    QMenuBar::item {
                        background-color: #666666;
                        color: white;
                    }
                    QMenuBar::item::selected {
                        background-color: #888888;
                    }
                    QMenu {
                        background-color: #666666;
                        color: white;
                    }
                    QMenu::item {
                        background-color: #666666;
                        color: white;
                    }
                    QMenu::item::selected {
                        background-color: #888888;
                    }
                    QAction {
                        color: white;
                    }
                    QAction:hover {
                        background-color: #888888;
                    }
                """,
        )

    def createActions(self):
        self.open_act = QAction(
            QIcon("ico/open.png"),
            "&�������...",
            self,
            triggered=self.openImage,
            shortcut="Ctrl+O",
        )
        self.save_act = QAction(
            QIcon("ico/save.png"),
            "&��������� ���...",
            self,
            triggered=self.saveImage,
            shortcut="Ctrl+S",
        )
        self.exit_act = QAction(
            QIcon("ico/exit.png"),
            "&����� �� ���������",
            self,
            triggered=self.close,
            shortcut="Ctrl+Q",
        )
        self.grayscale_act = QAction(
            "&�������� ������",
            self,
            triggered=self.applyGrayscale,
        )
        self.blur_act = QAction("&��������", self, triggered=self.applyBlur)
        self.canny_act = QAction("&�������� ������", self, triggered=self.applyCanny)
        self.rotate_act = QAction("&�������", self, triggered=self.applyRotate)
        self.resize_act = QAction("&�������� ������", self, triggered=self.applyResize)
        self.brightness_contrast_act = QAction(
            "&�������/��������",
            self,
            triggered=self.applyBrightnessContrast,
        )
        self.text_act = QAction("&�������� �����", self, triggered=self.addText)
        self.draw_line_act = QAction("&���������� �����", self, triggered=self.drawLine)
        self.draw_circle_act = QAction(
            "&���������� ����",
            self,
            triggered=self.drawCircle,
        )
        self.rectangle_act = QAction(
            "&���������� �������������",
            self,
            triggered=self.drawRectangle,
        )
        self.detect_face_act = QAction(
            "&���������� �����������",
            self,
            triggered=self.detectFace,
        )
        self.undo_act = QAction(
            QIcon("ico/undo.png"),
            "��������",
            self,
            triggered=self.undoAction,
            shortcut="Ctrl+Z",
        )
        self.redo_act = QAction(
            QIcon("ico/redo.png"),
            "���������",
            self,
            triggered=self.redoAction,
            shortcut="Ctrl+Y",
        )
        self.help_act = QAction(
            QIcon("ico/info.png"),
            "&������",
            self,
            triggered=self.show_help,
        )
        self.about_act = QAction(
            QIcon("ico/proga.png"),
            "&� ���������",
            self,
            triggered=self.show_about,
        )

    def createMenus(self):
        self.file_menu = self.menuBar().addMenu("&����")
        self.file_menu.addAction(self.open_act)
        self.file_menu.addAction(self.save_act)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_act)

        self.detect_menu = self.menuBar().addAction(
            "&���������� �����������",
            self.detectFace,
        )

        self.edit_menu = self.menuBar().addMenu("&�����������")
        self.edit_menu.addAction(self.grayscale_act)
        self.edit_menu.addAction(self.blur_act)
        self.edit_menu.addAction(self.canny_act)
        self.edit_menu.addAction(self.rotate_act)
        self.edit_menu.addAction(self.resize_act)
        self.edit_menu.addAction(self.brightness_contrast_act)
        self.edit_menu.addAction(self.text_act)
        self.edit_menu.addAction(self.draw_line_act)
        self.edit_menu.addAction(self.draw_circle_act)
        self.edit_menu.addAction(self.rectangle_act)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.undo_act)
        self.edit_menu.addAction(self.redo_act)

        self.help_menu = self.menuBar().addMenu("&������")
        self.help_menu.addAction(self.help_act)
        self.help_menu.addAction(self.about_act)

    def createToolBars(self):
        toolbar = QToolBar("��� �����������", self)

        toolbar.addAction(self.undo_act)
        toolbar.addAction(self.redo_act)

        self.addToolBar(toolbar)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        context_menu.addAction(self.open_act)
        context_menu.addAction(self.save_act)
        context_menu.addSeparator()
        context_menu.addAction(self.grayscale_act)
        context_menu.addAction(self.blur_act)
        context_menu.addAction(self.canny_act)
        context_menu.addAction(self.rotate_act)
        context_menu.addAction(self.resize_act)
        context_menu.addAction(self.brightness_contrast_act)
        context_menu.addAction(self.text_act)
        context_menu.addAction(self.rectangle_act)
        context_menu.addSeparator()
        context_menu.addAction(self.undo_act)
        context_menu.addAction(self.redo_act)
        context_menu.exec(event.globalPos())

    def show_help(self):
        QMessageBox.information(
            self,
            "������",
            "����������� �� �����: andrey.paydak@mail.ru",
        )

    def show_about(self):
        QMessageBox.information(
            self,
            "� ���������",
            "��������� ��� ��������� �����������\n������ 1.2",
        )

    def detectFace(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return
        result = self.processor.detect_face()
        self.statusBar.showMessage(result)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        x, y = int(pos.x()), int(pos.y())

        self.statusBar.showMessage(f"���������� �������: ({x}, {y})")

    def openImage(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Открыть (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            if not self.processor.load_image(file_path):
                QMessageBox.warning(
                    self,
                    "������",
                    "�� ������� ��������� �����������. ��������� ���� � ����� � ��� �����������.",
                )
            else:
                self.displayImage(self.processor.image)
                self.statusBar.showMessage(f"����������� ���������: {file_path}")

    def saveImage(self):
        if self.processor.image is None:
            QMessageBox.warning(self, "��������������", "���������� ��������� ����")
            return
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("����������� (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.processor.save_image(file_path)
            self.statusBar.showMessage(f"����������� ���������: {file_path}")

    def displayImage(self, image):
        qformat = QImage.Format.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format.Format_RGBA8888
            else:
                qformat = QImage.Format.Format_RGB888
        img = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        img = img.rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(img))

    def show_warning(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()

    def applyGrayscale(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return
        image = self.processor.apply_grayscale()
        self.displayImage(image)
        self.statusBar.showMessage("�������� ������ ���������")

    def applyBlur(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return
        kernel_size, ok = QInputDialog.getInt(
            self,
            "��������",
            "������ ���� (�������� �����):",
            min=1,
            max=49,
            step=2,
        )
        if ok:
            self.displayImage(self.processor.apply_blur(kernel_size))
            self.statusBar.showMessage(
                f"�������� ��������� � �������� ���� {kernel_size}",
            )

    def applyCanny(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return
        threshold1, ok1 = QInputDialog.getInt(
            self,
            "�������� ������",
            "����� 1:",
            min=0,
            max=255,
        )
        threshold2, ok2 = QInputDialog.getInt(
            self,
            "�������� ������",
            "����� 2:",
            min=0,
            max=255,
        )
        if ok1 and ok2:
            self.displayImage(self.processor.apply_canny(threshold1, threshold2))
            self.statusBar.showMessage(
                f"�������� �������� � �������� {threshold1} � {threshold2}",
            )

    def applyRotate(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return
        angle, ok = QInputDialog.getInt(
            self,
            "�������",
            "���� ��������:",
            min=-360,
            max=360,
        )
        if ok:
            self.displayImage(self.processor.rotate_image(angle))
            self.statusBar.showMessage(f"����������� ��������� �� {angle} ��������")

    def applyResize(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return
        width, ok1 = QInputDialog.getInt(self, "�������� ������", "������:", min=1)
        height, ok2 = QInputDialog.getInt(self, "�������� ������", "������:", min=1)
        if ok1 and ok2:
            self.displayImage(self.processor.resize_image(width, height))
            self.statusBar.showMessage(
                f"������ ����������� ������� �� {width}x{height}",
            )

    def applyBrightnessContrast(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("�������/��������")
        layout = QVBoxLayout()

        form_layout = QFormLayout()
        brightness = QSlider(Qt.Horizontal)
        brightness.setRange(-255, 255)
        contrast = QSlider(Qt.Horizontal)
        contrast.setRange(-127, 127)
        form_layout.addRow("�������:", brightness)
        form_layout.addRow("��������:", contrast)

        apply_button = QPushButton("���������")
        apply_button.clicked.connect(dialog.accept)

        layout.addLayout(form_layout)
        layout.addWidget(apply_button)
        dialog.setLayout(layout)

        if dialog.exec() == QDialog.accepted:
            image = self.processor.change_brightness_contrast(
                brightness.value(),
                contrast.value(),
            )
            self.displayImage(image)
            self.statusBar.showMessage(
                f"������� �������� �� {brightness}, �������� �� {contrast}",
            )

    def choose_color(self, button):
        color_dialog = QColorDialog(self)
        color_dialog.setWindowTitle("����� �����")
        if color_dialog.exec() == QColorDialog.accepted:
            color = color_dialog.selectedColor()
            r, g, b, _ = color.getRgb()
            bgr_color = (b, g, r)
            button.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")
            return bgr_color

    def addText(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("�������� �����")
        layout = QVBoxLayout()

        form_layout = QFormLayout()
        text_input = QLineEdit()
        x = QSpinBox()
        x.setRange(0, self.processor.image.shape[1])
        y = QSpinBox()
        y.setRange(0, self.processor.image.shape[0])
        font_scale = QSpinBox()
        font_scale.setRange(1, 10)
        color_button = QPushButton("����")

        color_button.clicked.connect(lambda: self.choose_color(color_button))

        form_layout.addRow("�����:", text_input)
        form_layout.addRow("X:", x)
        form_layout.addRow("Y:", y)
        form_layout.addRow("������ ������:", font_scale)
        form_layout.addRow("����:", color_button)

        apply_button = QPushButton("���������")
        apply_button.clicked.connect(dialog.accept)

        layout.addLayout(form_layout)
        layout.addWidget(apply_button)
        dialog.setLayout(layout)

        if dialog.exec() == QDialog.accepted:
            text = text_input.text()
            color = color_button.palette().button().color().getRgb()[:3]
            bgr_color = (color[2], color[1], color[0])
            image = self.processor.draw_text(
                text,
                x.value(),
                y.value(),
                font_scale.value(),
                bgr_color,
            )
            self.displayImage(image)
            self.statusBar.showMessage(
                f'����� "{text}" �������� � ��������� ������ {font_scale.value()}',
            )

    def drawLine(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("���������� �����")
        layout = QVBoxLayout()

        form_layout = QFormLayout()
        x1 = QSpinBox()
        x1.setRange(0, self.processor.image.shape[1])
        y1 = QSpinBox()
        y1.setRange(0, self.processor.image.shape[0])
        x2 = QSpinBox()
        x2.setRange(0, self.processor.image.shape[1])
        y2 = QSpinBox()
        y2.setRange(0, self.processor.image.shape[0])
        color_button = QPushButton("����")

        color_button.clicked.connect(lambda: self.choose_color(color_button))

        form_layout.addRow("X1:", x1)
        form_layout.addRow("Y1:", y1)
        form_layout.addRow("X2:", x2)
        form_layout.addRow("Y2:", y2)
        form_layout.addRow("����:", color_button)

        apply_button = QPushButton("���������")
        apply_button.clicked.connect(dialog.accept)

        layout.addLayout(form_layout)
        layout.addWidget(apply_button)
        dialog.setLayout(layout)

        if dialog.exec() == QDialog.accept:
            color = color_button.palette().button().color().getRgb()[:3]
            bgr_color = (color[2], color[1], color[0])
            image = self.processor.draw_line(
                x1.value(),
                y1.value(),
                x2.value(),
                y2.value(),
                bgr_color,
            )
            self.displayImage(image)
            self.statusBar.showMessage("����� ����������")

    def drawCircle(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("���������� ����")
        layout = QVBoxLayout()

        form_layout = QFormLayout()
        center_x = QSpinBox()
        center_x.setRange(0, self.processor.image.shape[1])
        center_y = QSpinBox()
        center_y.setRange(0, self.processor.image.shape[0])
        radius = QSpinBox()
        radius.setRange(1, 100)
        color_button = QPushButton("����")

        color_button.clicked.connect(lambda: self.choose_color(color_button))

        form_layout.addRow("����� X:", center_x)
        form_layout.addRow("����� Y:", center_y)
        form_layout.addRow("������:", radius)
        form_layout.addRow("����:", color_button)

        apply_button = QPushButton("���������")
        apply_button.clicked.connect(dialog.accept)

        layout.addLayout(form_layout)
        layout.addWidget(apply_button)
        dialog.setLayout(layout)

        if dialog.exec() == QDialog.Accepted:
            color = color_button.palette().button().color().getRgb()[:3]
            bgr_color = (color[2], color[1], color[0])
            image = self.processor.draw_circle(
                center_x.value(),
                center_y.value(),
                radius.value(),
                bgr_color,
            )
            self.displayImage(image)
            self.statusBar.showMessage("���� ���������")

    def drawRectangle(self):
        if self.processor.image is None:
            QMessageBox.warning(
                self,
                "��������������",
                "���������� ��������� ����, ����� ���� ������ ��������������",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("���������� �������������")
        layout = QVBoxLayout()

        form_layout = QFormLayout()
        x = QSpinBox()
        x.setRange(0, self.processor.image.shape[1])
        y = QSpinBox()
        y.setRange(0, self.processor.image.shape[0])
        w = QSpinBox()
        w.setRange(1, self.processor.image.shape[1])
        h = QSpinBox()
        h.setRange(1, self.processor.image.shape[0])
        color_button = QPushButton("����")

        color_button.clicked.connect(lambda: self.choose_color(color_button))

        form_layout.addRow("X:", x)
        form_layout.addRow("Y:", y)
        form_layout.addRow("������:", w)
        form_layout.addRow("������:", h)
        form_layout.addRow("����:", color_button)

        apply_button = QPushButton("���������")
        apply_button.clicked.connect(dialog.accept)

        layout.addLayout(form_layout)
        layout.addWidget(apply_button)
        dialog.setLayout(layout)

        if dialog.exec() == QDialog.accept:
            color = color_button.palette().button().color().getRgb()[:3]
            bgr_color = (color[2], color[1], color[0])
            image = self.processor.draw_rectangle(
                x.value(),
                y.value(),
                w.value(),
                h.value(),
                bgr_color,
            )
            self.displayImage(image)
            self.statusBar.showMessage(
                f"������������� ��������� � ������� {w.value()} � ������� {h.value()}",
            )

    def undoAction(self):
        image = self.processor.undo()
        if image is not None:
            self.displayImage(image)
            self.statusBar.showMessage("������ ���������� ��������")

    def redoAction(self):
        image = self.processor.redo()
        if image is not None:
            self.displayImage(image)
            self.statusBar.showMessage("������ ���������� ��������")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Вы уверены, что хотите закрыть приложение?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())

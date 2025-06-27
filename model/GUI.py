import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QFileDialog, QScrollArea,
                             QProgressBar, QMessageBox, QSplitter, QSizePolicy, QFrame,
                             QTextEdit)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import cv2
import re
from collections import defaultdict
from ultralytics import YOLO


class InferenceThread(QThread):
    progress_updated = pyqtSignal(int)
    result_updated = pyqtSignal(str)
    inference_finished = pyqtSignal()

    def __init__(self, model, image_files, dataset_name, parent=None):
        super().__init__(parent)
        self.model = model
        self.image_files = image_files
        self.dataset_name = dataset_name
        self.inference_results = {}
        self.detection_details = {}
        self.running = True
        self.filter_classes = None
        self.filter_threshold = 0.5  # 默认过滤阈值
        self.class_counts = defaultdict(int)
        self.total_detections = 0

    def run(self):
        total = len(self.image_files)
        if total == 0:
            return

        # 第一步：收集类别统计并确定过滤规则
        self.collect_class_statistics()
        self.determine_filtering()

        # 第二步：进行推理并应用过滤
        for i, img_file in enumerate(self.image_files):
            if not self.running:
                break

            # 读取图像并进行推理
            image = cv2.imread(img_file)
            results = self.model(image)

            # 获取带标注的结果图像
            result_image = results[0].plot()

            # 转换为RGB格式用于显示
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            self.inference_results[img_file] = result_image_rgb

            # 获取检测结果详情
            detections = results[0].boxes
            detection_text = "<div style='font-size: 30px; color: #999;'>无检测结果</div>"

            if detections is not None and len(detections) > 0:
                detection_list = []
                for j in range(len(detections)):
                    class_id = int(detections.cls[j])
                    confidence = float(detections.conf[j])
                    class_name = self.model.names[class_id]

                    # 应用过滤规则
                    if self.filter_classes is None or class_name in self.filter_classes:
                        detection_list.append(f"{class_name} {confidence:.2f}")

                if detection_list:
                    detection_text = ""
                    for item in detection_list:
                        detection_text += f"<div style='font-size: 30px; font-weight: bold; margin-bottom: 10px;'>{item}</div>"
                    if len(detection_list) > 5:
                        detection_text += f"<div style='font-size: 26px; color: #666;'>...共{len(detection_list)}个检测</div>"
                else:
                    detection_text = "<div style='font-size: 30px; color: #999;'>无相关检测结果</div>"

            self.detection_details[img_file] = detection_text

            self.progress_updated.emit(int((i + 1) / total * 100))
            self.result_updated.emit(detection_text)

        self.inference_finished.emit()

    def collect_class_statistics(self):
        """收集所有图像的类别统计信息"""
        for img_file in self.image_files:
            if not self.running:
                break

            image = cv2.imread(img_file)
            results = self.model(image)

            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                for j in range(len(detections)):
                    class_id = int(detections.cls[j])
                    class_name = self.model.names[class_id]
                    self.class_counts[class_name] += 1
                    self.total_detections += 1

    def determine_filtering(self):
        """根据数据集名称和检测统计确定是否应该过滤结果"""
        # 预定义数据集规则
        dataset_rules = {
            "coco": None,
            "voc": None,
            "helmet": ["helmet", "safety helmet", "hard hat"],
            "mask": ["mask", "face mask", "surgical mask"],
            "fire": ["fire", "smoke", "flame"],
            "person": ["person"],
            "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "animal": ["dog", "cat", "bird", "horse", "sheep", "cow"],
        }

        if not self.dataset_name:
            self.filter_classes = None
            return

        dataset_lower = self.dataset_name.lower()

        # 1. 精确匹配预设规则
        for key, value in dataset_rules.items():
            if key == dataset_lower:
                self.filter_classes = value
                print(f"精确匹配规则: {self.filter_classes}")
                return

        # 2. 部分匹配预设规则
        for key, value in dataset_rules.items():
            if key in dataset_lower:
                self.filter_classes = value
                print(f"部分匹配规则: {self.filter_classes}")
                return

        # 3. 智能匹配：基于检测统计和文件夹名称
        # 提取关键词（去除数字、特殊字符）
        keywords = [word for word in re.split(r'[\W\d]+', dataset_lower) if word]

        relevant_detections = 0
        related_classes = set()

        for class_name, count in self.class_counts.items():
            class_name_lower = class_name.lower()
            # 模糊匹配：类名包含任一关键词或关键词包含类名
            if any(keyword in class_name_lower or class_name_lower in keyword for keyword in keywords):
                relevant_detections += count
                related_classes.add(class_name)

        # 动态调整阈值：小数据集降低阈值
        dynamic_threshold = self.filter_threshold if self.total_detections > 50 else 0.3

        if self.total_detections > 0:
            relevance_ratio = relevant_detections / self.total_detections
        else:
            relevance_ratio = 0

        if relevance_ratio >= dynamic_threshold and related_classes:
            self.filter_classes = list(related_classes)
            print(f"智能过滤启用: {self.filter_classes} (相关度: {relevance_ratio:.2%})")
        else:
            self.filter_classes = None
            print(f"智能过滤未启用 (相关度: {relevance_ratio:.2%} < {dynamic_threshold})")

    def stop(self):
        self.running = False


class ImageInferenceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能图像分析工具")
        self.setGeometry(100, 100, 1400, 1000)

        # 美化页面
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        self.setPalette(palette)

        # 中心部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 主分割布局（垂直和水平结合）
        main_splitter = QSplitter(Qt.Vertical)
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setHandleWidth(5)
        top_splitter.setStyleSheet("QSplitter::handle { background: #ccc; }")
        main_splitter.addWidget(top_splitter)

        # 左侧边栏
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar.setMinimumWidth(220)
        sidebar.setStyleSheet("""
            background-color: #f0f0f0; 
            border: 1px solid #ccc; 
            border-radius: 10px; 
            padding: 20px;
        """)

        # 按钮样式模板
        button_style_template = """
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 18px 12px;
                min-width: 200px;
                min-height: 70px;
                font-size: 22px;
                font-weight: bold;
                border-radius: 10px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #a0a0a0;
            }}
        """

        # 定义不同按钮的颜色
        button_styles = [
            ("选择多张图片", self.load_multiple_images, "#2196F3", "#1976D2"),  # 蓝色
            ("上一张", self.show_previous_image, "#FF9800", "#F57C00"),  # 橙色
            ("下一张", self.show_next_image, "#4CAF50", "#45A049"),  # 绿色
            ("选择模型", self.select_model, "#2196F3", "#1976D2"),  # 蓝色
            ("运行推理", self.run_inference, "#F44336", "#D32F2F"),  # 红色
            ("保存结果", self.save_results, "#9C27B0", "#7B1FA2")  # 紫色
        ]

        # 创建按钮并设置样式
        for text, handler, color, hover_color in button_styles:
            btn = QPushButton(text)
            btn.setStyleSheet(button_style_template.format(color=color, hover_color=hover_color))
            btn.clicked.connect(handler)
            sidebar_layout.addWidget(btn)

        # 存储特定按钮以便后续引用
        self.model_select_btn = sidebar_layout.itemAt(3).widget()
        self.infer_btn = sidebar_layout.itemAt(4).widget()
        self.save_btn = sidebar_layout.itemAt(5).widget()

        sidebar_layout.addStretch()
        top_splitter.addWidget(sidebar)

        # 中心图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 2px solid #ccc;
            border-radius: 10px;
            background-color: white;
            padding: 15px;
        """)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        top_splitter.addWidget(scroll_area)

        # 右侧信息面板
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_panel.setMinimumWidth(350)  # 增加宽度
        info_panel.setStyleSheet("""
            background-color: #f0f0f0; 
            border: 1px solid #ccc; 
            border-radius: 10px;
            padding: 20px;
        """)

        # 数据集信息显示
        info_layout.addWidget(QLabel("数据集信息:", styleSheet="font-size: 22px; font-weight: bold; color: #333;"))
        self.dataset_info_label = QLabel("未检测到数据集")
        self.dataset_info_label.setWordWrap(True)
        self.dataset_info_label.setStyleSheet("""
            padding: 20px; 
            background-color: white; 
            border: 1px solid #ddd; 
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 20px;
            line-height: 1.6;
        """)
        info_layout.addWidget(self.dataset_info_label)

        # 模型信息显示
        info_layout.addWidget(QLabel("模型信息:", styleSheet="font-size: 22px; font-weight: bold; color: #333;"))
        self.model_info_label = QLabel("未加载模型")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("""
            padding: 20px; 
            background-color: white; 
            border: 1px solid #ddd; 
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 20px;
            line-height: 1.6;
        """)
        info_layout.addWidget(self.model_info_label)

        # 检测结果展示 - 显著增大字体
        info_layout.addWidget(QLabel("检测结果:", styleSheet="font-size: 22px; font-weight: bold; color: #333;"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                padding: 25px; 
                background-color: white; 
                border: 1px solid #ddd; 
                border-radius: 10px;
                font-size: 30px;  /* 增大字体 */
                line-height: 1.8;
                font-weight: bold; /* 加粗字体 */
            }
        """)
        self.result_text.setAlignment(Qt.AlignLeft)
        self.result_text.setText("等待推理...")
        info_layout.addWidget(self.result_text, 1)  # 给检测结果更多空间

        info_layout.addStretch()
        top_splitter.addWidget(info_panel)
        top_splitter.setStretchFactor(1, 1)  # 让中间区域可扩展
        main_splitter.setStretchFactor(0, 1)  # 主分割器让顶部区域占主要空间

        # 底部状态栏
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(5, 5, 5, 5)

        # 左下角预览区域
        preview_frame = QFrame()
        preview_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.addWidget(
            QLabel("下一张预览:", styleSheet="font-size: 20px; font-weight: bold;", alignment=Qt.AlignCenter))

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(250, 180)  # 增大预览尺寸
        self.preview_label.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px dashed #ccc;
            border-radius: 5px;
            font-size: 18px;
        """)
        self.preview_label.setText("无预览")
        preview_layout.addWidget(self.preview_label)

        bottom_layout.addWidget(preview_frame)

        # 进度条区域
        progress_frame = QFrame()
        progress_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 10px;
                text-align: center;
                background-color: #f0f0f0;
                height: 40px;
                font-size: 22px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 10px;
            }
        """)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("进度: %p%")
        progress_layout.addWidget(self.progress_bar)

        bottom_layout.addWidget(progress_frame, 1)  # 让进度条占据更多空间

        main_splitter.addWidget(bottom_widget)

        main_layout.addWidget(main_splitter)
        self.setCentralWidget(central_widget)

        self.image_files = []
        self.current_index = 0
        self.selected_model = None
        self.inference_results = {}
        self.inference_thread = None
        self.detection_details = {}
        self.dataset_name = ""  # 存储数据集名称
        self.dataset_path = ""  # 存储数据集路径

    def load_multiple_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择多个图片", "", "Images (*.jpg *.jpeg *.png)")
        if files:
            self.image_files = files
            self.current_index = 0
            self.display_image()
            self.progress_bar.setValue(0)
            self.result_text.setText("等待推理...")
            self.detection_details = {}

            # 提取数据集信息
            self.extract_dataset_info(files[0])

            self.update_preview()

    def extract_dataset_info(self, file_path):
        """从文件路径提取数据集信息"""
        # 获取文件所在目录
        dir_path = os.path.dirname(file_path)
        # 获取目录名称
        dir_name = os.path.basename(dir_path)

        # 设置数据集信息
        self.dataset_path = dir_path
        self.dataset_name = dir_name

        # 更新UI显示
        self.dataset_info_label.setText(f"数据集名称: {dir_name}\n数据集路径: {dir_path}")

    def display_image(self):
        if not self.image_files or not (0 <= self.current_index < len(self.image_files)):
            return

        file_path = self.image_files[self.current_index]

        # 如果该图像有推理结果，显示带标注的结果
        if file_path in self.inference_results:
            result_image = self.inference_results[file_path]
            height, width, channel = result_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
        # 否则显示原始图像
        else:
            pixmap = QPixmap(file_path)

        # 缩放并显示图像
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        # 更新检测结果文本
        if file_path in self.detection_details:
            self.result_text.setHtml(self.detection_details[file_path])
        else:
            self.result_text.setText("未推理")

        # 更新预览
        self.update_preview()

    def update_preview(self):
        """更新左下角的预览图片"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            next_file = self.image_files[self.current_index + 1]
            pixmap = QPixmap(next_file)

            if not pixmap.isNull():
                # 缩放为预览尺寸
                preview_pixmap = pixmap.scaled(
                    self.preview_label.width(), self.preview_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.preview_label.setPixmap(preview_pixmap)
                self.preview_label.setText("")
            else:
                self.preview_label.setText("预览加载失败")
        else:
            self.preview_label.setText("无下一张图片")
            self.preview_label.setPixmap(QPixmap())

    def show_previous_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.display_image()

    def show_next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "YOLO模型文件 (*.pt)")
        if model_path:
            try:
                # 加载模型
                self.selected_model = YOLO(model_path)

                # 更新模型信息显示
                model_name = os.path.basename(model_path)
                model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

                # 获取模型类别信息
                class_info = ""
                try:
                    if hasattr(self.selected_model.model, 'names') and self.selected_model.model.names:
                        num_classes = len(self.selected_model.model.names)
                        class_names = list(self.selected_model.model.names.values())

                        # 显示类别信息
                        class_info = f"<br>类别数量: {num_classes}<br>"
                        class_info += "支持类别: " + ", ".join(class_names[:8])
                        if num_classes > 8:
                            class_info += f" 等{num_classes}个类别"
                except Exception as e:
                    class_info = f"<br>类别信息获取失败: {str(e)}"

                model_info = f"模型名称: {model_name}<br>模型大小: {model_size:.2f} MB{class_info}"
                self.model_info_label.setText(model_info)

                QMessageBox.information(self, "成功", "模型加载成功!")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
                self.selected_model = None
                self.model_info_label.setText("模型加载失败")

    def run_inference(self):
        if not self.image_files:
            QMessageBox.warning(self, "警告", "请先选择图片!")
            return

        if not self.selected_model:
            QMessageBox.warning(self, "警告", "请先选择模型!")
            return

        # 如果已有推理线程在运行，先停止
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.stop()
            self.inference_thread.wait()

        # 清空之前的结果
        self.inference_results = {}
        self.progress_bar.setValue(0)
        self.result_text.setText("推理中...")
        self.detection_details = {}

        # 创建并启动推理线程
        self.inference_thread = InferenceThread(
            self.selected_model,
            self.image_files,
            self.dataset_name,
            self
        )
        self.inference_thread.progress_updated.connect(self.update_progress)
        self.inference_thread.result_updated.connect(self.update_result)
        self.inference_thread.inference_finished.connect(self.on_inference_finished)
        self.inference_thread.finished.connect(self.on_thread_finished)
        self.inference_thread.start()

        # 禁用推理按钮防止重复点击
        self.infer_btn.setEnabled(False)
        self.infer_btn.setText("推理中...")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_result(self, result_text):
        # 更新当前图像的检测结果
        current_file = self.image_files[self.current_index]
        self.detection_details[current_file] = result_text
        self.result_text.setHtml(result_text)

    def on_inference_finished(self):
        # 保存推理结果
        self.inference_results = self.inference_thread.inference_results
        self.detection_details = self.inference_thread.detection_details
        # 显示当前图像的推理结果
        self.display_image()

    def on_thread_finished(self):
        # 重新启用推理按钮
        self.infer_btn.setEnabled(True)
        self.infer_btn.setText("运行推理")

        # 显示完成消息
        QMessageBox.information(self, "完成", "所有图片推理完成!")

    def save_results(self):
        if not self.image_files:
            QMessageBox.warning(self, "警告", "没有图片可保存!")
            return

        if self.image_files[self.current_index] not in self.inference_results:
            QMessageBox.warning(self, "警告", "当前图片尚未推理!")
            return

        # 获取默认文件名（基于原始文件名）
        base_name = os.path.basename(self.image_files[self.current_index])
        name, ext = os.path.splitext(base_name)
        default_name = f"{name}_result{ext}"

        # 弹出保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存结果",
            os.path.join(os.path.dirname(self.image_files[self.current_index]), default_name),
            "Images (*.jpg *.png)"
        )

        if file_path:
            # 获取当前图像的推理结果
            result_image = self.inference_results[self.image_files[self.current_index]]
            # 转换为BGR格式并保存
            cv2.imwrite(file_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "成功", f"结果已保存到:\n{file_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageInferenceGUI()
    window.show()
    sys.exit(app.exec_())
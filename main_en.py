from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from core_en import YoloPredictor

from pathlib import Path
from utils.rtsp_win import Window
import traceback
import json
import sys
import cv2
import os
import numpy as np

class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # Main window sends a signal to YOLO instance to start execution

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Basic UI setup
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # Rounded transparent corners
        self.setWindowFlags(Qt.FramelessWindowHint)     # Set window flag: hide window border
        UIFuncitons.uiDefinitions(self)  # Custom UI definitions

        # Initial page
        self.task = ''
        self.PageIndex = 1
        self.content.setCurrentIndex(self.PageIndex)
        self.pushButton_detect.clicked.connect(self.button_detect)
        self.pushButton_pose.clicked.connect(self.button_pose)
        self.pushButton_classify.clicked.connect(self.button_classify)
        self.pushButton_segment.clicked.connect(self.button_segment)
        self.pushButton_track.clicked.connect(self.button_track)

        UIFuncitons.setup_button(self.pushButton_detect, ':/all/img/detect.png', ':/all/img/detect_hover.png')
        UIFuncitons.setup_button(self.pushButton_pose, ':/all/img/pose.png', ':/all/img/pose_hover.png')
        UIFuncitons.setup_button(self.pushButton_classify, ':/all/img/classify.png', ':/all/img/classify_hover.png')
        UIFuncitons.setup_button(self.pushButton_segment, ':/all/img/segment.png', ':/all/img/segment_hover.png')
        UIFuncitons.setup_button(self.pushButton_track, ':/all/img/track.png', ':/all/img/track_hover.png')

        self.src_home_button.setEnabled(False)
        self.src_file_button.setEnabled(False)
        self.src_img_button.setEnabled(False)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(False)
        self.settings_button.setEnabled(False)

        self.src_home_button.clicked.connect(self.return_home)

        #################################### image or video ####################################
        # Display shadow effects for modules
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # YOLO-v8 thread
        self.yolo_predict = YoloPredictor()                          # Create YOLO instance
        self.select_model = self.model_box.currentText()             # Default model

        self.yolo_thread = QThread()                                 # Create YOLO thread
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video, 'img'))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video, 'img'))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        self.Qtimer_ModelBox = QTimer(self)     # Timer: monitor model file changes every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))    # IOU textbox
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))      # IOU slider
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # Conf textbox
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))    # Conf slider
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))# Speed textbox
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # Speed slider

        # Initialize status displays
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        # Select input source
        self.src_file_button.clicked.connect(self.open_src_file)   # Select local file
        self.src_img_button.clicked.connect(self.open_src_img)     # Single image file
        self.run_button.clicked.connect(self.run_or_continue)      # Start/pause
        self.stop_button.clicked.connect(self.stop)                # Stop

        # Other feature buttons
        self.save_res_button.toggled.connect(self.is_save_res)     # Save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)     # Save label option
        #################################### image or video ####################################

        #################################### camera ####################################
        # Display shadow effects for camera modules
        UIFuncitons.shadow_style(self, self.Class_QF_cam, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF_cam, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF_cam, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF_cam, QColor(64, 186, 193))

        # YOLO-v8-camera thread
        self.yolo_predict_cam = YoloPredictor()                         # Create YOLO instance
        self.select_model_cam = self.model_box_cam.currentText()        # Default model

        self.yolo_thread_cam = QThread()                                # Create YOLO thread
        self.yolo_predict_cam.yolo2main_pre_img.connect(lambda c: self.cam_show_image(c, self.pre_cam))
        self.yolo_predict_cam.yolo2main_res_img.connect(lambda c: self.cam_show_image(c, self.res_cam))
        self.yolo_predict_cam.yolo2main_status_msg.connect(lambda c: self.show_status(c))
        self.yolo_predict_cam.yolo2main_fps.connect(lambda c: self.fps_label_cam.setText(c))
        self.yolo_predict_cam.yolo2main_class_num.connect(lambda c: self.Class_num_cam.setText(str(c)))
        self.yolo_predict_cam.yolo2main_target_num.connect(lambda c: self.Target_num_cam.setText(str(c)))
        self.yolo_predict_cam.yolo2main_progress.connect(self.progress_bar_cam.setValue(0))
        self.main2yolo_begin_sgl.connect(self.yolo_predict_cam.run)
        self.yolo_predict_cam.moveToThread(self.yolo_thread_cam)

        self.Qtimer_ModelBox_cam = QTimer(self)     # Timer: monitor model file changes every 2 seconds
        self.Qtimer_ModelBox_cam.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox_cam.start(2000)

        # Camera model parameters
        self.model_box_cam.currentTextChanged.connect(self.cam_change_model)
        self.iou_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'iou_spinbox_cam'))
        self.iou_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'iou_slider_cam'))
        self.conf_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'conf_spinbox_cam'))
        self.conf_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'conf_slider_cam'))
        self.speed_spinbox_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'speed_spinbox_cam'))
        self.speed_slider_cam.valueChanged.connect(lambda c: self.cam_change_val(c, 'speed_slider_cam'))

        # Initialize status displays
        self.Class_num_cam.setText('--')
        self.Target_num_cam.setText('--')
        self.fps_label_cam.setText('--')
        self.Model_name_cam.setText(self.select_model_cam)

        # Select detection source
        self.src_cam_button.clicked.connect(self.cam_button)  # Select camera

        # Start/pause buttons
        self.run_button_cam.clicked.connect(self.cam_run_or_continue)
        self.stop_button_cam.clicked.connect(self.cam_stop)

        # Other feature buttons
        self.save_res_button_cam.toggled.connect(self.cam_is_save_res)
        self.save_txt_button_cam.toggled.connect(self.cam_is_save_txt)
        #################################### camera ####################################

        #################################### RTSP ####################################
        self.src_rtsp_button.clicked.connect(self.rtsp_button)
        #################################### RTSP ####################################

        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))  # Left navigation toggle

        # Initialization
        self.load_config()
        self.show_status("Welcome to the YOLOv8 detection system. Please select a mode.")

    def switch_mode(self, task):
        self.task = task
        self.yolo_predict.task = task
        self.yolo_predict_cam.task = task

        # Update model folders
        self.update_model_lists()

        self.PageIndex = 0
        self.content.setCurrentIndex(0)
        self.src_home_button.setEnabled(True)
        self.src_file_button.setEnabled(True)
        self.src_img_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.setEnabled(True)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # Top-right settings button

        self.show_status(f"Current page: Image or Video detection, Mode: {task}")

    def update_model_lists(self):
        # Load model folder
        model_dir = f'./models/{self.task.lower()}/'
        
        # Get a list of model files with .pt, .onnx, or .engine extensions
        self.pt_list = [file for file in os.listdir(model_dir) if file.endswith(('.pt', 'onnx', 'engine'))]
        
        # Sort the model files by file size
        self.pt_list.sort(key=lambda x: os.path.getsize(os.path.join(model_dir, x)))
        
        # Update the model selection dropdown for image/video
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = os.path.join(model_dir, self.select_model)
        
        # Load model list for camera, using the same files
        self.pt_list_cam = self.pt_list.copy()
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.yolo_predict_cam.new_model_name = os.path.join(model_dir, self.select_model_cam)

    def button_classify(self):
        self.switch_mode('Classify')

    def button_detect(self):
        self.switch_mode('Detect')

    def button_pose(self):
        self.switch_mode('Pose')

    def button_segment(self):
        self.switch_mode('Segment')

    def button_track(self):
        self.switch_mode('Track')

    def reset_yolo_thread(self):
        self.yolo_thread.requestInterruption()     # Request the thread to stop
        self.yolo_thread.quit()                    # Quit the thread
        self.yolo_thread.wait()                    # Wait for the thread to finish

        self.yolo_predict.deleteLater()            # Delete the current YOLO object
        self.yolo_predict = YoloPredictor()        # Create a new YOLO instance

        self.yolo_thread = QThread()               # Create a new thread
        self.yolo_predict.yolo2main_pre_img.connect(lambda x: self.show_image(x, self.pre_video, 'img'))
        self.yolo_predict.yolo2main_res_img.connect(lambda x: self.show_image(x, self.res_video, 'img'))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

    def reset_yolo_thread_cam(self):
        self.yolo_thread_cam.requestInterruption()     # Request the thread to stop
        self.yolo_thread_cam.quit()                    # Quit the thread
        self.yolo_thread_cam.wait()                    # Wait for the thread to finish

        self.yolo_predict_cam.deleteLater()            # Delete the current YOLO cam object
        self.yolo_predict_cam = YoloPredictor()        # Create a new YOLO cam instance

        self.yolo_thread_cam = QThread()               # Create a new thread
        self.yolo_predict_cam.yolo2main_pre_img.connect(lambda c: self.cam_show_image(c, self.pre_cam))
        self.yolo_predict_cam.yolo2main_res_img.connect(lambda c: self.cam_show_image(c, self.res_cam))
        self.yolo_predict_cam.yolo2main_status_msg.connect(lambda c: self.show_status(c))
        self.yolo_predict_cam.yolo2main_fps.connect(lambda c: self.fps_label_cam.setText(c))
        self.yolo_predict_cam.yolo2main_class_num.connect(lambda c: self.Class_num_cam.setText(str(c)))
        self.yolo_predict_cam.yolo2main_target_num.connect(lambda c: self.Target_num_cam.setText(str(c)))
        self.yolo_predict_cam.yolo2main_progress.connect(self.progress_bar_cam.setValue(0))
        self.main2yolo_begin_sgl.connect(self.yolo_predict_cam.run)
        self.yolo_predict_cam.moveToThread(self.yolo_thread_cam)

    def reset(self):
        self.stop()
        self.reset_yolo_thread()
        self.cam_stop()
        self.reset_yolo_thread_cam()

    def return_home(self):
        # Disable buttons
        self.src_home_button.setEnabled(False)
        self.src_file_button.setEnabled(False)
        self.src_img_button.setEnabled(False)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(False)
        self.settings_button.setEnabled(False)

        # Return to the main page, reset state and buttons
        self.PageIndex = 1
        self.yolo_predict.source = ''
        self.yolo_predict_cam.source = ''
        self.content.setCurrentIndex(1)

        self.reset()
        self.show_status("Welcome to the YOLOv8 detection system. Please select a Mode.")
    ####################################image or video####################################
    # Select a local folder
    def open_src_file(self):
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)

        # Show status info based on task type
        mode_status = {
            'Classify': "Current page: image or video detection, Mode: Classify",
            'Detect': "Current page: image or video detection, Mode: Detect",
            'Pose': "Current page: image or video detection, Mode: Pose",
            'Segment': "Current page: image or video detection, Mode: Segment",
            'Track': "Current page: image or video detection, Mode: Track"
        }

        self.reset()
        self.switch_mode(self.task)

        # Show status info based on the selected task
        if self.task in mode_status:
            self.show_status(mode_status[self.task])

        # Set config file path
        config_file = 'config/fold.json'

        # Read config file to get the last opened folder path; use current directory if missing
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config.get('open_fold', os.getcwd())

        # Open file dialog for user to select folder
        FolderPath = QFileDialog.getExistingDirectory(self, 'Select Folder', open_fold)
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))  # Top-right settings button

        # If a folder was selected
        if FolderPath:
            FileFormat = [".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2", ".mp4", ".avi"]
            Foldername = [(FolderPath + "/" + filename) for filename in os.listdir(FolderPath)
                          for ext in FileFormat if ext in filename]
            if Foldername:
                self.yolo_predict.source = Foldername  # Set folder path as source
                self.show_status(f'Loaded folder: {os.path.basename(FolderPath)}')  # Show loading status
                config['open_fold'] = os.path.dirname(FolderPath)  # Update config

                # Save updated config
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)

                self.stop()  # Stop current detection
            else:
                self.show_status('No images found in the folder...')

    # Select local file
    def open_src_img(self):
        if self.PageIndex != 0:
            self.PageIndex = 0
        self.content.setCurrentIndex(0)

        # Display different status messages based on the task type
        mode_status = {
            'Classify': "Current page: image or video detection page, Mode: Classify",
            'Detect': "Current page: image or video detection page, Mode: Detect",
            'Pose': "Current page: image or video detection page, Mode: Pose",
            'Segment': "Current page: image or video detection page, Mode: Segment",
            'Track': "Current page: image or video detection page, Mode: Track"
        }
        if self.task in mode_status:
            self.show_status(mode_status[self.task])

        self.reset()
        self.switch_mode(self.task)

        # Set config file path
        config_file = 'config/fold.json'

        # Read contents of config file
        config = json.load(open(config_file, 'r', encoding='utf-8'))

        # Get the path of the last opened folder
        open_fold = config.get('open_fold', os.getcwd())

        # Open file dialog for user to select an image or video file
        if self.task == 'Track':
            title = 'Video'
            filters = "Pic File(*.mp4 *.mkv *.avi *.flv)"
        else:
            title = 'Video/Image'
            filters = "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)"

        name, _ = QFileDialog.getOpenFileName(self, title, open_fold, filters)

        # Top-right settings button
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))

        # If the user selected a file
        if name:
            # Set the selected file path as the source for yolo_predict
            self.yolo_predict.source = name

            # Show file loading status
            self.show_status('Loaded file: {}'.format(os.path.basename(name)))

            # Update the last opened folder path in the config
            config['open_fold'] = os.path.dirname(name)

            # Write the updated config back to the file
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

            # Stop detection
            self.stop()

    # Display the original image and detection results in the main window
    @staticmethod
    def show_image(img_src, label, flag):
        if flag == "path":
            img_src = cv2.imdecode(np.fromfile(img_src, dtype=np.uint8), -1)

        # Get original image height, width, and channels
        ih, iw, _ = img_src.shape

        # Get the label widget's width and height
        w, h = label.geometry().width(), label.geometry().height()

        # Maintain original aspect ratio, calculate scaled size
        if iw / w > ih / h:
            scal = w / iw
            nw, nh = w, int(scal * ih)
        else:
            scal = h / ih
            nw, nh = int(scal * iw), h

        # Resize the image and convert to RGB format
        frame = cv2.cvtColor(cv2.resize(img_src, (nw, nh)), cv2.COLOR_BGR2RGB)

        # Convert image data to a Qt image object and display it in the label
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    # Control start/pause of detection
    def run_or_continue(self):
        def handle_no_source():
            self.show_status('Please select an image or video source before starting detection...')
            self.run_button.setChecked(False)

        def start_detection():
            self.save_txt_button.setEnabled(False)  # Disable save options once detection starts
            self.save_res_button.setEnabled(False)
            self.show_status('Detecting...')
            self.yolo_predict.continue_dtc = True  # Set YOLO to continue detection

            if not self.yolo_thread.isRunning():
                self.yolo_thread.start()
                self.main2yolo_begin_sgl.emit()

        def pause_detection():
            self.yolo_predict.continue_dtc = False
            self.show_status("Detection paused...")
            self.run_button.setChecked(False)

        if not self.yolo_predict.source:
            handle_no_source()
        else:
            self.yolo_predict.stop_dtc = False

            if self.run_button.isChecked():  # If the start button is checked
                start_detection()
            else:  # If unchecked, pause the detection
                pause_detection()

    # Save test result button -- image/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            # Show message: image results will not be saved
            self.show_status('NOTE: Image results will NOT be saved')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            # Show message: image results will be saved
            self.show_status('NOTE: Image results WILL be saved')
            self.yolo_predict.save_res = True

    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            # Show message: label results will not be saved
            self.show_status('NOTE: Label results will NOT be saved')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            # Show message: label results will be saved
            self.show_status('NOTE: Label results WILL be saved')
            self.yolo_predict.save_txt = True

    # Stop button and related status handling
    def stop(self):
        def stop_yolo_thread():
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # End the thread
                # self.show_status('Detection stopped')
            self.yolo_predict.stop_dtc = True

        def reset_ui_elements():
            self.run_button.setChecked(False)
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.pre_video.clear()
            self.res_video.clear()
            self.progress_bar.setValue(0)
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

        stop_yolo_thread()
        reset_ui_elements()

    # Change detection parameters
    def change_val(self, x, flag):
        def update_iou():
            value = x / 100
            self.iou_spinbox.setValue(value)
            self.show_status(f'IOU Threshold: {value}')
            self.yolo_predict.iou_thres = value

        def update_conf():
            value = x / 100
            self.conf_spinbox.setValue(value)
            self.show_status(f'Conf Threshold: {value}')
            self.yolo_predict.conf_thres = value

        def update_speed():
            self.speed_spinbox.setValue(x)
            self.show_status(f'Delay: {x} ms')
            self.yolo_predict.speed_thres = x  # Milliseconds

        update_actions = {
            'iou_spinbox': lambda: self.iou_slider.setValue(int(x * 100)),
            'iou_slider': update_iou,
            'conf_spinbox': lambda: self.conf_slider.setValue(int(x * 100)),
            'conf_slider': update_conf,
            'speed_spinbox': lambda: self.speed_slider.setValue(x),
            'speed_slider': update_speed
        }

        if flag in update_actions:
            update_actions[flag]()

    # Change model
    def change_model(self, x):
        # Get the currently selected model name
        self.select_model = self.model_box.currentText()
        
        # Set model path prefix based on the task
        model_prefix = {
            'Classify': './models/classify/',
            'Detect': './models/detect/',
            'Pose': './models/pose/',
            'Segment': './models/segment/',
            'Track': './models/track/'
        }.get(self.task, './models/')

        # Set the new model name for the YOLO instance
        self.yolo_predict.new_model_name = f"{model_prefix}{self.task.lower()}/{self.select_model}"

        # Display message indicating model change
        self.show_status(f'Change Model: {self.select_model}')
        
        # Display the new model name on the interface
        self.Model_name.setText(self.select_model)
    ####################################image or video####################################

    ####################################camera####################################
    # Switch to Webcam detection page
    def cam_button(self):
        self.yolo_predict_cam.source = 0
        self.show_status('Current page: Webcam detection page')
        self.reset()
        self.switch_mode(self.task)

        if self.PageIndex != 2:
            self.PageIndex = 2
        self.content.setCurrentIndex(2)
        self.settings_button.clicked.connect(lambda: UIFuncitons.cam_settingBox(self, True))  # Top right settings button

    # Webcam control for start/pause detection
    def cam_run_or_continue(self):
        def handle_no_camera():
            self.show_status('No camera detected')
            self.run_button_cam.setChecked(False)

        def start_detection():
            self.run_button_cam.setChecked(True)  # Start button
            self.save_txt_button_cam.setEnabled(False)  # Disable saving after starting detection
            self.save_res_button_cam.setEnabled(False)
            self.show_status('Detection in progress...')
            self.yolo_predict_cam.continue_dtc = True

            if not self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.start()
                self.main2yolo_begin_sgl.emit()

        def pause_detection():
            self.yolo_predict_cam.continue_dtc = False
            self.show_status("Detection paused...")
            self.run_button_cam.setChecked(False)  # Stop button

        if self.yolo_predict_cam.source == '':
            handle_no_camera()
        else:
            self.yolo_predict_cam.stop_dtc = False

            if self.run_button_cam.isChecked():
                start_detection()
            else:
                pause_detection()

    # Display the original image and detection results on the webcam
    @staticmethod
    def cam_show_image(img_src, label):
        # Get the height, width, and channels of the original image
        ih, iw, _ = img_src.shape

        # Get the width and height of the label
        w, h = label.geometry().width(), label.geometry().height()

        # Maintain the original aspect ratio and calculate the resized dimensions
        if iw / w > ih / h:
            scal = w / iw
            nw, nh = w, int(scal * ih)
        else:
            scal = h / ih
            nw, nh = int(scal * iw), h

        # Resize the image and convert it to RGB format
        frame = cv2.cvtColor(cv2.resize(img_src, (nw, nh)), cv2.COLOR_BGR2RGB)

        # Convert the image data to a Qt image object and display it on the label
        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    # Change detection parameters for the webcam
    def cam_change_val(self, c, flag):
        def update_iou():
            value = c / 100
            self.iou_spinbox_cam.setValue(value)
            self.show_status(f'IOU Threshold: {value}')
            self.yolo_predict_cam.iou_thres = value

        def update_conf():
            value = c / 100
            self.conf_spinbox_cam.setValue(value)
            self.show_status(f'Conf Threshold: {value}')
            self.yolo_predict_cam.conf_thres = value

        def update_speed():
            self.speed_spinbox_cam.setValue(c)
            self.show_status(f'Delay: {c} ms')
            self.yolo_predict_cam.speed_thres = c  # milliseconds

        update_actions = {
            'iou_spinbox_cam': lambda: self.iou_slider_cam.setValue(int(c * 100)),
            'iou_slider_cam': update_iou,
            'conf_spinbox_cam': lambda: self.conf_slider_cam.setValue(int(c * 100)),
            'conf_slider_cam': update_conf,
            'speed_spinbox_cam': lambda: self.speed_spinbox_cam.setValue(c),
            'speed_slider_cam': update_speed,
        }

        if flag in update_actions:
            update_actions[flag]()

    # Change model for webcam detection
    def cam_change_model(self, c):
        # Get the currently selected model name
        self.select_model_cam = self.model_box_cam.currentText()
        
        # Set the model path prefix according to the task
        model_prefix = {
            'Classify': './models/classify/',
            'Detect': './models/detect/',
            'Pose': './models/pose/',
            'Segment': './models/segment/',
            'Track': './models/track/'
        }.get(self.task, './models/')

        # Set the new model name for the YOLO instance
        self.yolo_predict_cam.new_model_name = f"{model_prefix}{self.task.lower()}/{self.select_model_cam}"

        # Show a message indicating that the model has been changed
        self.show_status(f'Change Model: {self.select_model_cam}')
        
        # Display the new model name on the interface
        self.Model_name_cam.setText(self.select_model_cam)

    # Save detection results button -- Images/Videos
    def cam_is_save_res(self):
        if self.save_res_button_cam.checkState() == Qt.CheckState.Unchecked:
            # Show a message indicating the webcam results will not be saved
            self.show_status('NOTE: Webcam results will not be saved')
            
            # Set the save result flag in the YOLO instance to False
            self.yolo_thread_cam.save_res = False
        elif self.save_res_button_cam.checkState() == Qt.CheckState.Checked:
            # Show a message indicating the webcam results will be saved
            self.show_status('NOTE: Webcam results will be saved')
            
            # Set the save result flag in the YOLO instance to True
            self.yolo_thread_cam.save_res = True

    # Save detection results button -- Labels (txt)
    def cam_is_save_txt(self):
        if self.save_txt_button_cam.checkState() == Qt.CheckState.Unchecked:
            # Show a message indicating the label results will not be saved
            self.show_status('NOTE: Label results will not be saved')
            
            # Set the save label flag in the YOLO instance to False
            self.yolo_thread_cam.save_txt_cam = False
        elif self.save_txt_button_cam.checkState() == Qt.CheckState.Checked:
            # Show a message indicating the label results will be saved
            self.show_status('NOTE: Label results will be saved')
            
            # Set the save label flag in the YOLO instance to True
            self.yolo_thread_cam.save_txt_cam = True

    # Stop button and related state handling for webcam detection
    def cam_stop(self):
        def stop_yolo_thread():
            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()  # End the thread
                # self.show_status('Detection stopped')
            self.yolo_predict_cam.stop_dtc = True

        def reset_ui_elements():
            self.run_button_cam.setChecked(False)
            self.save_res_button_cam.setEnabled(True)
            self.save_txt_button_cam.setEnabled(True)
            self.pre_cam.clear()
            self.res_cam.clear()
            self.Class_num_cam.setText('--')
            self.Target_num_cam.setText('--')
            self.fps_label_cam.setText('--')

        stop_yolo_thread()
        reset_ui_elements()
    ####################################camera####################################
    ####################################rtsp####################################
    # RTSP input address
    def rtsp_button(self):
        self.reset()
        self.switch_mode(self.task)
        
        # Switch to RTSP detection page
        self.PageIndex = 2
        self.content.setCurrentIndex(2)
        self.show_status('Current page: RTSP detection page')
        
        def load_rtsp_window():
            self.rtsp_window = Window()
            config_file = 'config/ip.json'

            if not os.path.exists(config_file):
                ip = "rtsp://admin:admin888@192.168.1.2:555"
                new_config = {"ip": ip}
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(new_config, f, ensure_ascii=False, indent=2)
            else:
                config = json.load(open(config_file, 'r', encoding='utf-8'))
                ip = config['ip']

            self.rtsp_window.rtspEdit.setText(ip)
            self.rtsp_window.show()
            self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

        self.yolo_predict_cam.stream_buffer = True

        # Load the RTSP settings window
        load_rtsp_window()

        # Set the function for the settings button in the top right corner
        self.settings_button.clicked.connect(lambda: UIFuncitons.cam_settingBox(self, True))

    # Load network source
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title='Tip', text='Loading rtsp...', time=1000, auto=True).exec()
            self.yolo_predict_cam.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.show_status(f'Loading rtsp: {ip}')
            self.rtsp_window.close()
        except Exception as e:
            self.show_status(f'{e}')
    ####################################rtsp####################################
    ####################################common####################################
    # Display bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        
        def handle_page_0():
            if msg == 'Detection Complete' or msg == 'Detection Terminated':
                self.save_res_button.setEnabled(True)
                self.save_txt_button.setEnabled(True)
                self.run_button.setChecked(False)

                if self.yolo_thread.isRunning():
                    self.yolo_thread.quit()

                if msg == 'Detection Terminated':
                    self.progress_bar.setValue(0)
                    self.pre_video.clear()
                    self.res_video.clear()
                    self.Class_num.setText('--')
                    self.Target_num.setText('--')
                    self.fps_label.setText('--')

        def handle_page_1():
            if msg == 'Detection Terminated':
                if self.yolo_thread.isRunning():
                    self.yolo_thread.quit()
                    
                if self.yolo_thread_cam.isRunning():
                    self.yolo_thread_cam.quit()

                self.progress_bar.setValue(0)
                self.pre_video.clear()
                self.res_video.clear()
                self.Class_num.setText('--')
                self.Target_num.setText('--')
                self.fps_label.setText('--')

                self.progress_bar_cam.setValue(0)
                self.pre_cam.clear()
                self.res_cam.clear()
                self.Class_num_cam.setText('--')
                self.Target_num_cam.setText('--')
                self.fps_label_cam.setText('--')

        def handle_page_2():
            if msg == 'Detection Complete' or msg == 'Detection Terminated':
                self.save_res_button_cam.setEnabled(True)
                self.save_txt_button_cam.setEnabled(True)
                self.run_button_cam.setChecked(False)

                if self.yolo_thread_cam.isRunning():
                    self.yolo_thread_cam.quit()

                if msg == 'Detection Terminated':
                    self.progress_bar_cam.setValue(0)
                    self.pre_cam.clear()
                    self.res_cam.clear()
                    self.Class_num_cam.setText('--')
                    self.Target_num_cam.setText('--')
                    self.fps_label_cam.setText('--')

        # Handle different states based on the current page
        if self.PageIndex == 0:
            handle_page_0()
        elif self.PageIndex == 1:
            handle_page_1()
        elif self.PageIndex == 2:
            handle_page_2()

    # Continuously monitor model file changes
    def ModelBoxRefre(self):
        def update_model_box(folder):
            pt_list = os.listdir(folder)
            pt_list = [file for file in pt_list if file.endswith(('.pt', 'onnx', 'engine'))]
            pt_list.sort(key=lambda x: os.path.getsize(os.path.join(folder, x)))
            return pt_list

        folder_paths = {
            'Classify': './models/classify',
            'Detect': './models/detect',
            'Pose': './models/pose',
            'Segment': './models/segment',
            'Track': './models/track'
        }

        if self.task in folder_paths:
            pt_list = update_model_box(folder_paths[self.task])
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

    # Get mouse position (used for dragging the window by the title bar)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize adjustments when resizing the window (for dragging the bottom-right corner to resize the window)
    def resizeEvent(self, event):
        # Update resizing handle
        UIFuncitons.resize_grips(self)

    # Initialize configuration
    def load_config(self):
        config_file = 'config/setting.json'
        
        default_config = {
            "iou": 0.26,
            "conf": 0.33,
            "rate": 10,
            "save_res": 0,
            "save_txt": 0,
            "save_res_cam": 0,
            "save_txt_cam": 0
        }

        config = default_config.copy()
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
        else:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        
        def update_ui(config):
            ui_elements = {
                "save_res": (self.save_res_button, self.yolo_predict, "save_res"),
                "save_txt": (self.save_txt_button, self.yolo_predict, "save_txt"),
                "save_res_cam": (self.save_res_button_cam, self.yolo_predict_cam, "save_res_cam"),
                "save_txt_cam": (self.save_txt_button_cam, self.yolo_predict_cam, "save_txt_cam"),
            }

            for key, (button, instance, attr) in ui_elements.items():
                button.setCheckState(Qt.Checked if config[key] else Qt.Unchecked)
                setattr(instance, attr, config[key] != 0)

            self.run_button.setChecked(False)
            self.run_button_cam.setChecked(False)
        
        update_ui(config)

    # Close event, exit threads, and save settings
    def closeEvent(self, event):
        # Save configuration to settings file
        config_file = 'config/setting.json'
        config = {
            "iou": self.iou_spinbox.value(),
            "conf": self.conf_spinbox.value(),
            "rate": self.speed_spinbox.value(),
            "save_res": 0 if self.save_res_button.checkState() == Qt.Unchecked else 2,
            "save_txt": 0 if self.save_txt_button.checkState() == Qt.Unchecked else 2,
            "save_res_cam": 0 if self.save_res_button_cam.checkState() == Qt.Unchecked else 2,
            "save_txt_cam": 0 if self.save_txt_button_cam.checkState() == Qt.Unchecked else 2
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Exit threads and application
        def quit_threads():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()

            self.yolo_predict_cam.stop_dtc = True
            self.yolo_thread_cam.quit()
            
            # Display exit message and wait for 3 seconds
            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=3000, auto=True).exec()
            
            # Exit the application
            sys.exit(0)
        
        if self.yolo_thread.isRunning() or self.yolo_thread_cam.isRunning():
            quit_threads()
        else:
            sys.exit(0)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        
    def dropEvent(self, event):
        def handle_directory(directory):
            image_formats = {".jpg", ".png", ".jpeg", ".bmp", ".dib", ".jpe", ".jp2", ".mp4", ".avi"}
            image_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if os.path.splitext(filename)[1].lower() in image_formats]

            if image_files:
                self.yolo_predict.source = image_files
                self.show_status('Loading folder: {}'.format(os.path.basename(directory)))
                if ".avi" or ".mp4" in self.yolo_predict.source[0]:
                    self.cap = cv2.VideoCapture(self.yolo_predict.source[0])
                    ret, frame = self.cap.read()
                    if ret:
                        self.show_image(frame, self.pre_video, 'img')
                else:
                    self.show_image(self.yolo_predict.source[0], self.pre_video, 'path')
            else:
                self.show_status('No images in the folder...')

        def handle_file(file):
            self.yolo_predict.source = file
            file_ext = os.path.splitext(file)[1].lower()

            if file_ext in {".avi", ".mp4"}:
                self.cap = cv2.VideoCapture(self.yolo_predict.source)
                ret, frame = self.cap.read()
                if ret:
                    self.show_image(frame, self.pre_video, 'img')
            else:
                self.show_image(self.yolo_predict.source, self.pre_video, 'path')

            self.show_status('Loading file: {}'.format(os.path.basename(self.yolo_predict.source)))

        try:
            file = event.mimeData().urls()[0].toLocalFile()
            if file:
                if os.path.isdir(file):
                    handle_directory(file)
                else:
                    handle_file(file)
        except Exception as e:
            self.show_status('Error: {}'.format(e))
    ####################################common####################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())

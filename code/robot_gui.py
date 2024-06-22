import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
from pyniryo2 import NiryoRobot
import pyniryo
import os
import time
import threading
import importlib


class FaceRecognitionApp:
    def __init__(self, master):
        # Initialize the main application window
        self.master = master
        master.title("Face Recognition and Robot Control")

        # Default IP address for the robot
        self.robot_ip_address = tk.StringVar()
        self.robot_ip_address.set("127.0.0.1")

        self.robot_connected = False
        self.robot = None

        # Default database path
        self.db_path = tk.StringVar()
        self.db_path.set("/home/cps/catkin_ws_niryo_ned/code/db")

        # Create a frame for robot connection controls
        connect_robot_frame = tk.Frame(master)
        connect_robot_frame.pack()

        # Button to connect to the robot
        self.connect_button = tk.Button(connect_robot_frame, text="Connect to Robot", command=self.connect_to_robot)
        self.connect_button.pack(side=tk.LEFT)

        # Entry field for the robot IP address
        self.robot_ip_entry = tk.Entry(connect_robot_frame, textvariable=self.robot_ip_address)
        self.robot_ip_entry.pack(side=tk.LEFT)

        # Create a frame for database path selection
        db_path_frame = tk.Frame(master)
        db_path_frame.pack()

        # Button to select the database path
        self.select_db_button = tk.Button(db_path_frame, text="Select DB Path", command=self.select_db_path)
        self.select_db_button.pack(side=tk.LEFT)

        # Entry field to display the selected database path
        self.db_path_entry = tk.Entry(db_path_frame, textvariable=self.db_path, width=50)
        self.db_path_entry.pack(side=tk.LEFT)

        # Create a frame for model selection
        model_selection_frame = tk.Frame(master)
        model_selection_frame.pack()

        self.face_detection_backend = tk.StringVar()
        self.face_recognition_model = tk.StringVar()

        # Label and dropdown for face detection model selection
        tk.Label(model_selection_frame, text="Face Detection Model:").pack(side=tk.LEFT)
        self.face_detection_menu = tk.OptionMenu(model_selection_frame, self.face_detection_backend, *[
            'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn', 'retinaface',
            'mediapipe', 'yolov8', 'yunet', 'centerface'
        ])
        self.face_detection_menu.pack(side=tk.LEFT)

        # Label and dropdown for face recognition model selection
        tk.Label(model_selection_frame, text="Face Recognition Model:").pack(side=tk.LEFT)
        self.face_recognition_menu = tk.OptionMenu(model_selection_frame, self.face_recognition_model, *[
            "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
            "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"
        ])
        self.face_recognition_menu.pack(side=tk.LEFT)

        # Button to confirm model selection
        self.select_model_button = tk.Button(model_selection_frame, text="Select Model", command=self.select_model)
        self.select_model_button.pack(side=tk.LEFT)

        # Create a frame for adding/loading images
        add_load_image_frame = tk.Frame(master)
        add_load_image_frame.pack()

        # Button to add an image to the database
        self.add_image_button = tk.Button(add_load_image_frame, text="Add Image to DB", command=self.add_image)
        self.add_image_button.pack(side=tk.LEFT)

        self.loaded_db_images = None

        # Button to load database images
        self.load_images_button = tk.Button(add_load_image_frame, text="Load Database Images",
                                            command=self.load_database_images)
        self.load_images_button.pack(side=tk.RIGHT)

        # Label to display images
        self.image_label = tk.Label(master)
        self.image_label.pack()

        # Status labels for detection and verification
        self.status_detection_var = tk.StringVar()
        self.status_detection_var.set("Face detection Status: N/A")
        self.status_detection_label = tk.Label(master, textvariable=self.status_detection_var)
        self.status_detection_label.pack()

        self.status_verification_var = tk.StringVar()
        self.status_verification_var.set("Face Verification Status: N/A")
        self.status_verification_label = tk.Label(master, textvariable=self.status_verification_var)
        self.status_verification_label.pack()

        # New labels for time display
        self.time_detection_var = tk.StringVar()
        self.time_verification_var = tk.StringVar()

        self.time_detection_label = tk.Label(master, textvariable=self.time_detection_var)
        self.time_detection_label.pack()

        self.time_verification_label = tk.Label(master, textvariable=self.time_verification_var)
        self.time_verification_label.pack()

        # Create a frame for recognition start/stop buttons
        recogniton_start_end_btn_frame = tk.Frame(master)
        recogniton_start_end_btn_frame.pack()
        self.start_recognition_button = tk.Button(recogniton_start_end_btn_frame, text="Start Face Recognition",
                                                  command=self.start_recognition)
        self.start_recognition_button.pack(side=tk.LEFT)
        self.stop_recognition_button = tk.Button(recogniton_start_end_btn_frame, text="Stop Face Recognition",
                                                 command=self.stop_recognition)
        self.stop_recognition_button.pack(side=tk.RIGHT)

        # Create a frame for robot movement/end buttons
        move_end_btn_frame = tk.Frame(master)
        move_end_btn_frame.pack()
        self.move_robot_button = tk.Button(move_end_btn_frame, text="Move Robot", command=self.move_robot)
        self.move_robot_button.pack(side=tk.LEFT)
        self.end_button = tk.Button(move_end_btn_frame, text="End", command=self.end_program)
        self.end_button.pack(side=tk.RIGHT)

        # Set the protocol for the window close event
        self.master.protocol("WM_DELETE_WINDOW", self.end_program)

        # Initialize application state variables
        self.face_recogntion_running = False
        self.robot_is_moving = False
        self.recognition_thread = None

        # Display a blank image initially
        self.display_blank_image()

    ## Selects the face detection and recognition models
    def select_model(self):
        detection_backend = self.face_detection_backend.get()
        recognition_model = self.face_recognition_model.get()
        if detection_backend and recognition_model:
            self.load_required_libraries(detection_backend, recognition_model)
            messagebox.showinfo("Selected Models",
                                f"Face Detection: {detection_backend}\nFace Recognition: {recognition_model}")
        else:
            messagebox.showwarning("Selection Error", "Please select both face detection and recognition models.")

    ## Loads the required libraries for the selected face detection and recognition models
    def load_required_libraries(self, detection_backend, recognition_model):
        backend_library_mapping = {
            'mtcnn': 'mtcnn',
            'retinaface': 'retinaface',
            'yolov8': 'ultralytics',
            'yunet': 'opencv',
            'centerface': 'centerface'
        }

        recognition_library_mapping = {
            'Dlib': 'dlib'
        }

        if detection_backend in backend_library_mapping:
            backend_library = backend_library_mapping[detection_backend]
            try:
                importlib.import_module(backend_library)
            except ImportError:
                messagebox.showerror("Import Error", f"Required library for {detection_backend} could not be imported.")

        if recognition_model in recognition_library_mapping:
            recognition_library = recognition_library_mapping[recognition_model]
            try:
                importlib.import_module(recognition_library)
            except ImportError:
                messagebox.showerror("Import Error", f"Required library for {recognition_model} could not be imported.")

    ## Opens a dialog to select the database path
    def select_db_path(self):
        selected_path = filedialog.askdirectory()
        if selected_path:
            self.db_path.set(selected_path)

    ## Connects to the robot using the provided IP address
    def connect_to_robot(self):
        try:
            if not self.robot_connected:
                ip_address = self.robot_ip_address.get()
                self.robot = NiryoRobot(ip_address)
                self.robot_connected = True
                messagebox.showinfo("Success", f"Connected to robot at {ip_address}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to connect to robot: {e}")

    ## Loads images from the database directory
    def load_database_images(self):
        try:
            db_images = {}
            for root, dirs, files in os.walk(self.db_path.get()):
                for file in files:
                    if file.endswith((".jpg", ".jpeg", ".png")):
                        person_name = os.path.basename(root)
                        if person_name not in db_images:
                            db_images[person_name] = []
                        db_images[person_name].append(os.path.join(root, file))
            self.loaded_db_images = db_images
            messagebox.showinfo("Success", "Database images loaded successfully.")
        except Exception as e:
            messagebox.showerror(f"Error loading database images: {e}")
            return {}

    """
       Verifies detected faces against the database images.

       Args:
           frame (numpy.ndarray): The current frame/image from the video stream.
           faces (list): List of detected faces, each containing facial area coordinates.
           db_images (dict): Dictionary mapping person names to lists of image paths in the database.

       This method iterates over detected faces, compares them against stored database images using DeepFace's
       verification method, and updates the GUI with verification results.

       """

    def verify_faces(self, frame, faces, db_images):
        try:
            for face in faces:
                x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], \
                             face['facial_area']['h']
                face_img = frame[y:y + h, x:x + w]
                is_verified = False
                for person_name, images in db_images.items():
                    for image_path in images:
                        start_time = time.time()  # Start time for face verification
                        result = DeepFace.verify(face_img, image_path, model_name="VGG-Face", enforce_detection=False)
                        if result["verified"]:
                            end_time = time.time()  # End time for face verification
                            self.status_verification_var.set(f"Face verified as {person_name}")
                            duration = end_time - start_time
                            self.time_verification_var.set(f"Verification time: {duration:.2f} seconds")
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f"Hello {person_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 255, 0), 2)
                            is_verified = True
                            break
                    if is_verified:
                        break
                if not is_verified:
                    self.status_verification_var.set("Face not verified")
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error verifying faces: {e}")

    # Add an image to the database
    def add_image(self):

        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            person_name = simpledialog.askstring("Person Name", "Enter the name of the person:")
            if person_name:
                person_folder = os.path.join(self.db_path.get(), person_name)
                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)
                new_image_path = os.path.join(person_folder, os.path.basename(image_path))
                os.rename(image_path, new_image_path)
                messagebox.showinfo("Success", "Image added to database successfully.")

    """
       Starts the face recognition process if database images are loaded.

       This method sets up the necessary flags and threads to begin the face recognition loop,
       updating the GUI status accordingly. If no database images are loaded, an error message
       is displayed.

       """

    def start_recognition(self):

        try:
            if self.loaded_db_images:
                self.face_recogntion_running = True
                self.status_verification_var.set("Face Recognition Status: Running")
                self.status_detection_var.set("Face Detection Status: Running")
                self.recognition_thread = threading.Thread(target=self.start_recognition_loop)
                self.recognition_thread.start()
            else:
                messagebox.showerror("Error", "No database images loaded. Load database images first.")
        except Exception as e:
            print(f"Error starting recognition: {str(e)}")

    """
            Continuous loop for face recognition.

            This method continuously updates the video feed and performs face detection and recognition
            as long as the `face_recogntion_running` flag is True. It updates the GUI with detection
            status and verifies faces against loaded database images.

            """

    def start_recognition_loop(self):
        while self.face_recogntion_running:
            self.update_video()
            # time.sleep(0.1)

    """
            Update the video stream and perform face detection.

            This method retrieves a compressed image from the robot's vision system, decompresses it,
            corrects for camera distortions, and detects faces using the selected face detection backend.
            If faces are detected with sufficient confidence, it updates the GUI with detection status
            and verifies the detected faces against loaded database images if face recognition is running.

            """

    def update_video(self):
        try:
            img_compressed = self.robot.vision.get_img_compressed()
            img_raw = pyniryo.uncompress_image(img_compressed)
            mtx, dist = self.robot.vision.get_camera_intrinsics()
            img_undistort = pyniryo.undistort_image(img_raw, mtx, dist)
            start_time = time.time()  # Start time for face detection
            faces = DeepFace.extract_faces(img_undistort, detector_backend=self.face_detection_backend.get(),
                                           enforce_detection=False)

            if faces and faces[0]['confidence'] > 0:
                end_time = time.time()  # End time for face detection
                duration = end_time - start_time
                self.time_detection_var.set(f"Detection time: {duration:.2f} seconds")
                self.status_detection_var.set("Face detected")
                if self.face_recogntion_running:
                    self.verify_faces(img_undistort, faces, self.loaded_db_images)
            else:
                self.status_detection_var.set("No face detected")
            self.display_image(img_undistort)

        except Exception as e:
            print(f"Error updating video stream: {e}")

    """
           Stop the face recognition process.

           If face recognition is currently running, this method sets the status variables to indicate
           that recognition has stopped. It stops the recognition loop thread with a timeout period
           and clears the detection and verification times from the GUI.
           """

    def stop_recognition(self):
        # Stop the face recognition process
        if self.face_recogntion_running:
            self.status_verification_var.set("Face Recognition Status: Stopped")
            self.status_detection_var.set("Face Detection Status: Stopped")
            self.face_recogntion_running = False
            timeout_seconds = 5  # Set a timeout period in seconds
            self.recognition_thread.join(timeout=timeout_seconds)
            self.time_detection_var.set("")
            self.time_verification_var.set("")
            self.status_verification_var.set("Face Recognition Status: Stopped")
            self.status_detection_var.set("Face Detection Status: Stopped")

    """
           Display a blank image on the GUI.

           Creates a blank RGB image with dimensions 640x480 pixels and sets it as the image displayed
           on the self.image_label widget in the graphical user interface.
           """
    def display_blank_image(self):
        blank_image = Image.new('RGB', (640, 480), (0, 0, 0))
        img_tk = ImageTk.PhotoImage(blank_image)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    """
            Display an image on the GUI.

            Converts the provided OpenCV BGR-format image `img` to RGB format using cv2.cvtColor,
            then converts it to a PIL Image object using Image.fromarray. Finally, converts the PIL Image
            to a Tkinter-compatible PhotoImage `img_tk` and updates the self.image_label widget in the GUI.

            Args:
                img: An OpenCV image (BGR format) to be displayed.

            Raises:
                Exception: If there is an error during the image conversion or displaying process.
            """
    def display_image(self, img):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
        except Exception as e:
            print(f"Error displaying image: {str(e)}")

    """
            Move the robot to specified joint coordinates.

            If the robot is connected (`self.robot_connected` is True), prompts the user to enter
            joint coordinates through a dialog box. If valid coordinates are provided, starts a new
            thread to execute the `perform_move` method with the specified joint coordinates.

            Raises:
                Exception: If there is an error during the thread creation or movement execution process.
            """
    def move_robot(self):
        if self.robot_connected:
            try:
                joint_coords = simpledialog.askstring("Joint Coordinates", "Enter joint coordinates (comma-separated):")
                if joint_coords:
                    joint_coords = tuple(map(float, joint_coords.split(',')))
                    threading.Thread(target=self.perform_move, args=(joint_coords,)).start()
            except Exception as e:
                print(f"Error moving robot: {str(e)}")

    """
            Perform the robot movement to reach the target joint coordinates.

            Args:
                new_joint (tuple of float): Target joint coordinates to move the robot arm to.

            If the robot is connected (`self.robot_connected` is True) and not already moving (`self.robot_is_moving`
            is False), this method calculates an interpolated path to smoothly move from the current joint position
            to the target `new_joint`. It uses a step size of 0.1 radians (`step = 0.1`) for each joint movement.

            During the movement:
            - Interpolates each joint movement using a helper function `interpolate`.
            - Checks if the face recognition process (`self.face_recogntion_running`) is active; if so, updates the video
              feed during each movement step.
            - If face recognition is not running, captures and displays a live video stream from the robot's camera.

            Raises:
                Exception: If there is an error during the movement execution process.
            """
    def perform_move(self, new_joint):
        if self.robot_connected and not self.robot_is_moving:
            self.robot_is_moving = True
            step = 0.1
            current_joint = self.robot.arm.get_joints()

            def interpolate(start, end, step):
                return start + step if start < end else start - step

            joints = list(current_joint)
            while any(abs(curr - target) > step for curr, target in zip(joints, new_joint)):
                try:
                    for i in range(len(joints)):
                        if abs(joints[i] - new_joint[i]) > step:
                            joints[i] = interpolate(joints[i], new_joint[i], step)

                    self.robot.arm.move_joints(joints)
                    time.sleep(0.1)
                    if self.face_recogntion_running:
                        self.update_video()
                    else:
                        img_compressed = self.robot.vision.get_img_compressed()
                        img_raw = pyniryo.uncompress_image(img_compressed)
                        mtx, dist = self.robot.vision.get_camera_intrinsics()
                        img_undistort = pyniryo.undistort_image(img_raw, mtx, dist)
                        self.display_image(img_undistort)

                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    for i in range(len(joints)):
                        if abs(joints[i] - new_joint[i]) <= step:
                            joints[i] = new_joint[i]
            self.robot_is_moving = False

    """
           End the program by terminating the robot connection (if connected) and closing the GUI window.

           """
    def end_program(self):
        if self.robot_connected:
            self.robot.end()
        self.master.destroy()
        self.master.quit()


import contextlib

if __name__ == "__main__":
    # Redirect stdout and stderr to devnull to suppress terminal output
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            root = tk.Tk()
            app = FaceRecognitionApp(root)
            root.mainloop()

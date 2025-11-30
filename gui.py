"""CustomTkinter GUI for the Smart Waste Sorting App (real-time + image)"""
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import cv2
import time
from src import inference
from src.real_time import CameraWorker

ctk.set_appearance_mode('system')
ctk.set_default_color_theme('green')

class App(ctk.CTk):
    def __init__(self, model_path=None, class_indices=None):
        super().__init__()
        self.title('Smart Waste Sorting — Advanced')
        self.geometry('900x600')
        self.model_path = model_path
        self.class_indices = class_indices
        self.camera_worker = None
        self.create_widgets()
        if model_path:
            try:
                inference.load(model_path, class_indices)
            except Exception as e:
                print('Failed load model:', e)

    def create_widgets(self):
        self.sidebar = ctk.CTkFrame(self, width=180)
        self.sidebar.pack(side='left', fill='y', padx=10, pady=10)
        ctk.CTkLabel(self.sidebar, text='Menu', font=('Helvetica',16,'bold')).pack(pady=8)
        ctk.CTkButton(self.sidebar, text='Home', command=self.show_home).pack(fill='x', pady=6)
        ctk.CTkButton(self.sidebar, text='Image Classify', command=self.show_image_tab).pack(fill='x', pady=6)
        ctk.CTkButton(self.sidebar, text='Real-Time', command=self.show_realtime_tab).pack(fill='x', pady=6)
        ctk.CTkButton(self.sidebar, text='Model Insights', command=self.show_model_tab).pack(fill='x', pady=6)
        ctk.CTkButton(self.sidebar, text='Exit', command=self.on_close).pack(side='bottom', pady=12)

        self.container = ctk.CTkFrame(self)
        self.container.pack(side='right', expand=True, fill='both', padx=10, pady=10)
        self.current_frame = None
        self.show_home()

    def clear_container(self):
        for w in self.container.winfo_children():
            w.destroy()

    def show_home(self):
        self.clear_container()
        frame = ctk.CTkFrame(self.container)
        ctk.CTkLabel(frame, text='Smart Waste Sorting System', font=('Helvetica', 20, 'bold')).pack(pady=10)
        ctk.CTkLabel(frame, text='Advanced Python-only project with real-time camera classification.').pack(pady=5)
        ctk.CTkLabel(frame, text='Model path: ' + (self.model_path or 'Not loaded')).pack(pady=5)
        frame.pack(expand=True, fill='both')
        self.current_frame = frame

    def show_image_tab(self):
        self.clear_container()
        frame = ctk.CTkFrame(self.container)
        def open_image():
            path = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg;*.png;*.jpeg')])
            if not path:
                return
            img = Image.open(path).resize((400,300))
            img_tk = ImageTk.PhotoImage(img)
            img_label.configure(image=img_tk)
            img_label.image = img_tk
            # run prediction
            frame_bgr = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            try:
                label, prob = inference.predict(frame_bgr)
            except Exception:
                label, prob = ('Model not loaded', 0.0)
            result_label.configure(text=f'{label} ({prob:.2f})')
        img_label = ctk.CTkLabel(frame, text='No image')
        img_label.pack(pady=10)
        ctk.CTkButton(frame, text='Open Image', command=open_image).pack(pady=6)
        result_label = ctk.CTkLabel(frame, text='Result will appear here')
        result_label.pack(pady=8)
        frame.pack(expand=True, fill='both')
        self.current_frame = frame

    def show_realtime_tab(self):
        self.clear_container()
        frame = ctk.CTkFrame(self.container)
        self.video_label = ctk.CTkLabel(frame, text='Camera feed')
        self.video_label.pack(pady=8)
        btn_frame = ctk.CTkFrame(frame)
        btn_frame.pack(pady=6)
        start_btn = ctk.CTkButton(btn_frame, text='Start Camera', command=self.start_camera)
        start_btn.grid(row=0, column=0, padx=6)
        stop_btn = ctk.CTkButton(btn_frame, text='Stop Camera', command=self.stop_camera)
        stop_btn.grid(row=0, column=1, padx=6)
        self.pred_label = ctk.CTkLabel(frame, text='Prediction: -')
        self.pred_label.pack(pady=8)
        frame.pack(expand=True, fill='both')
        self.current_frame = frame

    def update_video(self):
        if self.camera_worker and self.camera_worker.frame is not None:
            frame = self.camera_worker.frame
            # convert to RGB then to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((640,360))
            img_tk = ImageTk.PhotoImage(img)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk
            label, prob = self.camera_worker.pred
            self.pred_label.configure(text=f'Prediction: {label} ({prob:.2f})')
        if self.camera_worker and self.camera_worker.running:
            self.after(30, self.update_video)

    def start_camera(self):
        if self.camera_worker and self.camera_worker.running:
            return
        self.camera_worker = CameraWorker(src=0)
        self.camera_worker.start()
        # give camera a moment
        self.after(100, self.update_video)

    def stop_camera(self):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
            self.video_label.configure(text='Camera stopped', image='')
            self.pred_label.configure(text='Prediction: -')

    def show_model_tab(self):
        self.clear_container()
        frame = ctk.CTkFrame(self.container)
        ctk.CTkLabel(frame, text='Model Insights (Training plots)').pack(pady=8)
        # show placeholder if exists
        try:
            from PIL import Image as PImage, ImageTk as PImageTk
            p = PImage.open('training_plots.png').resize((700,400))
            ptk = PImageTk.PhotoImage(p)
            lbl = ctk.CTkLabel(frame, image=ptk, text='')
            lbl.image = ptk
            lbl.pack()
        except Exception:
            ctk.CTkLabel(frame, text='No training plots found.').pack()
        frame.pack(expand=True, fill='both')
        self.current_frame = frame

    def on_close(self):
        self.stop_camera()
        self.destroy()


from src.gui import App

MODEL_PATH = 'model/waste_classifier.h5'  
CLASS_INDICES = None  
app = App(model_path=MODEL_PATH, class_indices=CLASS_INDICES)
app.mainloop()

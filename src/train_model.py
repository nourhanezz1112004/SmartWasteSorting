"""Train a transfer-learning classifier for waste images.

Directory structure expected:
dataset/
  train/
    plastic/
    metal/
    organic/
    paper/
    glass/
  val/
    plastic/
    ...
    
"""
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

def build_model(num_classes, input_shape=(224,224,3), lr=1e-4):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    # freeze base
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, out_dir):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(out_dir / 'training_plots.png')
    plt.close()

def main(args):
    dataset_dir = Path(args.dataset)
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'
    out_dir = Path('model')
    out_dir.mkdir(exist_ok=True)
    # data generators
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255)
    target_size = (224,224)
    batch_size = 16

    train_gen = train_datagen.flow_from_directory(str(train_dir), target_size=target_size, batch_size=batch_size, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(str(val_dir), target_size=target_size, batch_size=batch_size, class_mode='categorical')

    num_classes = len(train_gen.class_indices)
    model = build_model(num_classes, input_shape=(224,224,3), lr=args.lr)
    print(model.summary())

    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)
    model.save(args.output)
    print('Model saved to', args.output)
    plot_history(history, Path('.'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output', default='model/waste_classifier.h5')
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)

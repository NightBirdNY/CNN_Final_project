import kagglehub
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

# 1. KLASÖR AYARLARI
output_folder = 'train_v3'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Veri yolu kontrol ediliyor...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
base_dir = os.path.join(path, 'chest_xray')
if not os.path.exists(base_dir):
    base_dir = path

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
# val_dir ARTIK KULLANILMAYACAK (Çünkü içi boş sayılır)

# 2. HİPERPARAMETRELER
IMG_WIDTH, IMG_HEIGHT = 150, 150  # 150 gayet yeterli, sorun boyutta değil.
BATCH_SIZE = 32
EPOCHS = 15

# 3. VERİ ÖN İŞLEME (KRİTİK DÜZELTME BURADA)
# validation_split=0.2 diyerek Train klasörünü ikiye bölüyoruz.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # %20'sini ayır!
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Eğitim Verisi (Train klasörünün %80'i)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'  # Burası önemli
)

# Doğrulama Verisi (Train klasörünün %20'si - Artık 1000+ resim var!)
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'  # Burası önemli
)

# Test verisi (Buna dokunmuyoruz, final testi için)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# 4. CUSTOM CNN MODEL (Aynı Kalıyor)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])

# 5. EĞİTİM
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

print("Eğitim Başlıyor (Düzeltilmiş Validation Seti İle)...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr]
)

# 6. GRAFİKLERİ KAYDET
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
plt.title('Düzeltilmiş Model Doğruluğu')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Düzeltilmiş Model Kaybı')
plt.legend()

plt.savefig(os.path.join(output_folder, 'final_basari_grafigi.png'), dpi=300)
plt.show()

# Test ve Rapor
predictions = model.predict(test_generator)
y_pred = np.where(predictions > 0.5, 1, 0)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.title('Final Confusion Matrix')
plt.savefig(os.path.join(output_folder, 'final_confusion_matrix.png'), dpi=300)
plt.show()

report = classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia'])
print(report)

with open(os.path.join(output_folder, 'final_sonuc_raporu.txt'), 'w', encoding='utf-8') as f:
    f.write("--- FINAL RAPOR (VALIDATION SPLIT FIXED) ---\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
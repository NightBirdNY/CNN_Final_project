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

# ---------------------------------------------------------
# 1. KLASÖR VE VERİ AYARLARI
# ---------------------------------------------------------
# Sonuçları kaydedeceğimiz özel klasörü oluşturalım
output_folder = 'sonuclar_custom_v2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Klasör oluşturuldu: {output_folder}")

print("Veri yolu kontrol ediliyor...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
base_dir = os.path.join(path, 'chest_xray')
if not os.path.exists(base_dir):
    base_dir = path

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')

# ---------------------------------------------------------
# 2. HİPERPARAMETRELER
# ---------------------------------------------------------
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 20  # Biraz daha uzun tutalım, erken durdurma zaten var.

# ---------------------------------------------------------
# 3. VERİ ÖN İŞLEME (Data Augmentation)
# ---------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,  # Açıyı artırdık (Daha zor görev)
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ---------------------------------------------------------
# 4. CUSTOM CNN V2 (Optimize Edilmiş Mimari)
# ---------------------------------------------------------
model = models.Sequential([
    # 1. Blok
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # 2. Blok
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),  # Konvolüsyon arasına ufak dropout ekledik

    # 3. Blok
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # 4. Blok (Opsiyonel: Derinliği artırıp filtreyi sabit tuttuk)
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # Sınıflandırma Bloğu (Classifier)
    layers.Flatten(),
    # Dense katmanını 512'den 128'e düşürdük ve L2 Regularization ekledik
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),  # Ana dropout
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-4),  # Yavaş ve emin adımlarla
              metrics=['accuracy'])

model.summary()

# ---------------------------------------------------------
# 5. EĞİTİM (CALLBACKS İLE)
# ---------------------------------------------------------
# Patience'ı artırdık, hemen pes etmesin.
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

print("Custom CNN V2 Eğitimi Başlıyor...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr]
)

# ---------------------------------------------------------
# 6. SONUÇLARI KAYDETME (V2 Klasörüne)
# ---------------------------------------------------------
# Accuracy Grafiği
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
plt.title('V2 Model Doğruluğu')
plt.legend()

# Loss Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('V2 Model Kaybı')
plt.legend()

save_path_graph = os.path.join(output_folder, 'v2_basari_grafigi.png')
plt.savefig(save_path_graph, dpi=300)
plt.show()

# Test Raporu
print("Test ediliyor...")
predictions = model.predict(test_generator)
y_pred = np.where(predictions > 0.5, 1, 0)
y_true = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.title('V2 Confusion Matrix')
save_path_cm = os.path.join(output_folder, 'v2_confusion_matrix.png')
plt.savefig(save_path_cm, dpi=300)
plt.show()

# Metin Raporu
report = classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia'])
print(report)

save_path_txt = os.path.join(output_folder, 'v2_sonuc_raporu.txt')
with open(save_path_txt, 'w', encoding='utf-8') as f:
    f.write("--- CUSTOM CNN V2 (Regularized) RAPORU ---\n")
    f.write("Yapılan Değişiklikler: Dense 128'e düşürüldü, L2 Regularization eklendi, Dropout artırıldı.\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

print(f"Tüm sonuçlar '{output_folder}' klasörüne kaydedildi.")
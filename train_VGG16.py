import kagglehub
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

# ---------------------------------------------------------
# 1. KLASÖR AYARLARI (YENİ KLASÖR)
# ---------------------------------------------------------
output_folder = 'sonuclar_transfer_vgg16'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Veri yolu kontrol ediliyor...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
base_dir = os.path.join(path, 'chest_xray')
if not os.path.exists(base_dir):
    base_dir = path

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# ---------------------------------------------------------
# 2. HİPERPARAMETRELER
# ---------------------------------------------------------
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 15

# ---------------------------------------------------------
# 3. VERİ HAZIRLAMA (Split Taktigini Aynen Uyguluyoruz)
# ---------------------------------------------------------
# VGG16 için de rescale kullanabiliriz, basit tutalım.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,      # VGG16 zaten yetenekli, veriyi çok bozmayalım
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2    # %20 Doğrulama için ayır
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim Seti
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# Doğrulama Seti
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Test Seti
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ---------------------------------------------------------
# 4. TRANSFER LEARNING MODELİ (VGG16)
# ---------------------------------------------------------
print("VGG16 Ağırlıkları Yükleniyor...")
# include_top=False -> Sadece özellik çıkarıcı (gövde) kısmını al, sınıflandırma kısmını at.
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# DONDURMA (Freezing): Hazır modelin ağırlıklarını eğitmeyeceğiz, onlar zaten mükemmel.
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Yine dropout ekliyoruz
    layers.Dense(1, activation='sigmoid')
])

# Transfer learning'de learning rate genelde daha düşük tutulur.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=2e-5),
              metrics=['accuracy'])

model.summary()

# ---------------------------------------------------------
# 5. EĞİTİM
# ---------------------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

print("VGG16 Eğitimi Başlıyor...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr]
)

# ---------------------------------------------------------
# 6. SONUÇLARI KAYDETME
# ---------------------------------------------------------
# Grafikler
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
plt.title('VGG16 Model Doğruluğu')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('VGG16 Model Kaybı')
plt.legend()

plt.savefig(os.path.join(output_folder, 'vgg16_basari_grafigi.png'), dpi=300)
plt.show()

# Test ve Rapor
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
plt.title('VGG16 Confusion Matrix')
plt.savefig(os.path.join(output_folder, 'vgg16_confusion_matrix.png'), dpi=300)
plt.show()

# Metin Raporu
report = classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia'])
print(report)

with open(os.path.join(output_folder, 'vgg16_sonuc_raporu.txt'), 'w', encoding='utf-8') as f:
    f.write("--- TRANSFER LEARNING (VGG16) RAPORU ---\n")
    f.write("Kullanılan Model: VGG16 (ImageNet Weights)\n")
    f.write("Base Model Trainable: False (Frozen)\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

print(f"Tüm sonuçlar '{output_folder}' klasörüne kaydedildi.")
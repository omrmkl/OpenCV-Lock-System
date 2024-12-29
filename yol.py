import os

model_path = 'C:/Users/omere/PycharmProjects/Görüntüişlemeprojesi/dosyalar/hand_landmarker.task'

# Dosyanın var olup olmadığını kontrol etme
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
else:
    print(f"Model dosyası bulundu: {model_path}")

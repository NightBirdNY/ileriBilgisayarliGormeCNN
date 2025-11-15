import kagglehub

try:
    path = kagglehub.dataset_download("msambare/fer2013")
    print(f"Başarılı! İndirilen yol: {path}")
except Exception as e:
    print("--- HATA ALINDI ---")
    print(e)
    print("\nBu hata, API anahtarının (kaggle.json) eksik olduğunu gösteriyor.")
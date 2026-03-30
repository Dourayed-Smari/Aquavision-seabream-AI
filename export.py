# Créez un fichier nommé expert_pro.py et lancez-le
from ultralytics import YOLO
# Chargez votre nouveau modèle Elite
# Assurez-vous que le fichier s'appelle bien best+.pt dans votre dossier weights
model = YOLO("weights/best+.pt") 
# Exportation optimisée pour votre Intel i5
model.export(format="openvino", imgsz=1024, half=True)
print("🚀 MODÈLE ÉLITE PRÊT ! Vérifiez votre dossier weights/best+_openvino_model")
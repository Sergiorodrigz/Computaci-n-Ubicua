import cv2, json, os, numpy as np
from main import FacialRecognition

# 1) Instancia tu clase (ajusta rutas si tus modelos están en otra carpeta)
fr = FacialRecognition(
    det_proto='modelos/face_detector/deploy.prototxt',
    det_model='modelos/face_detector/res10_300x300_ssd_iter_140000.caffemodel',
    embed_model_path='modelos/nn4.small2.v1.t7',
    encoding_path='caras/encodings.json'
)

username = input("¿Cómo quieres que te llame? ").strip()
cap = cv2.VideoCapture(0)
print("Presiona ESPACIO para capturar tu cara (5 muestras)...")

samples = 0
while samples < 5:
    ret, frame = cap.read()
    cv2.imshow("Enfócate y pulsa ESPACIO", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        # Detecta con el DNN
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0,
                                     (300,300),(104.0,177.0,123.0))
        fr.face_detector.setInput(blob)
        dets = fr.face_detector.forward()

        # Elige la cara con mayor confianza
        best = None; best_conf = 0
        for i in range(dets.shape[2]):
            conf = float(dets[0,0,i,2])
            if conf > best_conf:
                best_conf, best = conf, dets[0,0,i,3:7]
        if best_conf < 0.5:
            print("No te detecto, inténtalo de nuevo.")
            continue

        # Recorta ROI, extrae embedding y guarda
        box = (best * np.array([w, h, w, h])).astype(int)
        x1,y1,x2,y2 = box
        face = frame[y1:y2, x1:x2]
        emb  = fr._get_embedding(face)
        data = {}
        if os.path.exists(fr.encoding_path):
            data = json.load(open(fr.encoding_path))
        data.setdefault(username, []).append(emb.tolist())
        json.dump(data, open(fr.encoding_path,'w'))
        samples += 1
        print(f"Muestra {samples}/5 tomada.")
cap.release()
cv2.destroyAllWindows()
print("¡Tu cara ya está en el sistema!")

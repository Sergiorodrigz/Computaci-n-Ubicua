import os
import cv2
import numpy as np
import json
import requests
import logging
import io
import base64
panel_img = cv2.imread('assets/info_panel.png', cv2.IMREAD_UNCHANGED)
if panel_img is None:
    raise FileNotFoundError("No se encontró assets/info_panel.png")
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse  
from pydantic import BaseModel

from google.oauth2 import service_account
from google.cloud import vision
from google.cloud import speech
from google.cloud import texttospeech

# ————— Configuración de APIs —————
SA_JSON = r"C:\Users\sergi\OneDrive\Escritorio\CUIA\backend\cuia-462109-33083f37da3e.json"
creds = service_account.Credentials.from_service_account_file(
    SA_JSON,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Clientes de Google Cloud
vision_client = vision.ImageAnnotatorClient(credentials=creds)
speech_client = speech.SpeechClient(credentials=creds)
tts_client = texttospeech.TextToSpeechClient(credentials=creds)

# ————— Configuración Gemini —————
GEMINI_API_KEY = "AIzaSyDLzfantGOW9oWjqzxz-FAB1xunTxTxvLI"
GEMINI_MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class FacialRecognition:
    def __init__(self,
                 data_dir='caras',
                 det_proto='modelos/face_detector/deploy.prototxt',
                 det_model='modelos/face_detector/res10_300x300_ssd_iter_140000.caffemodel',
                 embed_model_path='modelos/nn4.small2.v1.t7',
                 encoding_path='caras/encodings.json'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        # Detector Haar solo para localizar caras
        self.face_detector = cv2.dnn.readNetFromCaffe(det_proto, det_model)
        # Carga DNN de OpenFace para obtener embeddings de 128 dims
        self.embedder = cv2.dnn.readNetFromTorch(embed_model_path)
        self.embedder_path = embed_model_path
        # Ruta al JSON donde guardaremos los encodings
        self.encoding_path = encoding_path
        # Cargamos (o inicializamos) el diccionario de encodings
        self._load_encodings()

    def _load_encodings(self):
        if os.path.exists(self.encoding_path):
            with open(self.encoding_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        # Convertir listas a arrays numpy
        self.known = {
            user: [np.array(enc, dtype="float32") for enc in lst]
            for user, lst in data.items()
        }
    def _get_embedding(self, face_roi):
        blob = cv2.dnn.blobFromImage(face_roi, 1.0/255, (96,96),
                                     (0,0,0), swapRB=True, crop=False)
        # En vez de reutilizar self.embedder, carga un Net nuevo (seguro entre hilos)
        net = cv2.dnn.readNetFromTorch(self.embedder_path)
        net.setInput(blob)
        vec = net.forward().flatten()
        return vec / np.linalg.norm(vec)
    def recognize(self, frame):
        # 1) Detectar caras con DNN
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        # 2) Extraer rects válidas
        rects = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            # Clampeamos al tamaño de la imagen
            x1 = max(0, startX)
            y1 = max(0, startY)
            x2 = min(w - 1, endX)
            y2 = min(h - 1, endY)
            if x2 > x1 and y2 > y1:
                rects.append((x1, y1, x2 - x1, y2 - y1))

        # 3) Para cada cara extraemos embedding y comparamos
        names = []
        for (x, y, w_, h_) in rects:
            face = frame[y:y+h_, x:x+w_]
            # Si el recorte está vacío, saltamos
            if face.size == 0:
                continue

            # **aquí** vec siempre existe si seguimos
            vec = self._get_embedding(face)

            best = ("Desconocido", float('inf'))
            for user, enc_list in self.known.items():
                # apilamos los encodings en un array (N,128)
                encodings = np.vstack(enc_list)
                dists = np.linalg.norm(encodings - vec, axis=1)
                idx = np.argmin(dists)
                if dists[idx] < 0.6 and dists[idx] < best[1]:
                    best = (user, float(dists[idx]))
            names.append(best[0])

        return names, rects


# --------------------------------------------------
#  Streaming MJPEG con AR
# --------------------------------------------------
usuarios_actuales = []

@app.get("/face_recognition_status")
def face_recognition_status():
    global usuarios_actuales
    if latest_frame is None:
        return {"users": []}
    
    frame_np = np.frombuffer(latest_frame, np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

    usuarios_actuales, _ = face_recognition.recognize(frame)
    return {"users": usuarios_actuales}
latest_frame = None
latest_raw_frame = None
face_recognition = FacialRecognition()

@app.get("/video_feed")
def video_feed():
    global latest_frame
    # Uso de DirectShow para evitar errores MSMF en Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        raise HTTPException(500, "No se pudo abrir la cámara")

    def gen():
        global latest_raw_frame, latest_frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            latest_raw_frame = frame.copy()    
            # Detectar cuadros (como antes)
            quads = detect_paintings(frame)
            for corners in quads:
                corners_ordered = order_points(np.array(corners))
                frame = augment_with_panel(frame, corners_ordered, panel_img)

            # Detectar caras y reconocer usuarios
            usuarios, caras = face_recognition.recognize(frame)
            for (x,y,w,h), nombre in zip(caras, usuarios):
                color = (0,255,0) if nombre != "Desconocido" else (0,0,255)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, nombre, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            _, buf = cv2.imencode('.jpg', frame)
            latest_frame = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')

        cap.release()

    return StreamingResponse(gen(),
                             media_type='multipart/x-mixed-replace; boundary=frame')
@app.get("/latest_frame")
def get_latest_frame():
    global latest_frame
    if latest_frame is None:
        # Devuelve un JPEG en blanco de 1×1 píxel
        blank = np.zeros((1,1,3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', blank)
        return Response(content=buf.tobytes(), media_type="image/jpeg", status_code=status.HTTP_200_OK)
    return Response(content=latest_frame, media_type="image/jpeg")

class ChatRequest(BaseModel):
    message: str
    context: str = ""
    detected_artworks: list = []

class ChatResponse(BaseModel):
    response: str

class VoiceChatRequest(BaseModel):
    context: str = ""
    detected_artworks: list = []

class VoiceChatResponse(BaseModel):
    transcription: str
    response: str

# ————— Funciones de voz —————
def transcribe_audio(audio_data: bytes, audio_format: str = "webm") -> str:
    """Transcribir audio a texto usando Google Speech-to-Text"""
    try:
        # Logging para debugging
        logger.info(f"Recibido audio de {len(audio_data)} bytes, formato: {audio_format}")
        
        # Configurar el audio según el formato
        if audio_format.lower() in ["webm", "ogg"]:
            encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
            sample_rate = 48000
        elif audio_format.lower() in ["wav"]:
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
            sample_rate = 16000
        elif audio_format.lower() in ["mp3"]:
            encoding = speech.RecognitionConfig.AudioEncoding.MP3
            sample_rate = 16000
        else:
            # Fallback para formato desconocido
            encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
            sample_rate = 48000
        
        # Crear objetos de configuración
        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sample_rate,
            language_code="es-ES",
            alternative_language_codes=["es-MX", "es-AR"],
            enable_automatic_punctuation=True,
            model="latest_long",
            use_enhanced=True  # Usar modelo mejorado si está disponible
        )
        
        logger.info(f"Configuración de audio: encoding={encoding}, sample_rate={sample_rate}")
        
        # Realizar la transcripción
        response = speech_client.recognize(config=config, audio=audio)
        
        if response.results:
            transcription = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            logger.info(f"Audio transcrito con confianza {confidence}: {transcription}")
            return transcription
        else:
            logger.warning("No se pudo transcribir el audio - sin resultados")
            return ""
            
    except Exception as e:
        logger.error(f"Error detallado en transcripción: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ""
def text_to_speech(text: str) -> bytes:
    """Convertir texto a audio usando Google Text-to-Speech"""
    try:
        # Configurar la síntesis de voz
        input_text = texttospeech.SynthesisInput(text=text)
        
        # Configurar la voz (española)
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            name="es-ES-Standard-A"  # Voz femenina española
        )
        
        # Configurar el audio de salida
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        # Realizar la síntesis
        response = tts_client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config
        )
        
        logger.info(f"Audio generado para texto: {text[:50]}...")
        return response.audio_content
        
    except Exception as e:
        logger.error(f"Error en text-to-speech: {e}")
        return b""

# ————— Funciones de detección —————
def four_point_transform(img: np.ndarray, pts: list) -> np.ndarray:
    # Recorta el bounding-box del polígono
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # Recorta, con cuidado de no salirse de los límites
    return img[max(y_min, 0):min(y_max, img.shape[0]),
               max(x_min, 0):min(x_max, img.shape[1])]

def detect_paintings(img: np.ndarray, min_area: int = 500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quads = []
    ih, iw = img.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)
           
            if w < 50 or h < 50:
               continue
            if w > iw * 0.9 or h > ih * 0.9:
               continue
            if aspect < 0.3 or aspect > 3.0:
                continue

            pts = order_points(approx.reshape(4, 2))
            quads.append(pts.astype(int).tolist())

    return quads

def order_points(pts):
    """Ordenar puntos en sentido horario: arriba-izq, arriba-der, abajo-der, abajo-izq"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Suma de coordenadas: arriba-izq tendrá la suma más pequeña
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Diferencia de coordenadas: arriba-der tendrá la diferencia más pequeña
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def overlay_transparent(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    b_h, b_w = bg.shape[:2]
    f_h, f_w = fg.shape[:2]
    alpha = fg[:, :, 3] / 255.0
    for c in range(3):
        bg[:, :, c] = fg[:, :, c] * alpha + bg[:, :, c] * (1.0 - alpha)
    return bg

def augment_with_panel(frame: np.ndarray, corners: list, panel: np.ndarray) -> np.ndarray:
    h_p, w_p = panel.shape[:2]

    # Coord. originales del panel (rectángulo perfecto)
    src_pts = np.array([[0, 0], [w_p, 0], [w_p, h_p], [0, h_p]], dtype=np.float32)

    # Coord. destino son las esquinas del cuadro detectado
    dst_pts = np.array(corners, dtype=np.float32)

    # Calcula la homografía del panel a las esquinas del cuadro
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Deforma el panel con perspectiva y tamaño de frame original
    warped_panel = cv2.warpPerspective(panel, H, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)

    # Superpón el panel deformado sobre el frame original
    combined_frame = overlay_transparent(frame, warped_panel)

    return combined_frame


def identify_with_vision(crop_img: np.ndarray) -> str:
    """Identificar obra usando Google Vision API"""
    try:
        _, buf = cv2.imencode('.jpg', crop_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img = vision.Image(content=buf.tobytes())
        
        # Primero intentar con web detection
        web = vision_client.web_detection(image=img).web_detection
        if web.best_guess_labels:
            guess = web.best_guess_labels[0].label
            logger.info(f"Vision API identificó: {guess}")
            return guess
            
        # Si no hay web detection, usar label detection
        labels = vision_client.label_detection(image=img, max_results=3).label_annotations
        if labels:
            # Buscar etiquetas relacionadas con arte
            art_keywords = ['painting', 'art', 'artwork', 'canvas', 'portrait', 'landscape']
            for label in labels:
                if any(keyword in label.description.lower() for keyword in art_keywords):
                    logger.info(f"Vision API identificó (label): {label.description}")
                    return label.description
            
            # Si no hay etiquetas de arte, usar la primera etiqueta
            logger.info(f"Vision API identificó (primera etiqueta): {labels[0].description}")
            return labels[0].description
            
        return "Obra desconocida"
        
    except Exception as e:
        logger.error(f"Error en Vision API: {e}")
        return "Error en identificación"

def identify_with_gemini(name: str) -> dict:
    """Función para usar la API de Gemini"""
    prompt_text = (
        f'Analiza esta obra de arte: "{name}". '
        "Proporciona información sobre el título, autor y una breve descripción. "
        "Responde ÚNICAMENTE con un JSON válido en este formato exacto:\n"
        '{"title": "título de la obra", "author": "nombre del autor", "description": "breve descripción de 1-2 líneas"}\n'
        "No agregues texto adicional fuera del JSON."
    )
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 20,
            "topP": 0.8,
            "maxOutputTokens": 500,
        }
    }
    
    url = f"{GEMINI_MODEL_URL}?key={GEMINI_API_KEY}"
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Gemini response: {result}")
        
        # Extraer el texto de la respuesta
        if 'candidates' in result and result['candidates']:
            content = result['candidates'][0]['content']['parts'][0]['text']
            
            # Limpiar el contenido para obtener solo el JSON
            content = content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            # Intentar parsear el JSON
            try:
                parsed_data = json.loads(content)
                logger.info(f"Gemini parsed data: {parsed_data}")
                return parsed_data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON: {e}, content: {content}")
                # Intentar extraer información básica del texto
                return extract_fallback_info(content, name)
        else:
            logger.error("No candidates in Gemini response")
            return create_fallback_response(name)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Gemini API: {e}")
        return create_fallback_response(name)

def extract_fallback_info(content: str, original_name: str) -> dict:
    """Extraer información básica si el JSON falla"""
    try:
        # Intentar extraer información básica del texto
        lines = content.split('\n')
        title = original_name
        author = "Autor desconocido"
        description = "Información no disponible"
        
        for line in lines:
            line = line.strip()
            if 'title' in line.lower() or 'título' in line.lower():
                if ':' in line:
                    title = line.split(':', 1)[1].strip(' "')
            elif 'author' in line.lower() or 'autor' in line.lower():
                if ':' in line:
                    author = line.split(':', 1)[1].strip(' "')
            elif 'description' in line.lower() or 'descripción' in line.lower():
                if ':' in line:
                    description = line.split(':', 1)[1].strip(' "')
        
        return {
            "title": title,
            "author": author,
            "description": description
        }
    except:
        return create_fallback_response(original_name)

def create_fallback_response(name: str) -> dict:
    """Crear respuesta por defecto"""
    return {
        "title": name,
        "author": "Autor desconocido",
        "description": "Información no disponible en este momento"
    }

def chat_with_gemini(message: str, context: str = "") -> str:
    """Función para el chat usando la API de Gemini"""
    system_prompt = (
        "Eres un asistente especializado en historia del arte. "
        "Responde de manera concisa, educativa y amigable. "
        "Si te preguntan sobre obras de arte detectadas, usa el contexto proporcionado. "
        "Mantén tus respuestas en español y enfócate en información artística e histórica relevante. "
        "Las respuestas deben ser apropiadas para ser escuchadas por voz, evita listas largas o formatos complejos."
    )
    
    full_prompt = f"{system_prompt}\n\nContexto: {context}\n\nPregunta del usuario: {message}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": full_prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 800,
        }
    }
    
    url = f"{GEMINI_MODEL_URL}?key={GEMINI_API_KEY}"
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        
        result = response.json()
        
        if 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Lo siento, no pude procesar tu pregunta en este momento."
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in chat with Gemini: {e}")
        return "Lo siento, ocurrió un error al procesar tu pregunta. Por favor, intenta de nuevo."

# ————— Endpoints —————
@app.get("/")
async def root():
    return {"message": "API de detección y chat de arte con funcionalidades de voz"}

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    quads = detect_paintings(img)  # tu función de detección

    resultados = []
    for corners in quads:
        # 1) Recortar con la función que acabamos de definir
        crop = four_point_transform(img, corners)

        # 2) Identificar (síncrono)
        nombre = identify_with_vision(crop)

        # 3) Obtener metadatos de Gemini SIN await
        detalles = identify_with_gemini(nombre)

        resultados.append({
            "title":       detalles.get("title", ""),
            "author":      detalles.get("author", ""),
            "description": detalles.get("description", "")
        })

    
    return JSONResponse(content={"cuadros": resultados})
@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Endpoint para chat de texto (mantenido para compatibilidad)"""
    try:
        response_text = chat_with_gemini(req.message, req.context)
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(response="Lo siento, ocurrió un error al procesar tu pregunta.")
@app.post("/voice-chat/")
async def voice_chat_endpoint(
    audio: UploadFile = File(...),
    context: str = "",
    detected_artworks: str = "[]"
):
    """Endpoint para chat por voz con mejor manejo de errores"""
    try:
        logger.info(f"Recibido archivo de audio: {audio.filename}, tipo: {audio.content_type}")
        
        # Validar que sea un archivo de audio
        if not audio.content_type or not audio.content_type.startswith("audio/"):
            logger.error(f"Tipo de archivo inválido: {audio.content_type}")
            raise HTTPException(400, f"Tipo de archivo inválido: {audio.content_type}. Debes subir un archivo de audio.")
        
        # Leer el archivo de audio
        audio_data = await audio.read()
        logger.info(f"Archivo de audio leído: {len(audio_data)} bytes")
        
        if len(audio_data) == 0:
            raise HTTPException(400, "El archivo de audio está vacío")
        
        # Transcribir audio a texto
        audio_format = audio.content_type.split('/')[-1] if audio.content_type else "webm"
        transcription = transcribe_audio(audio_data, audio_format)
        
        if not transcription:
            logger.warning("Transcripción vacía")
            raise HTTPException(400, "No se pudo transcribir el audio. Intenta hablar más claro o revisar el micrófono.")
        
        logger.info(f"Transcripción exitosa: {transcription}")
        
        # Procesar la pregunta con Gemini
        try:
            response_text = chat_with_gemini(transcription, context)
            logger.info(f"Respuesta de Gemini obtenida: {response_text[:100]}...")
        except Exception as e:
            logger.error(f"Error con Gemini: {e}")
            response_text = "Lo siento, ocurrió un error al procesar tu pregunta."
        
        # Convertir respuesta a audio
        try:
            audio_response = text_to_speech(response_text)
            if not audio_response:
                logger.warning("No se pudo generar audio de respuesta")
                # Continuar sin audio
                audio_base64 = ""
            else:
                audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                logger.info(f"Audio de respuesta generado: {len(audio_response)} bytes")
        except Exception as e:
            logger.error(f"Error generando audio de respuesta: {e}")
            audio_base64 = ""
        
        # Retornar respuesta
        return {
            "transcription": transcription,
            "response": response_text,
            "audio": audio_base64
        }
        
    except HTTPException:
        # Re-raise HTTPExceptions para que FastAPI las maneje correctamente
        raise
    except Exception as e:
        logger.error(f"Error inesperado en voice chat endpoint: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise HTTPException(500, f"Error interno del servidor: {str(e)}")
@app.get("/artworks_status")
def artworks_status():
    global latest_raw_frame
    # Si aún no hay frame crudo, no hay obras
    if latest_raw_frame is None:
        return JSONResponse({"artworks": []})

    # Usa el frame ORIGINAL (sin UI) para detectar y recortar
    frame = latest_raw_frame.copy()

    # Detecta los contornos de las pinturas
    quads = detect_paintings(frame)

    artworks = []
    for corners in quads:
        # Recorte con perspectiva en el frame crudo
        crop = four_point_transform(frame, corners)

        # 1) Identifica el nombre con Vision API (imagen limpia)
        name = identify_with_vision(crop)

        # 2) Consulta a Gemini para título, autor y descripción
        details = identify_with_gemini(name)
        artworks.append({
            "title":       details.get("title", name),
            "author":      details.get("author", "Autor desconocido"),
            "description": details.get("description", "")
        })

    return JSONResponse({"artworks": artworks})
@app.post("/speech-to-text/")
async def speech_to_text_endpoint(audio: UploadFile = File(...)):
    """Endpoint para convertir voz a texto con mejor manejo de errores"""
    try:
        logger.info(f"Endpoint speech-to-text - Archivo: {audio.filename}, Tipo: {audio.content_type}")
        
        if not audio.content_type or not audio.content_type.startswith("audio/"):
            raise HTTPException(400, f"Tipo de archivo inválido: {audio.content_type}")
        
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(400, "El archivo de audio está vacío")
        
        audio_format = audio.content_type.split('/')[-1]
        transcription = transcribe_audio(audio_data, audio_format)
        
        if not transcription:
            raise HTTPException(400, "No se pudo transcribir el audio")
        
        return {"transcription": transcription}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en speech-to-text endpoint: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Error transcribiendo audio: {str(e)}")
@app.post("/add_user/{username}")
async def add_user(username: str, file: UploadFile = File(...)):
    user_path = os.path.join(face_recognition.data_dir, username)
    os.makedirs(user_path, exist_ok=True)

    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1) localiza la cara (puede haber más de una, pero tomamos la primera)
    # 1) Localiza cara con DNN detector
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_recognition.face_detector.setInput(blob)
    detections = face_recognition.face_detector.forward()
    rects = []
    for i in range(detections.shape[2]):
        if float(detections[0, 0, i, 2]) < 0.5:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        rects.append((x1, y1, x2 - x1, y2 - y1))
    if not rects:
        raise HTTPException(400, "No se detectó ninguna cara en la imagen.")
    x,y,w_,h_ = rects[0]
    face = img[y:y+h_, x:x+w_]
    emb = face_recognition._get_embedding(face)
    # Leer o crear JSON
    if os.path.exists(face_recognition.encoding_path):
        with open(face_recognition.encoding_path,'r') as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault(username, []).append(emb.tolist())
    with open(face_recognition.encoding_path,'w') as f:
        json.dump(data, f)
    return {"status": "Usuario añadido correctamente"}
@app.post("/text-to-speech/")
async def text_to_speech_endpoint(text: str):
    """Endpoint para convertir texto a voz con mejor manejo de errores"""
    try:
        if not text or len(text.strip()) == 0:
            raise HTTPException(400, "El texto no puede estar vacío")
        
        logger.info(f"Generando audio para texto: {text[:50]}...")
        
        audio_data = text_to_speech(text)
        
        if not audio_data:
            raise HTTPException(500, "Error generando el audio")
        
        # Retornar el audio como stream
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en text-to-speech endpoint: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Error generando audio: {str(e)}")
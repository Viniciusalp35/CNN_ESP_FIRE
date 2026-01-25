import numpy as np
import cv2
import tensorflow as tf
from utils.gamma import gamma_correction

MODEL_PATH = "./train/model_fire_int8.tflite"
LABELS = ["Nao fogo", "Fogo"]
THRESHOLD = 0.50
HEIGHT = 96
WIDTH = 96
BOX_SIZE = 300


def classification_loop():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Erro ao carregar o modelo '{MODEL_PATH}': {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]
    print("Modelo Carregado")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a camera")
        return

    print("Pressione 'q' para sair")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        h_frame, w_frame, _ = frame.shape
        x1 = (w_frame - BOX_SIZE) // 2
        y1 = (h_frame - BOX_SIZE) // 2
        x2 = x1 + BOX_SIZE
        y2 = y1 + BOX_SIZE

        roi = frame[y1:y2, x1:x2]

        roi = gamma_correction(roi, gamma=0.1)
        img = cv2.resize(roi, (WIDTH, HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.expand_dims(img, axis=-1)
        input_data = np.expand_dims(img, 0).astype(np.uint8)  # / 255

        interpreter.set_tensor(tensor_index=input_details[0]["index"], value=input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])

        raw_score = output_data[0][0]

        # Se a saída for INT8 (0 a 255), normaliza para 0.0 a 1.0
        if output_dtype == np.uint8:
            raw_score = raw_score / 255.0

        confidence = float(raw_score)

        if confidence > THRESHOLD:
            color = (0, 255, 0)
            label = f"{LABELS[1]} ({confidence:.2%})"
        else:
            color = (0, 0, 255)
            label = f"{LABELS[0]}({confidence:.2%})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2
        )

        input_debug = cv2.resize(img, (150, 150))
        input_debug = cv2.cvtColor(input_debug, cv2.COLOR_RGB2BGR)
        frame[0:150, 0:150] = input_debug

        cv2.imshow("Detector de Fogo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    classification_loop()

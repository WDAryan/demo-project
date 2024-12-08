from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

model_path = 'model/koil.keras'
print("Loading model from:", os.path.abspath(model_path))
model = load_model(model_path)

def prepare_image(image):
    image = Image.open(io.BytesIO(image)).convert('L')
    image = image.resize((28, 28))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image.astype('float32') / 255.0
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"].read()
        image = prepare_image(file)
        prediction = model.predict(image)
        predicted_label = np.argmax(prediction)
        return render_template("index.html", prediction=predicted_label)
    return render_template("index.html", prediction="")

if __name__ == "__main__":
    app.run(debug=True)

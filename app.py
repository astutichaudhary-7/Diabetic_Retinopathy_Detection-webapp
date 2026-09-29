import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from validator import is_retina_image
from model import predict_grade, save_gradcam_overlay

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    if file.filename == "":
        return render_template(
            "invalid.html",
            message="No image selected. Please choose a retinal fundus image."
        )

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    retina_ok, gate_confidence = is_retina_image(filepath)
    if not retina_ok:
        return render_template(
            "invalid.html",
            message="This does not look like a retinal fundus image. Please upload a valid retina scan.",
            confidence=round(gate_confidence * 100, 2),
            image="uploads/" + filename
        )

    prediction, confidence, img_array, pred_index = predict_grade(filepath)

    gradcam_filename = "gradcam_" + filename
    gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_filename)
    save_gradcam_overlay(filepath, gradcam_path, img_array, pred_index)

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=round(confidence, 2),
        image="uploads/" + filename,
        gradcam_image="gradcam/" + gradcam_filename
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
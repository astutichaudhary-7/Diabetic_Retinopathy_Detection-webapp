Diabetic Retinopathy Detection Web Application
A web application that analyzes retinal fundus images and predicts the severity
of Diabetic Retinopathy (DR) using a deep learning model.

What it does?
Upload a retinal (fundus) image, and the app classifies it into one of five
severity grades:
- No_DR
- Mild
- Moderate
- Severe
- Proliferate_DR

Along with the prediction, it shows a confidence score and a Grad-CAM heatmap
highlighting the region of the retina the model focused on while making its
decision.

Key features
- Gatekeeper model — before running the main classifier, a separate
  MobileNetV2-based model checks whether the uploaded image is actually a
  retina scan, and rejects anything else (selfies, screenshots, random
  photos) with a clear message.
- EfficientNetV2S classifier — the main prediction model, fine-tuned on
  the Diabetic Retinopathy dataset.
- Grad-CAM explainability — generates a heatmap overlay showing which
  part of the retina influenced the prediction, instead of giving a plain
  black-box result.

How I got here?
I didn't land on this setup on the first try. I started with a basic CNN
built from scratch, then tried a pretrained ConvNeXt-Tiny model through
transfer learning to improve accuracy — but it kept confusing classes (for
example, predicting Mild or Severe images as No_DR) and stayed under 60%
accuracy even after 12 epochs. I eventually rebuilt the classifier on
EfficientNetV2S, fine-tuned the whole backbone instead of just the top
layers, and trained it on Google Colab's GPU, which fixed the confusion
issue and pushed validation accuracy above 80%.

Tech stack
- Backend: Python, Flask
- Deep Learning: TensorFlow / Keras, EfficientNetV2S, MobileNetV2
- Explainability: Grad-CAM (OpenCV)
- Frontend: HTML, CSS, Bootstrap
- Training: Google Colab (GPU)

Project structure
├── app.py                  # Flask app
├── model.py                # Loads the classifier + Grad-CAM logic
├── validator.py            # Gatekeeper model logic
├── preprocess.py           # Data preprocessing
├── train_model.py          # Baseline CNN training script
├── train_gatekeeper.py     # Gatekeeper model training script
├── model/                  # Trained model weights
├── templates/              # HTML pages (index, result, invalid)
├── static/                 # CSS, uploaded images, Grad-CAM outputs
└── requirements.txt

Running it locally
pip install -r requirements.txt
python app.py

Then open "http://127.0.0.1:5000/" in your browser.

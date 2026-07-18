from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# FLASK CONFIGURATION

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# CLASS LABELS

class_names = [
    "Eczema",
    "Psoriasis",
    "Tinea Pedis"
]

# LOAD MODEL

model = models.mobilenet_v3_large(weights=None)

num_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    num_features,
    3
)

model.load_state_dict(
    torch.load(
        "model/best_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# IMAGE PREPROCESSING

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# HOME PAGE

@app.route("/")
def home():
    return render_template("index.html")

# PREDICTION

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            prediction="Tidak ada gambar yang dipilih",
            confidence=None,
            description=""
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            prediction="Tidak ada gambar yang dipilih",
            confidence=None,
            description=""
        )

    image = Image.open(file).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    confidence = confidence.item() * 100

    # THRESHOLD

    if confidence < 85:

        prediction = "Jenis Penyakit Tidak Ditemukan"

        confidence_display = None

        description = """
        Gambar yang diunggah tidak sesuai dengan kategori penyakit apa pun
        di dalam sistem, harap unggah gambar kembali.
        """

    else:

        prediction = class_names[predicted.item()]

        confidence_display = f"{confidence:.2f}%"

        if prediction == "Eczema":

            description = """
            Hindari sabun berbahan kimia keras yang mengandung parfum atau alkohol, detergen baju yang terlalu keras, mandi dengan air yang terlalu panas, stres psikologis, dan bahan pakaian yang kasar seperti wol.

            Rekomendasi Obat Umum:
            Krim Kortikosteroid seperti Hydrocortisone untuk gejala ringan atau Betamethasone untuk gejala lebih berat untuk meredakan radang dan gatal.
            Antihistamin seperti Cetirizine atau Loratadine untuk mengurangi rasa gatal yang mengganggu tidur.

            Perawatan Dasar:
            Wajib menggunakan pelembap khusus kulit sensitif sesering mungkin.
            """

        elif prediction == "Psoriasis":

            description = """
            Mengelola stres dengan baik, menjaga kelembapan kulit jangan membiarkan kulit sampai kering, menghindari cedera kulit seperti goresan atau tato, tidak merokok, membatasi alkohol, dan berjemur di bawah matahari pagi karna sinar UV alami dapat membantu meredakan psoriasis, tetapi jangan sampai terbakar.

            Rekomendasi Obat Umum:
            Salep Kortikosteroid poten, salep Analog Vitamin D (Calcipotriol) atau salep berbasis Ter batu bara (Coal Tar) untuk memperlambat pertumbuhan sel kulit.

            Obat minum/suntik golongan imunosupresan (Methotrexate, Cyclosporine) atau terapi agen biologis di bawah pengawasan dokter spesialis kulit.
            """

        elif prediction == "Tinea Pedis":

            description = """
            Hindari memakai sepatu tertutup saat kaki masih basah atau lembab, jangan memakai kaus kaki yang sama berhari-hari, jangan berbagi handuk atau sandal dengan orang lain, dan hindari menggaruk area yang gatal karena jamur bisa berpindah ke kuku atau tangan.

            Rekomendasi Obat Umum:
            Salep/Krim Antijamur yang dijual bebas atau dengan resep seperti Clotrimazole, Miconazole, Ketoconazole, atau Terbinafine. Obat harus dioleskan secara rutin hingga 1–2 minggu setelah gejala hilang agar jamur benar-benar mati total.

            Obat antijamur minum (Itraconazole atau Fluconazole) jika infeksinya sudah sangat luas atau menyerang kuku kaki.
            """

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence_display,
        description=description,
        show_result=True    
    )


@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template(
        "index.html",
        prediction="Ukuran gambar terlalu besar",
        confidence=None,
        description="Ukuran maksimum gambar yang dapat diunggah adalah 5 MB."
    ), 413


# RUN FLASK

if __name__ == "__main__":
    app.run(debug=True)
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Load model yang sudah disimpan
model = load_model("models.h5", compile=False)

# Load labels dari file JSON
with open("labels.json", "r") as f:
    class_indices = json.load(f)

# Balik mapping {label: index} menjadi {index: label}
labels = {v: k for k, v in class_indices.items()}

# Dictionary informasi tambahan rempah
rempah_info = {
    "adas": {
        
        "Nama": "Adas (Foeniculum vulgare)",
        "Aroma": "Manis, sedikit pedas, dengan aroma khas seperti licorice.",
        "Rasa": "Manis dan sedikit pahit.",
        "Kegunaan": "Sebagai bumbu dalam masakan khas India dan Timur Tengah, Digunakan dalam campuran jamu dan teh herbal.",
        "Manfaat Kesehatan": "Membantu melancarkan pencernaan, Dapat meredakan batuk dan masuk angin."
    },

    "andaliman": {
        
        "Nama": "Andaliman (Zanthoxylum acanthopodium)",
        "Aroma": "Citrus yang khas dengan sensasi pedas menggigit",
        "Rasa": "Pedas bergetar (mirip merica Sichuan).",
        "Kegunaan": "Sebagai bumbu khas masakan Batak seperti arsik dan saksang, Digunakan dalam masakan pedas untuk memberikan sensasi getir unik.",
        "Manfaat Kesehatan": "Membantu meningkatkan nafsu makan, Memiliki sifat antibakteri alami."
    },

    "asam jawa": {
        
        "Nama": "Asam Jawa (Tamarindus indica)",
        "Aroma": "Asam khas dengan sedikit aroma buah tropis.",
        "Rasa": "Asam segar, sedikit manis jika matang.",
        "Kegunaan": "Sebagai bahan utama dalam masakan seperti sayur asam dan sambal asam, Digunakan dalam minuman herbal untuk menyegarkan tubuh.",
        "Manfaat Kesehatan": "Membantu melancarkan pencernaan, Kaya akan antioksidan dan vitamin C."
    },

    "bawang bombai": {
        
        "Nama": "Bawang Bombai (Allium cepa)",
        "Aroma": "Tajam ketika mentah, tetapi manis setelah dimasak.",
        "Rasa": "Pahit ringan saat mentah, manis ketika dikaramelisasi.",
        "Kegunaan": "Digunakan dalam berbagai masakan internasional sebagai bumbu dasar, Biasa dijadikan bahan tumisan, sup, dan saus.",
        "Manfaat Kesehatan": "Mengandung antioksidan yang baik untuk kesehatan jantung, Dapat meningkatkan sistem kekebalan tubuh."
    },

    "bawang merah": {
        
        "Nama": "Bawang Merah (Allium cepa var. aggregatum)",
        "Aroma": "Tajam saat mentah, harum setelah dimasak.",
        "Rasa": "Sedikit pedas dan manis setelah dimasak.",
        "Kegunaan": "Digunakan sebagai bumbu dasar dalam berbagai masakan Nusantara, Dapat diolah menjadi bawang goreng sebagai pelengkap hidangan.",
        "Manfaat Kesehatan": "Membantu menurunkan kadar kolesterol, Memiliki sifat antiinflamasi alami."
    },

    "bawang putih": {
        
        "Nama": "Bawang Putih (Allium sativum)",
        "Aroma": "Kuat dan menyengat, khas bawang.",
        "Rasa": "Pedas dan tajam saat mentah, lebih lembut setelah dimasak.",
        "Kegunaan": "Sebagai bumbu utama dalam berbagai jenis masakan, Dapat digunakan dalam saus, tumisan, dan acar.",
        "Manfaat Kesehatan": "Dapat membantu menurunkan tekanan darah dan kolesterol, Mengandung senyawa antibakteri alami."
    },

    "biji ketumbar": {
        
        "Nama": "Biji Ketumbar (Coriandrum sativum)",
        "Aroma": "Hangat, sedikit citrus, dan rempah yang khas.",
        "Rasa": "Gurih dengan sentuhan manis.",
        "Kegunaan": "Digunakan dalam masakan berempah seperti kari dan gulai, Sebagai bahan utama dalam pembuatan bumbu dapur seperti bumbu rendang dan garam masala.",
        "Manfaat Kesehatan": "Membantu melancarkan pencernaan, Memiliki sifat antioksidan yang baik untuk tubuh."
    }, 

    "bunga lawang": {
        "Nama": "Bunga Lawang (Illicium verum)",
        "Aroma": "Manis dan pedas dengan aroma khas seperti licorice.",
        "Rasa": "Manis, sedikit pahit, dan pedas.",
        "Kegunaan": "Digunakan dalam masakan Asia seperti sup dan kari., Bahan utama dalam campuran bumbu rempah seperti five-spice powder.",
        "Manfaat Kesehatan": "Mengandung antioksidan yang baik untuk sistem imun, Membantu meredakan masalah pencernaan dan kembung."
    },

    "cengkeh": {
        "Nama": "Cengkeh (Syzygium aromaticum)",
        "Aroma": "Pedas, manis, dan harum yang kuat.",
        "Rasa": "Pedas dan sedikit pahit.",
        "Kegunaan": "Digunakan dalam masakan seperti rendang dan kari, Sering digunakan dalam minuman tradisional seperti wedang jahe.",
        "Manfaat Kesehatan": "Mengandung senyawa eugenol yang bersifat antiinflamasi, Dapat membantu meredakan sakit gigi dan infeksi mulut."
    },

    "daun jeruk": {
        "Nama": "Daun Jeruk (Citrus hystrix)",
        "Aroma": "Segar, citrusy, dan harum.",
        "Rasa": "Segar dengan sedikit rasa pahit.",
        "Kegunaan": "Digunakan dalam berbagai masakan seperti soto dan gulai, Menambah aroma segar pada hidangan ikan dan ayam.",
        "Manfaat Kesehatan": "Mengandung vitamin C yang baik untuk daya tahan tubuh, Membantu melancarkan pencernaan dan mengurangi bau badan."
    },

    "daun kemangi": {
        "Nama": "Daun Kemangi (Ocimum basilicum)",
        "Aroma": "Segar, wangi, dan sedikit pedas.",
        "Rasa": "Manis dan sedikit pahit.",
        "Kegunaan": "Biasa dikonsumsi sebagai lalapan atau pelengkap hidangan, Digunakan dalam masakan Thailand dan Indonesia seperti pepes.",
        "Manfaat Kesehatan": "Mengandung antioksidan tinggi yang baik untuk kesehatan kulit, Membantu meredakan stres dan meningkatkan relaksasi."
    },

    "daun ketumbar": {
        "Nama": "Daun Ketumbar (Coriandrum sativum)",
        "Aroma": "Segar, citrusy, dan sedikit rempah.",
        "Rasa": "Segar dengan sedikit rasa pahit.",
        "Kegunaan": "Biasa digunakan dalam masakan Meksiko, India, dan Indonesia, Menambah cita rasa segar dalam sup dan salad.",
        "Manfaat Kesehatan": "Membantu menurunkan kadar gula darah, Mengandung antioksidan yang dapat melawan infeksi bakteri."
    },

    "daun salam": {
        "Nama": "Daun Salam (Syzygium polyanthum)",
        "Aroma": "Harum dengan sedikit aroma kayu dan rempah.",
        "Rasa": "Sedikit pahit dan rempah.",
        "Kegunaan": "Digunakan sebagai bumbu dalam masakan seperti rendang dan gulai, Ditambahkan ke dalam rebusan daging untuk mengurangi bau amis.",
        "Manfaat Kesehatan": "Membantu menurunkan kadar kolesterol, Dapat mengontrol kadar gula darah bagi penderita diabetes."
    },

    "jahe": {
        "Nama": "Jahe (Zingiber officinale)",
        "Aroma": "Pedas, hangat, dan menyegarkan.",
        "Rasa": "Pedas dan sedikit manis.",
        "Kegunaan": "Digunakan dalam berbagai masakan dan minuman seperti wedang jahe, Bahan utama dalam jamu dan obat tradisional.",
        "Manfaat Kesehatan": "Membantu menghangatkan tubuh dan meningkatkan sirkulasi darah, Dapat meredakan mual dan masalah pencernaan."
    },

    "jinten": {
        "Nama": "Jinten (Cuminum cyminum)",
        "Aroma": "Aroma kuat, rempah-rempah, dan sedikit pahit.",
        "Rasa": "Pedas, sedikit pahit, dan beraroma khas.",
        "Kegunaan": "Sering digunakan dalam masakan Timur Tengah dan India, Menambah cita rasa dalam sup dan kari.",
        "Manfaat Kesehatan": "Membantu melancarkan pencernaan dan mengurangi kembung, Mengandung zat besi yang baik untuk kesehatan darah."
    },

    "kapulaga": {
        "Nama": "Kapulaga (Elettaria cardamomum)",
        "Aroma": "Manis, rempah-rempah, dan sedikit minty.",
        "Rasa": "Pedas dan manis dengan aroma khas.",
        "Kegunaan": "Digunakan dalam masakan dan minuman seperti teh tarik dan kari, Sering menjadi bahan dalam jamu dan ramuan herbal.",
        "Manfaat Kesehatan": "Membantu meningkatkan kesehatan jantung, Dapat mengurangi tekanan darah dan melancarkan pernapasan."
    },

    "kayu manis": {
        "Nama": "Kayu Manis (Cinnamomum verum)",
        "Aroma": "Manis, hangat, dan khas.",
        "Rasa": "Manis dengan sedikit pedas.",
        "Kegunaan": "Bahan utama dalam pembuatan kue dan minuman seperti kopi rempah, Digunakan dalam masakan seperti kari dan gulai.",
        "Manfaat Kesehatan": "Membantu mengontrol kadar gula darah, Mengandung antioksidan tinggi yang baik untuk sistem imun."
    },

    "kayu secang": {
        "Nama": "Kayu Secang (Caesalpinia sappan)",
        "Aroma": "Aroma kayu yang khas dengan sedikit aroma manis.",
        "Rasa": "Sedikit pahit dan sepat.",
        "Kegunaan": "Digunakan sebagai pewarna alami untuk minuman tradisional seperti wedang secang, Sering digunakan dalam jamu dan ramuan herbal.",
        "Manfaat Kesehatan": "Mengandung antioksidan tinggi yang baik untuk sistem imun, Membantu melancarkan peredaran darah dan meningkatkan kesehatan jantung."
    },

    "kemiri": {
        "Nama": "Kemiri (Aleurites moluccanus)",
        "Aroma": "Ringan dengan sedikit aroma kacang.",
        "Rasa": "Gurih dan sedikit pahit.",
        "Kegunaan": "Digunakan sebagai bumbu dasar dalam masakan seperti opor dan rendang, Sering digunakan dalam pembuatan minyak rambut tradisional.",
        "Manfaat Kesehatan": "Membantu menjaga kesehatan rambut dan kulit kepala, Mengandung lemak sehat yang baik untuk jantung."
    },

    "kemukus": {
        "Nama": "Kemukus (Piper cubeba)",
        "Aroma": "Pedas dan sedikit mirip lada hitam.",
        "Rasa": "Pedas dengan sedikit rasa pahit.",
        "Kegunaan": "Digunakan dalam campuran bumbu untuk masakan dan jamu, Bahan utama dalam ramuan herbal tradisional.",
        "Manfaat Kesehatan": "Membantu meredakan gangguan pernapasan, Memiliki sifat antibakteri dan antiinflamasi."
    },

    "kencur": {
        "Nama": "Kencur (Kaempferia galanga)",
        "Aroma": "Aroma khas yang kuat, segar, dan pedas.",
        "Rasa": "Pedas dan sedikit pahit.",
        "Kegunaan": "Digunakan dalam bumbu masakan seperti urap dan pecel, Bahan utama dalam jamu beras kencur.",
        "Manfaat Kesehatan": "Membantu meningkatkan nafsu makan, Memiliki sifat antiinflamasi dan baik untuk pencernaan."
    },

    "kluwek": {
        "Nama": "Kluwek (Pangium edule)",
        "Aroma": "Sedikit asam dengan aroma khas kacang fermentasi.",
        "Rasa": "Pahit jika mentah, tetapi gurih dan kaya rasa setelah diolah.",
        "Kegunaan": "Bumbu utama dalam masakan rawon dan brongkos, Digunakan dalam beberapa hidangan khas Indonesia untuk memberikan warna hitam alami.",
        "Manfaat Kesehatan": "Mengandung antioksidan yang baik untuk tubuh, Membantu meningkatkan sistem kekebalan tubuh."
    },

    "kunyit": {
        "Nama": "Kunyit (Curcuma longa)",
        "Aroma": "Hangat, sedikit pahit, dan khas rempah-rempah.",
        "Rasa": "Pahit dengan sedikit pedas.",
        "Kegunaan": "Bahan utama dalam masakan seperti kari dan gulai, Digunakan dalam minuman jamu kunyit asam.",
        "Manfaat Kesehatan": "Mengandung kurkumin yang bersifat antiinflamasi, Membantu meningkatkan kesehatan hati dan sistem pencernaan."
    },

    "lada": {
        "Nama": "Lada (Piper nigrum)",
        "Aroma": "Pedas, hangat, dan kuat.",
        "Rasa": "Pedas dan sedikit pahit.",
        "Kegunaan": "Digunakan dalam hampir semua masakan sebagai penyedap rasa, Bahan utama dalam berbagai campuran bumbu.",
        "Manfaat Kesehatan": "Membantu meningkatkan metabolisme tubuh, Memiliki sifat antibakteri yang baik untuk kesehatan usus."
    },

    "lengkuas": {
        "Nama": "Lengkuas (Alpinia galanga)",
        "Aroma": "Segar, tajam, dan sedikit pedas.",
        "Rasa": "Pedas dengan sedikit pahit dan getir.",
        "Kegunaan": "Digunakan dalam masakan seperti rendang dan soto, Bahan utama dalam ramuan jamu tradisional.",
        "Manfaat Kesehatan": "Membantu meredakan masalah pencernaan, Memiliki sifat antibakteri dan antioksidan."
    },

    "pala": {
        "Nama": "Pala (Myristica fragrans)",
        "Aroma": "Manis, hangat, dan khas rempah-rempah.",
        "Rasa": "Manis dan sedikit pedas.",
        "Kegunaan": "Digunakan dalam masakan dan minuman seperti wedang pala, Bahan utama dalam pembuatan kue dan saus.",
        "Manfaat Kesehatan": "Membantu meningkatkan kualitas tidur, Memiliki sifat antidepresan dan meningkatkan suasana hati."
    },

    "saffron": {
        "Nama": "Saffron (Crocus sativus)",
        "Aroma": "Manis, bunga, dan sedikit logam.",
        "Rasa": "Sedikit pahit dengan rasa khas bunga.",
        "Kegunaan": [
            "Digunakan sebagai pewarna alami dalam masakan seperti nasi biryani.",
            "Bahan utama dalam berbagai minuman kesehatan."
        ],
        "Manfaat Kesehatan": [
            "Membantu meningkatkan suasana hati dan mengurangi stres.",
            "Memiliki sifat antioksidan tinggi yang baik untuk kesehatan otak."
        ]
    },

    "serai": {
        "Nama": "Serai (Cymbopogon citratus)",
        "Aroma": "Segar, citrusy, dan sedikit pedas.",
        "Rasa": "Segar dengan sedikit pahit dan pedas.",
        "Kegunaan": "Digunakan dalam masakan seperti tom yum dan gulai, Bahan utama dalam teh herbal dan jamu.",
        "Manfaat Kesehatan": "Membantu menenangkan sistem saraf dan mengurangi stres, Memiliki sifat antibakteri dan baik untuk pencernaan."
    },

    "vanili": {
        "Nama": "Vanili (Vanilla planifolia)",
        "Aroma": "Manis, lembut, dan khas.",
        "Rasa": "Manis dan sedikit creamy.",
        "Kegunaan": "Digunakan sebagai bahan utama dalam pembuatan kue dan es krim, Menambah aroma dalam minuman seperti kopi dan teh.",
        "Manfaat Kesehatan": "Membantu mengurangi stres dan meningkatkan suasana hati, Mengandung antioksidan yang baik untuk kesehatan kulit."
    },

    "wijen": {
        "Nama": "Wijen (Sesamum indicum)",
        "Aroma": "Gurih, kacang-kacangan, dan sedikit manis.",
        "Rasa": "Gurih dan sedikit pahit.",
        "Kegunaan": "Digunakan dalam berbagai hidangan seperti sushi dan salad, Bahan utama dalam pembuatan minyak wijen.",
        "Manfaat Kesehatan": "Mengandung lemak sehat yang baik untuk jantung, Membantu meningkatkan kesehatan tulang karena kaya akan kalsium."
    }

}

# Fungsi untuk memproses gambar sebelum prediksi
def preprocess_image(img):
    img = img.resize((224, 224))  # Sesuaikan ukuran dengan model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalisasi
    return img

# Streamlit UI
st.title("🔍 Klasifikasi Rempah-Rempah")
st.write("Upload gambar rempah-rempah dan model akan memprediksi jenisnya!")
st.caption("Note : klasifikasi hanya terbatas pada beberapa jenis rempah saja seperti, adas, andaliman, asam jawa, bawang bombai, bawang merah, bawang putih, biji ketumbar, bunga lawang, cengkeh, daun jeruk, daun kemangi, daun ketumbar, daun salam, jahe, jinten, kapulaga, kayu manis, kayu secang, kemiri, kemukus, kencur, kluwek, kunyit, lada, lengkuas, pala, saffron, serai, vanili, wijen.")
with st.sidebar:
    st.header("My Profile")
    st.markdown("""
    **Name:** Fiyanda Ma'muri  
    **Email:** fiyandamamuri@gmail.com  
    **LinkedIn:** [Profil LinkedIn](https://id.linkedin.com/in/fiyandamamuri/)  
    **GitHub:** [Profil GitHub](https://github.com/fiyandamamuri)  
    """)

uploaded_file = st.file_uploader("Unggah gambar rempah", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)

    # Prediksi saat tombol ditekan
    if st.button("🔍 prediksi"):
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = labels[predicted_class]

        st.success(f"🎯 Prediksi : {predicted_label}")

        # Menampilkan informasi tambahan jika tersedia
        if predicted_label in rempah_info:
            info = rempah_info[predicted_label]
            st.subheader("ℹ️ **Informasi Tambahan**")
            st.write(f"**Nama:** {info['Nama']}")
            st.write(f"**Aroma:** {info['Aroma']}")
            st.write(f"**Rasa:** {info['Rasa']}")
            st.write(f"**Kegunaan:** {info['Kegunaan']}")
            st.write(f"**Manfaat Kesehatan:** {info['Manfaat Kesehatan']}")
        else:
            st.warning("⚠️ Informasi tambahan belum tersedia untuk rempah ini.")
st.caption('Copyright © Fiyanda Mamuri - 2025')

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

# ======================================================
# CONFIGURAÇÃO DA PÁGINA
# ======================================================

st.set_page_config(
    page_title="Comparação HE vs CLAHE",
    page_icon="🖼️",
    layout="wide"
)

# ======================================================
# ESTILO VISUAL
# ======================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #00D4FF;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# TÍTULO
# ======================================================

st.title("🖼️ Comparação entre Equalização Tradicional e CLAHE")

st.markdown("""
Esta aplicação realiza:

- Equalização Tradicional de Histograma (HE)
- CLAHE
- Comparação visual
- Histogramas
- Medição de desempenho
- Comparação computacional
""")

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.header("⚙️ Configurações")

clip_limit = st.sidebar.slider(
    "Clip Limit (CLAHE)",
    1.0,
    10.0,
    2.0,
    0.5
)

grid_size = st.sidebar.slider(
    "Tile Grid Size",
    2,
    16,
    8
)

# ======================================================
# UPLOAD
# ======================================================

uploaded_file = st.file_uploader(
    "Faça upload de uma imagem",
    type=["png", "jpg", "jpeg"]
)

# ======================================================
# FUNÇÕES
# ======================================================

def carregar_imagem(uploaded_file):

    image = Image.open(uploaded_file)
    image = np.array(image)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    return image, gray


# ======================================================
# EQUALIZAÇÃO TRADICIONAL
# ======================================================

def aplicar_he(imagem_gray):

    inicio = time.time()

    resultado = cv2.equalizeHist(imagem_gray)

    fim = time.time()

    tempo = (fim - inicio) * 1000

    return resultado, tempo


# ======================================================
# CLAHE
# ======================================================

def aplicar_clahe(imagem_gray, clip_limit, grid_size):

    inicio = time.time()

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )

    resultado = clahe.apply(imagem_gray)

    fim = time.time()

    tempo = (fim - inicio) * 1000

    return resultado, tempo


# ======================================================
# MÉTRICAS
# ======================================================

def calcular_contraste(imagem):
    return np.std(imagem)


# ======================================================
# HISTOGRAMA
# ======================================================

def plot_histograma(imagem, titulo):

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        imagem.ravel(),
        bins=256,
        range=[0, 256]
    )

    ax.set_title(titulo)
    ax.set_xlabel("Intensidade")
    ax.set_ylabel("Quantidade de Pixels")

    return fig


# ======================================================
# PROCESSAMENTO
# ======================================================

if uploaded_file:

    imagem_original, imagem_gray = carregar_imagem(uploaded_file)

    # ==================================================
    # PROCESSAMENTO HE
    # ==================================================

    imagem_he, tempo_he = aplicar_he(imagem_gray)

    # ==================================================
    # PROCESSAMENTO CLAHE
    # ==================================================

    imagem_clahe, tempo_clahe = aplicar_clahe(
        imagem_gray,
        clip_limit,
        grid_size
    )

    # ==================================================
    # CONTRASTE
    # ==================================================

    contraste_original = calcular_contraste(imagem_gray)
    contraste_he = calcular_contraste(imagem_he)
    contraste_clahe = calcular_contraste(imagem_clahe)

    # ==================================================
    # IMAGENS
    # ==================================================

    st.header("📷 Comparação Visual")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(imagem_gray, use_container_width=True)

    with col2:
        st.subheader("Equalização Tradicional")
        st.image(imagem_he, use_container_width=True)

    with col3:
        st.subheader("CLAHE")
        st.image(imagem_clahe, use_container_width=True)

    # ==================================================
    # HISTOGRAMAS
    # ==================================================

    st.header("📊 Histogramas")

    col4, col5, col6 = st.columns(3)

    with col4:
        fig1 = plot_histograma(
            imagem_gray,
            "Histograma Original"
        )
        st.pyplot(fig1)

    with col5:
        fig2 = plot_histograma(
            imagem_he,
            "Histograma HE"
        )
        st.pyplot(fig2)

    with col6:
        fig3 = plot_histograma(
            imagem_clahe,
            "Histograma CLAHE"
        )
        st.pyplot(fig3)

    # ==================================================
    # MÉTRICAS
    # ==================================================

    st.header("📈 Métricas")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Contraste Original",
            f"{contraste_original:.2f}"
        )

    with m2:
        st.metric(
            "Contraste HE",
            f"{contraste_he:.2f}"
        )

    with m3:
        st.metric(
            "Contraste CLAHE",
            f"{contraste_clahe:.2f}"
        )

    # ==================================================
    # TEMPO DE PROCESSAMENTO
    # ==================================================

    st.header("⏱️ Tempo de Processamento")

    t1, t2 = st.columns(2)

    with t1:
        st.metric(
            "Tempo HE",
            f"{tempo_he:.4f} ms"
        )

    with t2:
        st.metric(
            "Tempo CLAHE",
            f"{tempo_clahe:.4f} ms"
        )

    # ==================================================
    # ANÁLISE COMPUTACIONAL
    # ==================================================

    st.header("🧠 Análise Computacional")

    st.markdown(f"""
### Equalização Tradicional

- Técnica global
- Menor custo computacional
- Processamento rápido
- Pode amplificar ruídos

Tempo medido: **{tempo_he:.4f} ms**

---

### CLAHE

- Técnica adaptativa local
- Melhor preservação de detalhes
- Controle de ruído com Clip Limit
- Maior custo computacional

Tempo medido: **{tempo_clahe:.4f} ms**

---

### Complexidade Computacional

| Método | Complexidade |
|--------|-------------|
| HE | Global |
| AHE Ingênua | O(r²) |
| Huang | O(r) |
| Perreault e Hébert | O(1) |

""")

    # ==================================================
    # DOWNLOADS
    # ==================================================

    st.header("⬇️ Download das Imagens")

    col_download1, col_download2 = st.columns(2)

    with col_download1:

        he_bytes = cv2.imencode(
            '.png',
            imagem_he
        )[1].tobytes()

        st.download_button(
            label="Baixar Imagem HE",
            data=he_bytes,
            file_name="imagem_he.png",
            mime="image/png"
        )

    with col_download2:

        clahe_bytes = cv2.imencode(
            '.png',
            imagem_clahe
        )[1].tobytes()

        st.download_button(
            label="Baixar Imagem CLAHE",
            data=clahe_bytes,
            file_name="imagem_clahe.png",
            mime="image/png"
        )

# ======================================================
# RODAPÉ
# ======================================================

st.markdown("---")

st.markdown("""
Desenvolvido para a disciplina de Processamento Digital de Imagens.

Tecnologias utilizadas:

- Python
- OpenCV
- NumPy
- Matplotlib
- Streamlit
""")
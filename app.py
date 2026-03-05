"""
╔══════════════════════════════════════════════════════════╗
║   🎙️ Whisper Executive Transcriptor — Call Center AI    ║
║   Plataforma de Transcrição, Diarização e Análise IA    ║
╚══════════════════════════════════════════════════════════╝
"""
import os
import sys
import json
import time
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG — deve ser o primeiro comando Streamlit
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Whisper Executive Transcriptor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS EXECUTIVO — Paleta Branco / Cinza Claro / Preto / Vermelho
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── ROOT ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #F5F6F8;
    font-family: 'Inter', sans-serif;
    color: #111111;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 2px solid #E5E7EB;
}
[data-testid="stSidebar"] * { color: #111111 !important; }

/* ── PAINEL PRINCIPAL ── */
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1400px;
}

/* ── HEADER HERO ── */
.hero-header {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F0F0 100%);
    border-left: 5px solid #CC0000;
    border-radius: 8px;
    padding: 20px 28px;
    margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.hero-header h1 {
    font-size: 1.9rem;
    font-weight: 800;
    color: #111111;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    font-size: 0.9rem;
    color: #555555;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: #CC0000;
    color: white !important;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 999px;
    margin-right: 8px;
    letter-spacing: 0.5px;
}

/* ── CARDS MÉTRICA ── */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 18px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
div[data-testid="stMetricValue"] > div {
    color: #CC0000 !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}
div[data-testid="stMetricLabel"] > div {
    color: #444444 !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ── BOTÕES ── */
.stButton > button {
    background: #CC0000 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.25s, transform 0.15s, box-shadow 0.25s;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    background: #A80000 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(204,0,0,0.25) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── TABS ── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #666666 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #CC0000 !important;
    border-bottom: 3px solid #CC0000 !important;
}

/* ── EXPANDER / CARDS ── */
details[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ── INPUT ── */
input[data-testid="stTextInput"] { border-radius: 6px !important; }
.stSelectbox > div > div { border-radius: 6px !important; }

/* ── MENSAGENS ── */
.stAlert { border-radius: 8px !important; }

/* ── DIVIDER ── */
hr { border-color: #E5E7EB !important; }

/* ── TRANSCRIÇÃO SPEAKER BOX ── */
.speaker-atendente {
    background: #FFF0F0;
    border-left: 4px solid #CC0000;
    border-radius: 6px;
    padding: 8px 14px;
    margin: 5px 0;
}
.speaker-cliente {
    background: #F5F5F5;
    border-left: 4px solid #333333;
    border-radius: 6px;
    padding: 8px 14px;
    margin: 5px 0;
}
.speaker-label-a { color: #CC0000; font-weight: 700; font-size: 0.82rem; }
.speaker-label-c { color: #333333; font-weight: 700; font-size: 0.82rem; }
.speaker-text { color: #111111; font-size: 0.9rem; line-height: 1.5; }
.speaker-meta { color: #888888; font-size: 0.75rem; margin-top: 2px; }

/* ── INSIGHT CARD ── */
.insight-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.insight-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #888888;
    margin-bottom: 4px;
}
.insight-value {
    font-size: 1.05rem;
    font-weight: 600;
    color: #111111;
}

/* ── BADGE SENTIMENTO ── */
.badge-pos { background:#22C55E; color:white; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:700; }
.badge-neu { background:#6B7280; color:white; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:700; }
.badge-neg { background:#F59E0B; color:white; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:700; }
.badge-crit { background:#CC0000; color:white; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:700; }

/* ── FOOTER ── */
.footer {
    text-align: center;
    color: #AAAAAA;
    font-size: 0.78rem;
    padding: 20px 0 10px 0;
    border-top: 1px solid #E5E7EB;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SEGREDOS / API KEYS
# ═══════════════════════════════════════════════════════════════════════════════
def _get_secret(key: str, default: str = "") -> str:
    """Lê segredo do .streamlit/secrets.toml em qualquer nível."""
    try:
        # Nível raiz
        val = st.secrets.get(key, "")
        if val:
            return val
        # Seção [general]
        val = st.secrets.get("general", {}).get(key, "")
        return val or default
    except Exception:
        return default


OPENAI_KEY   = _get_secret("OPENAI_API_KEY")
GROQ_KEY     = _get_secret("GROQ_API_KEY")
HF_TOKEN     = _get_secret("HF_TOKEN")
ORACLE_USER  = _get_secret("ORACLE_USER")
ORACLE_PASS  = _get_secret("ORACLE_PASSWORD")
ORACLE_DSN   = _get_secret("ORACLE_DSN")

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for _k, _v in {
    "processed_data": [],
    "processing_errors": [],
    "transcriber": None,
    "current_model": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 18px 0;'>
      <span style='font-size:2.2rem;'>🎙️</span><br>
      <span style='font-weight:800;font-size:1.05rem;color:#111;'>Whisper Executive</span><br>
      <span style='font-size:0.78rem;color:#CC0000;font-weight:600;'>TRANSCRIPTOR PRO</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Motor de Transcrição")
    
    # Toggle entre Whisper e WhisperX
    transcription_engine = st.radio(
        "Engine",
        ["Whisper (Clássico)", "WhisperX (Diarização Pro)"],
        index=0,
        help="WhisperX usa Pyannote para diarização neural profissional. Requer HF_TOKEN.",
        horizontal=True,
    )
    use_whisperx = transcription_engine == "WhisperX (Diarização Pro)"
    
    # Modelo Whisper
    whisper_model = st.selectbox(
        "Modelo",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help="Modelos maiores = maior precisão, mas mais lentos e mais RAM/VRAM.",
    )
    
    # WhisperX: Configurações adicionais
    if use_whisperx:
        st.markdown("##### 🔑 HuggingFace Token")
        hf_token_input = st.text_input(
            "HF Token",
            type="password",
            value=HF_TOKEN,
            placeholder="hf_...",
            help="Obrigatório para Pyannote. Gere em: https://huggingface.co/settings/tokens",
        )
        effective_hf_token = hf_token_input or HF_TOKEN
        
        if not effective_hf_token:
            st.warning("⚠️ HF_TOKEN não configurado. Diarização usará fallback simples.")
        else:
            st.success("✅ HF_TOKEN configurado")
        
        # Opção de forçar CPU
        force_cpu = st.checkbox(
            "Forçar CPU (desabilita GPU)",
            value=False,
            help="Use para debugging ou se tiver problemas com CUDA.",
        )
        
        # Controle de speakers
        col_sp1, col_sp2 = st.columns(2)
        with col_sp1:
            min_speakers = st.number_input("Min Speakers", min_value=1, max_value=10, value=2)
        with col_sp2:
            max_speakers = st.number_input("Max Speakers", min_value=1, max_value=10, value=2)
    else:
        effective_hf_token = None
        force_cpu = False
        min_speakers = 2
        max_speakers = 2

    st.markdown("#### 🤖 Provedor de IA")
    ai_provider = st.radio("Provider", ["OpenAI", "Groq"], horizontal=True)

    if ai_provider == "OpenAI":
        api_key_input = st.text_input("OpenAI API Key", type="password",
                                       value=OPENAI_KEY, placeholder="sk-...")
        sel_model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
        effective_openai = api_key_input or OPENAI_KEY
        effective_groq   = GROQ_KEY
    else:
        api_key_input = st.text_input("Groq API Key", type="password",
                                       value=GROQ_KEY, placeholder="gsk_...")
        sel_model = st.selectbox("Modelo", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                                             "mixtral-8x7b-32768"], index=0)
        effective_groq   = api_key_input or GROQ_KEY
        effective_openai = OPENAI_KEY

    st.markdown("#### 🔒 Processamento PII")
    redact_toggle   = st.toggle("Redação PII (LGPD)", value=True,
                                  help="Remove CPF, telefone, e-mail, etc. das transcrições.")
    llm_redact      = st.toggle("Redação via IA (contextual)", value=False,
                                  help="Usa LLM para redação mais inteligente. Requer API key.")

    st.markdown("#### 🔍 Análise IA")
    run_insights    = st.toggle("Extrair Insights (RAG)", value=False,
                                  help="Analisa a transcrição e extrai motivo, sentimento, etc.")

    st.divider()

    # ── Oracle DB (para aba Consulta Oracle) ──
    with st.expander("🗄️ Conexão Oracle (opcional)"):
        oracle_user_input = st.text_input(
            "Usuário", value=ORACLE_USER, placeholder="oracle_user",
            key="oracle_user",
        )
        oracle_pass_input = st.text_input(
            "Senha", type="password", value=ORACLE_PASS, placeholder="••••••",
            key="oracle_pass",
        )
        oracle_dsn_input = st.text_input(
            "DSN", value=ORACLE_DSN, placeholder="host:1521/ORCL",
            help="Formato: host:porta/service_name  ou  TNS alias",
            key="oracle_dsn",
        )

    st.divider()
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_ok else None
    except Exception:
        gpu_ok = False
        gpu_name = None

    if gpu_ok:
        st.success(f"🚀 GPU: {gpu_name or 'Disponível'}")
    else:
        st.warning("⚠️ CPU mode (sem GPU)")

    if st.session_state.processed_data:
        st.info(f"📊 {len(st.session_state.processed_data)} áudio(s) processado(s)")
        if st.button("🗑️ Limpar Cache", use_container_width=True):
            st.session_state.processed_data = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  INICIALIZAR TRANSCRIBER (cacheado por modelo + engine)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 Carregando modelo Whisper...")
def get_transcriber_classic(model_name: str, openai_key: str, groq_key: str):
    """Carrega motor Whisper clássico."""
    from src.transcriber import CallCenterTranscriber
    return CallCenterTranscriber(
        model_name=model_name,
        openai_key=openai_key or None,
        groq_key=groq_key or None,
    )

@st.cache_resource(show_spinner="🔄 Carregando WhisperX + Pyannote...")
def get_transcriber_whisperx(model_name: str, openai_key: str, groq_key: str, hf_token: str, force_cpu: bool):
    """Carrega motor WhisperX com Pyannote."""
    from src.transcriber_whisperx import CallCenterTranscriberX
    return CallCenterTranscriberX(
        model_name=model_name,
        openai_key=openai_key or None,
        groq_key=groq_key or None,
        hf_token=hf_token or None,
        force_cpu=force_cpu,
    )

# Carrega o transcriber adequado baseado na engine escolhida
if use_whisperx:
    transcriber = get_transcriber_whisperx(
        whisper_model,
        effective_openai,
        effective_groq,
        effective_hf_token,
        force_cpu,
    )
    engine_label = f"WhisperX ({whisper_model})"
else:
    transcriber = get_transcriber_classic(
        whisper_model,
        effective_openai,
        effective_groq,
    )
    engine_label = f"Whisper ({whisper_model})"

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
  <h1>🎙️ Whisper Executive Transcriptor</h1>
  <p>
    <span class="hero-badge">CALL CENTER AI</span>
    <span class="hero-badge">DIARIZAÇÃO</span>
    <span class="hero-badge">PII LGPD</span>
    <span class="hero-badge">RAG INSIGHTS</span>
    Plataforma profissional de transcrição, análise e insights para operações de call center.
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS PRINCIPAIS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🚀 Upload & Transcrição",
    "📂 Processamento em Lote",
    "📊 Dashboard & Métricas",
    "🧠 Análise IA & Insights",
    "🗄️ Consulta Oracle",
])


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def render_quality_bar(score: float, label: str = ""):
    """Renderiza barra de qualidade colorida."""
    color = "#22C55E" if score >= 70 else "#F59E0B" if score >= 40 else "#CC0000"
    st.markdown(f"""
    <div style="margin:4px 0 10px 0;">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
        <span style="font-size:0.78rem;color:#555;font-weight:600;">{label}</span>
        <span style="font-size:0.78rem;color:{color};font-weight:700;">{score:.1f}%</span>
      </div>
      <div style="background:#E5E7EB;border-radius:4px;height:8px;overflow:hidden;">
        <div style="background:{color};height:100%;width:{min(100,score):.1f}%;border-radius:4px;
                    transition:width 0.6s ease;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_segments(segments: list):
    """Renderiza segmentos de transcrição diarizados."""
    for seg in segments:
        speaker = seg.get("speaker", "?")
        text    = seg.get("text", "")
        start   = seg.get("start", 0)
        conf    = seg.get("confidence", 0)
        reason  = seg.get("reason", "")
        canal   = seg.get("channel") or "auto"

        mins, secs = int(start // 60), int(start % 60)
        ts = f"{mins:02d}:{secs:02d}"

        if speaker == "Atendente":
            st.markdown(f"""
            <div class="speaker-atendente">
              <div class="speaker-label-a">🎧 {speaker} <span style="font-weight:400;color:#888;">[{ts}]</span></div>
              <div class="speaker-text">{text}</div>
              <div class="speaker-meta">Conf: {conf*100:.0f}% · Método: {reason} · Canal: {canal}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="speaker-cliente">
              <div class="speaker-label-c">👤 {speaker} <span style="font-weight:400;color:#888;">[{ts}]</span></div>
              <div class="speaker-text">{text}</div>
              <div class="speaker-meta">Conf: {conf*100:.0f}% · Método: {reason} · Canal: {canal}</div>
            </div>""", unsafe_allow_html=True)


def process_uploaded_file(uploaded_file, idx: int = 0, total: int = 1):
    """Processa um arquivo uploaded, salva temp, transcreve e retorna resultado."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # Parâmetros base
        params = {
            "redact": redact_toggle,
            "llm_redact": llm_redact,
            "redaction_provider": ai_provider.lower(),
            "redaction_model": sel_model,
            "run_insights": run_insights,
            "insights_provider": ai_provider.lower(),
            "insights_model": sel_model,
        }
        
        # Adiciona parâmetros específicos do WhisperX
        if use_whisperx:
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers
        
        result = transcriber.process_audio(tmp_path, **params)
        result["filename"] = uploaded_file.name
        return result
    finally:
        os.unlink(tmp_path)


def sentiment_badge(s: str) -> str:
    s_low = s.lower()
    if "positivo" in s_low: return f'<span class="badge-pos">{s}</span>'
    if "negativo" in s_low: return f'<span class="badge-neg">{s}</span>'
    if "crítico" in s_low or "critico" in s_low: return f'<span class="badge-crit">{s}</span>'
    return f'<span class="badge-neu">{s}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — UPLOAD & TRANSCRIÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 📁 Upload de Áudio — Transcrição Individual ou Múltipla")
    st.caption("Suporte a **.wav** e **.mp3** | Transcreve em Português do Brasil | Diarização automática")

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        uploaded_files = st.file_uploader(
            "Arraste ou selecione seus arquivos de áudio",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            help="Pode selecionar múltiplos arquivos para processamento sequencial.",
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} arquivo(s) selecionado(s):**")
            for f in uploaded_files:
                size_kb = len(f.getbuffer()) / 1024
                st.markdown(f"- `{f.name}` ({size_kb:.1f} KB)")

        btn_process = st.button(
            "▶ Iniciar Transcrição",
            use_container_width=True,
            disabled=not bool(uploaded_files),
        )

        if btn_process and uploaded_files:
            # Limpa erros anteriores
            st.session_state.processing_errors = []
            progress = st.progress(0, text="Iniciando transcrição...")
            status_ph = st.empty()
            n_ok = 0

            for i, uf in enumerate(uploaded_files):
                status_ph.info(f"⏳ Transcrevendo `{uf.name}` ({i+1}/{len(uploaded_files)})...")
                try:
                    result = process_uploaded_file(uf, i, len(uploaded_files))
                    st.session_state.processed_data.insert(0, result)
                    n_ok += 1
                except Exception as e:
                    import traceback
                    err_msg = f"{uf.name}: {e}\n\nDetalhes:\n{traceback.format_exc()}"
                    st.session_state.processing_errors.append(err_msg)

                progress.progress((i + 1) / len(uploaded_files),
                                   text=f"Processado: {uf.name}")

            if n_ok > 0:
                status_ph.success(f"✅ {n_ok}/{len(uploaded_files)} arquivo(s) processado(s)!")
            elif st.session_state.processing_errors:
                status_ph.error("❌ Falha ao processar. Veja detalhes abaixo.")
            time.sleep(1.5)
            st.rerun()

    with col_res:
        st.markdown("#### 📋 Resultados Recentes")

        # Exibe erros persistidos do processamento anterior
        if st.session_state.processing_errors:
            for err in st.session_state.processing_errors:
                with st.expander("❌ Erro de processamento — clique para detalhes", expanded=True):
                    st.error(err)

        if not st.session_state.processed_data and not st.session_state.processing_errors:
            st.info("Nenhum áudio transcrito ainda. Faça upload e clique em **Iniciar Transcrição**.")
        else:
            for item in st.session_state.processed_data[:5]:
                m = item.get("metrics", {})
                with st.expander(f"📄 {item['filename']} · {item['timestamp'][:16].replace('T',' ')}", expanded=False):
                    # Métricas de qualidade
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Qualidade", f"{m.get('quality_score', 0):.0f}%")
                    c2.metric("Diarização", f"{item.get('diarization_score', 0):.0f}%")
                    c3.metric("Canais", "Estéreo" if m.get("is_stereo") else "Mono")
                    c4.metric("Duração", f"{m.get('duration', 0):.1f}s")

                    render_quality_bar(float(m.get("quality_score", 0)), "Qualidade do Áudio")
                    render_quality_bar(float(item.get("diarization_score", 0)), "Precisão da Diarização")

                    col_i, col_ii = st.columns(2)
                    col_i.markdown(f"**SNR:** `{m.get('snr_db', 0):.1f} dB`")
                    col_ii.markdown(f"**Ruído:** `{m.get('noise_level', 'N/A')}`")
                    st.markdown(f"**Canal Estéreo:** `{m.get('stereo_balance', 'N/A')}`")
                    st.markdown(f"**Região:** `{item.get('region', 'N/A')}` · **Empresa:** `{item.get('company', 'N/A')}`")

                    st.divider()
                    st.markdown("**Transcrição Diarizada:**")
                    render_segments(item.get("segments", []))

                    # Download buttons
                    d_col1, d_col2 = st.columns(2)
                    txt_content = "\n".join(
                        f"[{s['speaker']}] {s['text']}" for s in item.get("segments", [])
                    )
                    # Chave única baseada no filename e timestamp
                    file_key = f"{item['filename']}_{item.get('timestamp', '')}"
                    d_col1.download_button(
                        "⬇ Baixar .txt",
                        data=txt_content,
                        file_name=f"{Path(item['filename']).stem}_transcricao.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key=f"download_txt_{file_key}",
                    )
                    d_col2.download_button(
                        "⬇ Baixar .json",
                        data=json.dumps(item, ensure_ascii=False, indent=2),
                        file_name=f"{Path(item['filename']).stem}_completo.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"download_json_{file_key}",
                    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PROCESSAMENTO EM LOTE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 📂 Processamento em Lote — Pasta Local")
    st.caption("Processa todos os arquivos `.wav` / `.mp3` de uma pasta no servidor.")

    default_samples = str(Path(__file__).parent / "samples")
    batch_dir = st.text_input("📁 Caminho da Pasta", value=default_samples)

    if batch_dir and Path(batch_dir).exists():
        audio_files = []
        for ext in ("*.wav", "*.mp3"):
            audio_files.extend(sorted(Path(batch_dir).glob(ext)))

        st.markdown(f"**Arquivos encontrados:** `{len(audio_files)}`")

        if audio_files:
            with st.expander("📋 Ver lista de arquivos"):
                for f in audio_files:
                    size_kb = f.stat().st_size / 1024
                    st.markdown(f"- `{f.name}` ({size_kb:.1f} KB)")

            if st.button("🎙️ Processar Pasta em Lote", use_container_width=True):
                st.session_state.processing_errors = []
                prog = st.progress(0, text="Iniciando...")
                stat = st.empty()
                n_ok = 0

                for i, fp in enumerate(audio_files):
                    stat.info(f"⏳ Processando `{fp.name}` ({i+1}/{len(audio_files)})...")
                    try:
                        # Parâmetros base
                        params = {
                            "redact": redact_toggle,
                            "llm_redact": llm_redact,
                            "redaction_provider": ai_provider.lower(),
                            "redaction_model": sel_model,
                            "run_insights": run_insights,
                            "insights_provider": ai_provider.lower(),
                            "insights_model": sel_model,
                        }
                        
                        # Adiciona parâmetros específicos do WhisperX
                        if use_whisperx:
                            params["min_speakers"] = min_speakers
                            params["max_speakers"] = max_speakers
                        
                        result = transcriber.process_audio(str(fp), **params)
                        st.session_state.processed_data.insert(0, result)
                        n_ok += 1
                    except Exception as e:
                        import traceback
                        err_msg = f"{fp.name}: {e}\n\nDetalhes:\n{traceback.format_exc()}"
                        st.session_state.processing_errors.append(err_msg)
                    prog.progress((i + 1) / len(audio_files), text=f"{fp.name}")

                if n_ok > 0:
                    stat.success(f"✅ Lote concluído! {n_ok}/{len(audio_files)} processados.")
                else:
                    stat.error("❌ Nenhum arquivo processado. Veja erros na Tab 1.")

                # Mostra erros imediatamente antes de rerun
                for err in st.session_state.processing_errors:
                    with st.expander("❌ Erro — clique para detalhes", expanded=True):
                        st.error(err)

                time.sleep(2)
                st.rerun()
        else:
            st.info("Nenhum arquivo `.wav` ou `.mp3` encontrado nessa pasta.")
            st.markdown("""
            💡 **Dica:** Adicione arquivos `.wav` ou `.mp3` à pasta `samples/` para processamento em lote.
            """)
    elif batch_dir:
        st.warning(f"⚠️ Pasta não encontrada: `{batch_dir}`")



# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DASHBOARD & MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 📊 Dashboard Executivo — Métricas de Performance")

    if not st.session_state.processed_data:
        st.info("📭 Nenhum áudio processado ainda. Processe áudios nas abas anteriores.")
    else:
        data = st.session_state.processed_data

        # ── KPI Cards ──
        total = len(data)
        avg_quality = sum(d.get("metrics", {}).get("quality_score", 0) for d in data) / total
        avg_diar    = sum(d.get("diarization_score", 0) for d in data) / total
        total_dur   = sum(d.get("metrics", {}).get("duration", 0) for d in data)
        stereo_pct  = sum(1 for d in data if d.get("is_stereo", False)) / total * 100
        avg_snr     = sum(d.get("metrics", {}).get("snr_db", 0) for d in data) / total

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("🎙️ Áudios", total)
        k2.metric("⭐ Qualidade Média", f"{avg_quality:.1f}%")
        k3.metric("🔀 Diarização", f"{avg_diar:.1f}%")
        k4.metric("⏱️ Duração Total", f"{total_dur/60:.1f} min")
        k5.metric("📡 SNR Médio", f"{avg_snr:.1f} dB")
        k6.metric("🎚️ Estéreo", f"{stereo_pct:.0f}%")

        st.divider()

        # ── Tabela consolidada ──
        rows = []
        for d in data:
            m = d.get("metrics", {})
            rows.append({
                "Arquivo":         d.get("filename", "N/A"),
                "Data/Hora":       d.get("timestamp", "")[:16].replace("T", " "),
                "Qualidade (%)":   round(m.get("quality_score", 0), 1),
                "SNR (dB)":        round(m.get("snr_db", 0), 1),
                "Ruído":           m.get("noise_level", "N/A"),
                "Diarização (%)":  round(d.get("diarization_score", 0), 1),
                "Duração (s)":     round(m.get("duration", 0), 1),
                "Canais":          "Estéreo" if m.get("is_stereo") else "Mono",
                "Balanço L/R":     m.get("stereo_balance", "N/A"),
                "Região":          d.get("region", "N/A"),
                "Empresa":         d.get("company", "N/A"),
                "Segs. Atendente": d.get("atendente_segments", 0),
                "Segs. Cliente":   d.get("cliente_segments", 0),
            })

        df = pd.DataFrame(rows)
        st.markdown("#### 📋 Tabela de Resultados")
        st.dataframe(df, use_container_width=True, height=300)

        # Download da tabela
        st.download_button(
            "⬇ Exportar CSV",
            data=df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"relatorio_transcricoes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        st.divider()

        # ── Gráficos ──
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### 📈 Qualidade do Áudio por Arquivo")
            chart_df = df.set_index("Arquivo")[["Qualidade (%)", "Diarização (%)"]].head(15)
            st.bar_chart(chart_df, color=["#CC0000", "#888888"])

        with col_g2:
            st.markdown("#### 📊 SNR (dB) por Arquivo")
            snr_df = df.set_index("Arquivo")[["SNR (dB)"]].head(15)
            st.line_chart(snr_df, color=["#CC0000"])

        # Distribuição de regiões e empresas
        col_r, col_e = st.columns(2)
        with col_r:
            st.markdown("#### 🗺️ Distribuição por Região")
            reg_counts = df["Região"].value_counts().reset_index()
            reg_counts.columns = ["Região", "Qtd"]
            st.dataframe(reg_counts, use_container_width=True)

        with col_e:
            st.markdown("#### 🏢 Distribuição por Empresa")
            emp_counts = df["Empresa"].value_counts().reset_index()
            emp_counts.columns = ["Empresa", "Qtd"]
            st.dataframe(emp_counts, use_container_width=True)

        # Métricas avançadas de diarização
        st.divider()
        st.markdown("#### 🎚️ Análise de Canais (Estéreo L/R)")
        stereo_data = [d for d in data if d.get("is_stereo")]
        if stereo_data:
            for d in stereo_data[:5]:
                m = d.get("metrics", {})
                channel_db = m.get('channel_db') or [0, 0]
                ch_l = channel_db[0] if len(channel_db) > 0 else 0
                ch_r = channel_db[-1] if len(channel_db) > 1 else 0
                st.markdown(f"""
                **`{d['filename']}`** — {m.get('stereo_balance', 'N/A')}
                Energia L: `{ch_l:.1f} dB` |
                Energia R: `{ch_r:.1f} dB`
                """)
        else:
            st.info("Nenhum arquivo estéreo processado ainda.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ANÁLISE IA & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🧠 Análise IA & Insights — RAG sobre Transcrições")
    st.caption("Extração automática de insights usando LLM como agente RAG sobre as transcrições processadas.")

    if not st.session_state.processed_data:
        st.info("Processe áudios primeiro para habilitar a análise de insights.")
    else:
        # ── Seletor de transcrição ──
        file_options = [d["filename"] for d in st.session_state.processed_data]
        selected_file = st.selectbox("Selecione uma transcrição para analisar", file_options)

        selected_item = next(
            (d for d in st.session_state.processed_data if d["filename"] == selected_file),
            None,
        )

        if selected_item:
            full_text = selected_item.get("full_text", "")
            existing_insights = selected_item.get("insights", {})

            col_txt, col_ins = st.columns([1, 1], gap="large")

            with col_txt:
                st.markdown("#### 📝 Transcrição Completa")
                st.text_area("Texto", value=full_text, height=280, disabled=True)
                st.markdown(f"**Tamanho:** `{len(full_text)} caracteres` · `{len(full_text.split())} palavras`")

                # WER (se o usuário fornecer referência)
                st.divider()
                st.markdown("#### 📐 Word Error Rate (WER)")
                st.caption("Cole uma transcrição de referência para calcular o WER (mede qualidade da transcrição).")
                ref_text = st.text_area("Transcrição de referência (opcional)", height=80, placeholder="Cole o texto de referência aqui...")
                if ref_text.strip() and st.button("Calcular WER"):
                    from src.audio_utils import AudioAnalyzer
                    wer_val = AudioAnalyzer.calculate_wer(ref_text.strip(), full_text.strip())
                    wer_pct = wer_val * 100
                    color = "#22C55E" if wer_pct < 10 else "#F59E0B" if wer_pct < 30 else "#CC0000"
                    st.markdown(f"""
                    <div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:8px;padding:16px;text-align:center;">
                      <div style="font-size:2rem;font-weight:800;color:{color};">{wer_pct:.1f}%</div>
                      <div style="color:#555;font-size:0.85rem;margin-top:4px;">Word Error Rate</div>
                      <div style="color:#888;font-size:0.78rem;">
                        {"🟢 Excelente" if wer_pct < 10 else "🟡 Regular" if wer_pct < 30 else "🔴 Alto erro"} — 
                        {"STT de alta qualidade" if wer_pct < 10 else "Considere modelo maior" if wer_pct < 30 else "Use large-v3"}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_ins:
                st.markdown("#### 🔍 Insights Extraídos por IA")

                has_insights = bool(existing_insights and existing_insights.get("motivo_ligacao"))

                if not has_insights:
                    st.info("Configure uma API key no Sidebar e ative **Extrair Insights (RAG)**, ou clique abaixo.")
                    if st.button("🤖 Analisar com IA agora", use_container_width=True):
                        if not (effective_openai or effective_groq):
                            st.error("Configure uma API key (OpenAI ou Groq) no sidebar.")
                        else:
                            with st.spinner("Analisando transcrição..."):
                                from src.redactor import PIIRedactor
                                analyzer = PIIRedactor(
                                    openai_api_key=effective_openai or None,
                                    groq_api_key=effective_groq or None,
                                )
                                insights = analyzer.analyze_with_llm(
                                    full_text,
                                    provider=ai_provider.lower(),
                                    model=sel_model,
                                )
                                selected_item["insights"] = insights
                                existing_insights = insights
                                has_insights = True
                                st.rerun()

                if has_insights:
                    ins = existing_insights

                    st.markdown(f"""
                    <div class="insight-card">
                      <div class="insight-title">📞 Motivo da Ligação</div>
                      <div class="insight-value">{ins.get('motivo_ligacao','N/A')}</div>
                    </div>
                    <div class="insight-card">
                      <div class="insight-title">😊 Sentimento do Cliente</div>
                      <div class="insight-value">{sentiment_badge(ins.get('sentimento_cliente','Neutro'))}</div>
                    </div>
                    <div class="insight-card">
                      <div class="insight-title">🚨 Criticidade</div>
                      <div class="insight-value">{ins.get('criticidade','N/A')}</div>
                    </div>
                    <div class="insight-card">
                      <div class="insight-title">📦 Produto / Serviço</div>
                      <div class="insight-value">{ins.get('produto_servico','N/A')}</div>
                    </div>
                    <div class="insight-card">
                      <div class="insight-title">💡 Ação Recomendada</div>
                      <div class="insight-value">{ins.get('acao_recomendada','N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**📝 Resumo:**")
                    st.markdown(f"> {ins.get('resumo','N/A')}")

                    if ins.get("keywords"):
                        st.markdown("**🏷️ Keywords:**")
                        tags = " ".join(f'`{k}`' for k in ins.get("keywords", []))
                        st.markdown(tags)

        # ── TOP Motivos (RAG coletivo) ──
        st.divider()
        st.markdown("#### 📊 TOP Motivos das Ligações — Análise Coletiva")
        st.caption("Consolida os motivos extraídos por IA de todas as transcrições processadas.")

        all_motivos = [
            d.get("insights", {}).get("motivo_ligacao", "")
            for d in st.session_state.processed_data
            if d.get("insights", {}).get("motivo_ligacao", "") not in ("", "Não identificado", "API não configurada")
        ]

        if all_motivos:
            motivo_counts = Counter(all_motivos).most_common(10)
            df_motivos = pd.DataFrame(motivo_counts, columns=["Motivo", "Qtd"])
            df_motivos["% do Total"] = (df_motivos["Qtd"] / len(all_motivos) * 100).round(1)

            col_m1, col_m2 = st.columns([2, 1])
            with col_m1:
                st.bar_chart(df_motivos.set_index("Motivo")["Qtd"], color="#CC0000")
            with col_m2:
                st.dataframe(df_motivos, use_container_width=True)

            # Sentimentos
            all_sents = [
                d.get("insights", {}).get("sentimento_cliente", "")
                for d in st.session_state.processed_data
                if d.get("insights", {}).get("sentimento_cliente", "")
            ]
            if all_sents:
                st.markdown("**Distribuição de Sentimentos:**")
                sent_df = pd.DataFrame(Counter(all_sents).most_common(), columns=["Sentimento", "Qtd"])
                st.dataframe(sent_df, use_container_width=True)
        else:
            st.info("Ative **Extrair Insights** e processe áudios para ver os TOP motivos aqui.")

        # ── Prompt personalizado ──
        st.divider()
        st.markdown("#### 💬 Prompt Personalizado — RAG sobre as Transcrições")
        st.caption("Faça perguntas sobre o conjunto de transcrições processadas (agente RAG simples).")

        user_prompt = st.text_area(
            "Sua pergunta / prompt",
            placeholder="Ex: Qual o principal problema técnico relatado pelos clientes?",
            height=80,
        )

        if st.button("🤖 Executar Prompt", use_container_width=True, disabled=not user_prompt.strip()):
            if not (effective_openai or effective_groq):
                st.error("Configure uma API key no Sidebar.")
            else:
                # Constrói contexto RAG com todas as transcrições
                context_parts = []
                for i, d in enumerate(st.session_state.processed_data[:10], 1):
                    context_parts.append(
                        f"[Ligação {i} — {d['filename']}]\n{d.get('full_text','')[:800]}"
                    )
                rag_context = "\n\n---\n\n".join(context_parts)

                rag_prompt = (
                    f"Você é um analista de call center especializado em Português do Brasil.\n"
                    f"A seguir estão {len(context_parts)} transcrições de ligações reais:\n\n"
                    f"{rag_context}\n\n"
                    f"---\n\nPergunta do usuário:\n{user_prompt}\n\n"
                    f"Responda de forma objetiva e profissional, baseado APENAS nas transcrições fornecidas."
                )

                with st.spinner("Consultando IA..."):
                    try:
                        if ai_provider == "OpenAI" and effective_openai:
                            from openai import OpenAI
                            client = OpenAI(api_key=effective_openai)
                            resp = client.chat.completions.create(
                                model=sel_model,
                                messages=[{"role": "user", "content": rag_prompt}],
                                temperature=0.3,
                                max_tokens=1024,
                            )
                            answer = resp.choices[0].message.content.strip()
                        else:
                            from groq import Groq
                            client = Groq(api_key=effective_groq)
                            resp = client.chat.completions.create(
                                model=sel_model,
                                messages=[{"role": "user", "content": rag_prompt}],
                                temperature=0.3,
                                max_tokens=1024,
                            )
                            answer = resp.choices[0].message.content.strip()

                        st.markdown("**💬 Resposta:**")
                        st.markdown(f"""
                        <div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:8px;
                                    padding:18px 22px;line-height:1.7;color:#111;">
                          {answer}
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Erro na consulta: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — CONSULTA ORACLE (Amostragem Estatística)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🗄️ Consulta Oracle — Amostragem Estatística Mensal")
    st.caption(
        "Gera (e executa, se conectado) uma query Oracle que retorna **N registros "
        "únicos por CAMPO1**, selecionados aleatoriamente, para um mês específico."
    )

    from src.db_oracle import build_sample_query, connect_oracle, run_sample_query

    col_q1, col_q2 = st.columns([1, 1], gap="large")

    with col_q1:
        st.markdown("#### ⚙️ Parâmetros da Consulta")

        # Mês de referência
        default_month = datetime.now().strftime("%Y-%m")
        ano_mes_input = st.text_input(
            "Mês de Referência (AAAA-MM)",
            value=default_month,
            placeholder="2024-03",
            help="Período a ser amostrado. Formato: Ano-Mês (ex.: 2024-03).",
        )

        # Tabela
        tabela_input = st.text_input(
            "Nome da Tabela",
            value="TABELA",
            help="Nome da tabela Oracle que contém os registros de transcrição.",
        )

        # CAMPO1
        campos_input = st.text_area(
            "Valores de CAMPO1 (um por linha)",
            value="123\n1234",
            height=100,
            help="Filtra apenas esses valores de CAMPO1. Coloque um valor por linha.",
        )
        campos_list = [c.strip() for c in campos_input.splitlines() if c.strip()]

        # Tamanho da amostra
        n_amostras = st.number_input(
            "Tamanho da Amostra (N)",
            min_value=1,
            max_value=100_000,
            value=6000,
            step=1000,
            help=(
                "Quantidade máxima de registros únicos por CAMPO1 a retornar. "
                "Para uma amostra estatisticamente válida de uma população "
                "grande, 6000 registros oferecem margem de erro < 1,3% com "
                "nível de confiança de 95%."
            ),
        )

    with col_q2:
        st.markdown("#### 📐 SQL Gerado")

        try:
            sql_preview = build_sample_query(
                ano_mes=ano_mes_input,
                campos=campos_list,
                tabela=tabela_input,
                n_amostras=n_amostras,
            )
            st.code(sql_preview, language="sql")

            # Botão de cópia (via download)
            st.download_button(
                "⬇️ Baixar SQL (.sql)",
                data=sql_preview.encode("utf-8"),
                file_name=f"amostra_{tabela_input}_{ano_mes_input}.sql",
                mime="text/plain",
                use_container_width=True,
            )
        except ValueError as ve:
            st.error(f"❌ Parâmetro inválido: {ve}")
            sql_preview = None

    # ── Estatísticas da Amostra ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Dimensionamento da Amostra")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    import math

    # Margem de erro com nível de confiança 95% (z=1.96), população infinita
    if n_amostras > 0:
        margem_erro = (1.96 / math.sqrt(n_amostras)) * 100
    else:
        margem_erro = 0.0

    col_s1.metric("Registros Solicitados", f"{n_amostras:,}")
    col_s2.metric("Nível de Confiança", "95%")
    col_s3.metric("Margem de Erro (máx.)", f"±{margem_erro:.2f}%")
    col_s4.metric("Período", ano_mes_input if ano_mes_input else "—")

    st.caption(
        "📌 **Nota metodológica:** A margem de erro foi calculada assumindo proporção "
        "p=0,5 (caso mais conservador) e distribuição normal com z=1,96 (95% confiança). "
        "Com N=6000 a margem é ≈±1,27%, adequada para análises de call center em produção."
    )

    # ── Execução no Banco ────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔌 Executar no Banco Oracle")

    eff_oracle_user = oracle_user_input or ORACLE_USER
    eff_oracle_pass = oracle_pass_input or ORACLE_PASS
    eff_oracle_dsn  = oracle_dsn_input  or ORACLE_DSN

    db_ready = bool(eff_oracle_user and eff_oracle_pass and eff_oracle_dsn)
    if not db_ready:
        st.info(
            "Configure **Usuário**, **Senha** e **DSN** Oracle no painel lateral "
            "(expanda a seção 🗄️ Conexão Oracle) para habilitar a execução."
        )

    run_btn = st.button(
        "▶️ Executar Consulta no Oracle",
        use_container_width=True,
        disabled=(not db_ready or sql_preview is None),
    )

    if run_btn and sql_preview is not None:
        with st.spinner("Conectando ao Oracle e executando query…"):
            try:
                conn = connect_oracle(
                    user=eff_oracle_user,
                    password=eff_oracle_pass,
                    dsn=eff_oracle_dsn,
                )
                df_result = run_sample_query(
                    conn=conn,
                    ano_mes=ano_mes_input,
                    campos=campos_list,
                    tabela=tabela_input,
                    n_amostras=n_amostras,
                )
                conn.close()

                st.success(f"✅ Consulta executada — {len(df_result):,} registros retornados.")
                st.dataframe(df_result, use_container_width=True)

                # Download CSV
                csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Baixar resultado (.csv)",
                    data=csv_bytes,
                    file_name=f"amostra_{tabela_input}_{ano_mes_input}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"❌ Erro ao executar no Oracle: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
  🎙️ <strong>Whisper Executive Transcriptor v2.0</strong> &nbsp;|&nbsp;
  Engine: <strong>{engine_label}</strong> &nbsp;|&nbsp;
  {datetime.now().strftime("%d/%m/%Y %H:%M")} &nbsp;|&nbsp;
  Portfólio Cloud Streamlit &nbsp;·&nbsp; PII LGPD Compliant
</div>
""", unsafe_allow_html=True)

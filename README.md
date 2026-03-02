# 🎙️ Whisper Executive Transcriptor

**Plataforma profissional de Transcrição, Diarização e Análise IA para Call Center**  
Portfólio Cloud Streamlit | PII LGPD Compliant | Português do Brasil 🇧🇷

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-orange)](https://github.com/openai/whisper)
[![WhisperX](https://img.shields.io/badge/WhisperX-Pyannote-purple)](https://github.com/m-bain/whisperX)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Visão Geral

O **Whisper Executive Transcriptor** é uma plataforma completa para transcrição e análise de áudios de call center. Combina o poder do **OpenAI Whisper** ou **WhisperX** para transcrição em Português do Brasil com:

- 🔀 **Diarização automática** (identificação de Atendente vs. Cliente)
  - **Whisper Clássico:** Heurística baseada em regex + canais
  - **WhisperX Pro:** Diarização neural com Pyannote.audio 3.x
- 📊 **Métricas avançadas de áudio** (SNR, qualidade, ruído, canais L/R)
- 🔒 **Redação PII LGPD** (CPF, CNPJ, e-mail, telefone via regex ou IA)
- 🧠 **Análise RAG com LLM** (motivo da ligação, sentimento, insights)

---

## ✨ Novidade: WhisperX com Pyannote

Agora você pode escolher entre **duas engines de transcrição**:

### 🎯 **Whisper Clássico**
- Diarização heurística (regex + canais estéreo)
- Mais rápido
- Não requer configuração adicional

### 🚀 **WhisperX Pro** (NOVO!)
- **Diarização neural profissional** via Pyannote.audio
- Precisão **85-95%** vs 60-75% do clássico
- Alinhamento temporal ultra-preciso (word-level)
- Funciona melhor com áudio mono
- **Requer:** HuggingFace Token (gratuito)

📖 **[Veja guia completo de setup do WhisperX →](SETUP_WHISPERX.md)**

---

## 📂 Estrutura do Projeto

```text
.
├── app.py                              # Aplicação principal Streamlit (4 abas)
├── regex_callcenter_br.json            # Padrões regex regionais (BR call center)
├── requirements.txt                    # Dependências (Whisper + WhisperX)
├── SETUP_WHISPERX.md                   # Guia de setup do WhisperX
├── src/
│   ├── __init__.py
│   ├── transcriber.py                  # Motor Whisper clássico (heurística)
│   ├── transcriber_whisperx.py         # Motor WhisperX (Pyannote neural) — NOVO
│   ├── redactor.py                     # Redação PII LGPD + análise de insights LLM
│   ├── audio_utils.py                  # Métricas de qualidade de áudio (SNR, PAPR, etc.)
├── .streamlit/
│   ├── secrets.toml                    # API Keys (NUNCA versionar no git!)
│   └── secrets.toml.example            # Template de configuração
└── samples/                            # Pasta padrão para áudios locais e downloads HF
```

---

## 🚀 Instalação e Execução

### Pré-requisitos
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda
- `ffmpeg` instalado no sistema (`sudo apt install ffmpeg` no Linux)

### Passo a passo

```bash
# 1. Criar e ativar ambiente conda
conda create --name transcritor python=3.11
source activate transcritor

# 2. Instalar pip dentro do ambiente
conda install pip

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar as API Keys
# Edite .streamlit/secrets.toml com suas chaves

# 5. Iniciar o app
streamlit run app.py
```

### Aceleração GPU (opcional)
Para usar GPU NVIDIA com CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ⚙️ Configuração — API Keys

Edite o arquivo `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
GROQ_API_KEY   = "gsk_..."
HF_TOKEN       = "hf_..."
```

> ⚠️ **IMPORTANTE:** Nunca versione o `secrets.toml` no git. Adicione ao `.gitignore`.

---

## 🖥️ Funcionalidades do App

### Tab 1 — 🚀 Upload & Transcrição
- Upload de arquivos `.wav` ou `.mp3` (unitário ou múltiplos)
- Transcrição Whisper em Português do Brasil
- Diarização automática: **Atendente** vs **Cliente**
- Métricas de qualidade por áudio
- Download da transcrição em `.txt` ou `.json`

### Tab 2 — 📂 Processamento em Lote
- Seleciona uma pasta local e processa todos os `.wav`/`.mp3`
- Ideal para uso com a pasta `samples/` populada via HuggingFace

### Tab 3 — 🌐 Dataset HuggingFace
- Download automático de amostras do dataset call center multilíngue
- Dataset: [`AxonData/multilingual-call-center-speech-dataset`](https://huggingface.co/datasets/AxonData/multilingual-call-center-speech-dataset)
- Filtro por idioma (foco: **Português do Brasil**)
- Salva automaticamente em `samples/` para uso imediato

### Tab 4 — 📊 Dashboard & Métricas

| Métrica | Descrição |
|---|---|
| **Qualidade do Áudio** | Score 0-100% baseado em SNR + RMS |
| **SNR (dB)** | Signal-to-Noise Ratio — quanto de sinal vs ruído |
| **Nível de Ruído** | Baixo 🟢 / Médio 🟡 / Alto 🔴 |
| **Diarização (%)** | Confiança média da identificação de locutores |
| **Canal L/R** | Análise de energia por canal (estéreo) |
| **WER** | Word Error Rate (disponível com referência) |
| **Duração** | Tempo total do áudio em segundos |
| **Região** | Nordeste / Sul / Sudeste detectado por expressões regionais |

### Tab 5 — 🧠 Análise IA & Insights

- **Extração de insights** por LLM (OpenAI ou Groq):
  - Motivo da ligação
  - Sentimento do cliente (Positivo / Neutro / Negativo / Crítico)
  - Criticidade (Baixa / Média / Alta / Crítica)
  - Produto/Serviço mencionado
  - Ação recomendada
  - Keywords
- **RAG Coletivo** — TOP motivos de todas as transcrições em cache
- **Prompt personalizado** — faça perguntas sobre o conjunto de transcrições

---

## 🌐 Download de Amostras — CLI

```bash
# Baixar 5 amostras em português (padrão)
python gerar_amostra_dataset_callcenter.py

# Customizado
python gerar_amostra_dataset_callcenter.py --samples 10 --lang Portuguese Spanish --output samples

# Com token HuggingFace
python gerar_amostra_dataset_callcenter.py --samples 5 --token hf_...
```

---

## 🔒 Redação PII (LGPD)

O sistema detecta e redige automaticamente:

| Tipo PII | Exemplo | Redigido como |
|---|---|---|
| CPF | 123.456.789-00 | `[CPF REDIGIDO]` |
| CNPJ | 12.345.678/0001-90 | `[CNPJ REDIGIDO]` |
| RG | 12.345.678-9 | `[RG REDIGIDO]` |
| Telefone | (11) 99999-8888 | `[TELEFONE REDIGIDO]` |
| E-mail | teste@email.com | `[EMAIL REDIGIDO]` |
| Endereço | Rua das Flores, 123 | `[ENDEREÇO REDIGIDO]` |
| Cartão | 4111-1111-1111-1111 | `[CARTÃO REDIGIDO]` |

**Redação por IA:** ativa a redação contextual via LLM para detecção mais inteligente de PII.

---

## 🧠 Modelos Whisper Disponíveis

| Modelo | VRAM | Velocidade | Precisão |
|---|---|---|---|
| `tiny` | ~1 GB | ⚡⚡⚡⚡⚡ | ⭐ |
| `base` | ~1 GB | ⚡⚡⚡⚡ | ⭐⭐ |
| `small` | ~2 GB | ⚡⚡⚡ | ⭐⭐⭐ |
| `medium` | ~5 GB | ⚡⚡ | ⭐⭐⭐⭐ |
| `large-v3` | ~10 GB | ⚡ | ⭐⭐⭐⭐⭐ |

> 💡 **Recomendação:** `small` ou `medium` para equilíbrio velocidade/precisão em CPU. Use `large-v3` com GPU dedicada.

---

## 📐 Word Error Rate (WER)

O WER mede a qualidade da transcrição comparando com um texto de referência:

```
WER = (Substituições + Inserções + Deleções) / Total de Palavras × 100%
```

| WER | Classificação |
|---|---|
| < 10% | 🟢 Excelente |
| 10–30% | 🟡 Regular — considere modelo maior |
| > 30% | 🔴 Alto — use `large-v3` com GPU |

---

## 🤝 Diarização — Como Funciona

A diarização identifica quem está falando em cada segmento usando uma hierarquia de heurísticas:

1. **Canal estéreo L/R** — maior energia em cada canal (confiança 100%)
2. **Primeira fala** — geralmente é o Atendente (95%)
3. **Empresa detectada** — padrões de saudação corporativa (98%)
4. **Expressões regionais** — detecta sotaque nordestino, sulista (93%)
5. **Pattern matching** — regex específico para atendente vs. cliente
6. **Silêncio** — pausas longas sugerem troca de turno (70%)
7. **Alternância** — fallback baseado no histórico (55%)

---

## 🌐 Deploy no Streamlit Cloud

1. Faça fork deste repositório
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Configure os **Secrets** no painel do app:
   ```toml
   OPENAI_API_KEY = "sk-..."
   GROQ_API_KEY   = "gsk_..."
   HF_TOKEN       = "hf_..."
   ```
4. Deploy! O app ficará disponível em `https://seu-usuario-app.streamlit.app`

---

## 📦 Dependências Principais

```
streamlit        — Interface web
openai-whisper   — Motor de transcrição STT (clássico)
whisperx         — Motor de transcrição + diarização profissional (NOVO)
pyannote.audio   — Diarização neural via embeddings (NOVO)
faster-whisper   — Inferência otimizada do Whisper (NOVO)
torch/torchaudio — Backend PyTorch (CPU/GPU)
librosa          — Análise de áudio (SNR, RMS, etc.)
datasets         — HuggingFace Datasets
openai           — API GPT para insights e redação PII
groq             — API Groq (LLaMA) para insights e redação PII
pandas           — Manipulação de dados
jiwer            — Cálculo de WER
soundfile        — Salvar amostras de áudio
```

### Instalação Rápida no Linux:
```bash
# Dependências de sistema
sudo apt update
sudo apt install -y ffmpeg libsndfile1 sox

# Ambiente Python
pip install -r requirements.txt

# GPU (opcional, mas recomendado para WhisperX)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 👤 Autor

Desenvolvido como portfólio pessoal de IA aplicada a call center.  
Plataforma disponível no **Streamlit Cloud** para demonstração pública.

---

*Whisper Executive Transcriptor v2.0 | 2026 | Português do Brasil 🇧🇷*

# ⚡ Quick Start — WhisperX em 5 Minutos

Guia rápido para começar a usar o WhisperX com diarização Pyannote.

---

## 🚀 Instalação Rápida

### 1. Instalar Dependências de Sistema (Linux)
```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 sox
```

### 2. Instalar Dependências Python
```bash
# No diretório do projeto
pip install -r requirements.txt

# Se tiver GPU NVIDIA (recomendado):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🔑 Configurar HuggingFace Token

### Passo 1: Criar Token
1. Acesse: https://huggingface.co/settings/tokens
2. Clique **"New token"** → Type: **Read**
3. Copie o token (formato: `hf_xxxxxxxx...`)

### Passo 2: Aceitar Termos Pyannote
Clique em **"Agree and access repository"** nesses 2 links:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### Passo 3: Configurar no Projeto

**Opção A — Arquivo secrets.toml (Recomendado):**
```bash
# Criar arquivo de configuração
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Editar e adicionar seu token
nano .streamlit/secrets.toml
```

Arquivo `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "hf_seu_token_aqui"
```

**Opção B — Via Interface:**
Cole o token diretamente na interface Streamlit (no campo HF Token)

---

## ▶️ Executar Aplicação

```bash
streamlit run app.py
```

Acesse: http://localhost:8501

---

## 🎯 Usar WhisperX

1. No **Sidebar**, em **"Motor de Transcrição"**:
   - Selecione: **"WhisperX (Diarização Pro)"**
   
2. Verifique se aparece: **✅ HF_TOKEN configurado**

3. Configure speakers (call center padrão):
   - **Min Speakers:** 2
   - **Max Speakers:** 2

4. Upload de áudio e clique: **"▶ Iniciar Transcrição"**

---

## 🎛️ Opções Avançadas

### Forçar CPU (sem GPU):
- Marque **"Forçar CPU (desabilita GPU)"**
- Use modelo `small` ou `base` para melhor performance

### Escolher Modelo:
- **Desenvolvimento/Testes:** `small` (rápido, ~70-80% precisão)
- **Produção:** `medium` (balanceado, ~85% precisão)
- **Máxima Qualidade:** `large-v3` (lento, ~90-95% precisão)

---

## ✅ Verificar Instalação

```bash
# Verificar Python
python --version  # deve ser >= 3.8

# Verificar torch
python -c "import torch; print(torch.__version__)"

# Verificar GPU (se aplicável)
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Verificar whisperx
python -c "import whisperx; print('WhisperX: OK')"

# Verificar pyannote
python -c "from pyannote.audio import Pipeline; print('Pyannote: OK')"
```

---

## 🆚 Comparação Rápida

| Recurso | Whisper Clássico | WhisperX Pro |
|---------|------------------|--------------|
| Setup | ✅ Simples | ⚙️ Requer HF_TOKEN |
| Precisão Diarização | 60-75% | **85-95%** |
| Velocidade | ⚡ Rápido | 🐢 ~30% mais lento |
| Áudio Mono | ⚠️ Regular | ✅ Excelente |
| GPU | Opcional | Recomendada |

---

## 🐛 Problemas Comuns

### "No module named 'whisperx'"
```bash
pip install whisperx
```

### "PersonalAccessTokenError"
- Confirme que aceitou termos do Pyannote (links acima)
- Verifique se o HF_TOKEN está correto

### "CUDA out of memory"
- Marque **"Forçar CPU"** no app
- Use modelo menor (`small` ou `base`)

### Diarização falhou → Fallback ativo
- Sistema continua funcionando com alternância simples
- Verifique logs no terminal: `streamlit run app.py`

---

## 📚 Mais Informações

- **Setup Completo:** [SETUP_WHISPERX.md](SETUP_WHISPERX.md)
- **README Principal:** [README.md](README.md)

---

**🎉 Pronto! Agora você tem diarização profissional com WhisperX!**

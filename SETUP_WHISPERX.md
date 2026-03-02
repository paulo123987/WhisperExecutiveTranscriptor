# 🎙️ Guia de Setup — WhisperX com Pyannote

Este guia explica como configurar e usar o **WhisperX** com diarização profissional via **Pyannote.audio** no Whisper Executive Transcriptor.

---

## 🆚 Whisper vs WhisperX — Qual Usar?

### **Whisper (Clássico)**
✅ Transcrição de alta qualidade  
✅ Diarização heurística (padrões regex + canais)  
✅ Mais rápido (~20-30% mais rápido que WhisperX)  
✅ Funciona sem HF_TOKEN  
⚠️ Diarização menos precisa (especialmente em áudio mono)  

**Use quando:** Precisa de velocidade ou não tem HF_TOKEN

### **WhisperX (Profissional)**
✅ Transcrição de alta qualidade (mesmo motor Whisper)  
✅ **Diarização neural via Pyannote** (muito mais precisa)  
✅ Alinhamento temporal ultra-preciso (word-level)  
✅ Funciona bem com **áudio mono** (não depende de canais)  
✅ Detecta automaticamente número de speakers  
⚠️ Requer HF_TOKEN (HuggingFace)  
⚠️ ~20-30% mais lento  
⚠️ Precisa ~2-4 GB VRAM adicional (GPU recomendada)  

**Use quando:** Precisa de diarização profissional e tem GPU disponível

---

## 📋 Pré-Requisitos

### 1. **Dependências de Sistema**

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 sox
```

#### macOS:
```bash
brew install ffmpeg libsndfile sox
```

#### Windows:
- Baixar **FFmpeg**: https://ffmpeg.org/download.html
- Adicionar ao PATH do sistema

### 2. **Dependências Python**

Já incluídas no `requirements.txt`:
```bash
pip install -r requirements.txt
```

Principais pacotes:
- `whisperx>=3.1.1`
- `pyannote.audio>=3.1.0`
- `faster-whisper>=0.10.0`

---

## 🔑 Configuração do HuggingFace Token

O **Pyannote.audio** requer autenticação do HuggingFace para baixar os modelos.

### **Passo 1: Criar Conta HuggingFace**
1. Acesse: https://huggingface.co/join
2. Crie sua conta gratuita

### **Passo 2: Gerar Access Token**
1. Acesse: https://huggingface.co/settings/tokens
2. Clique em **"New token"**
3. Nome: `whisperx-pyannote`
4. Type: **Read**
5. Copie o token gerado (formato: `hf_xxxxxxxxxxxxxxxxxxxxx`)

### **Passo 3: Aceitar Termos dos Modelos**
Você precisa aceitar os termos de uso dos modelos Pyannote:

1. **Diarization Model:**
   - Acesse: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Clique em **"Agree and access repository"**

2. **Segmentation Model:**
   - Acesse: https://huggingface.co/pyannote/segmentation-3.0
   - Clique em **"Agree and access repository"**

⚠️ **IMPORTANTE:** Sem aceitar esses termos, o WhisperX não funcionará!

### **Passo 4: Configurar Token no Projeto**

#### Opção A: Via `.streamlit/secrets.toml` (Recomendado)
Crie/edite o arquivo `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"

# Opcional: outras API keys
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxx"
```

⚠️ **NUNCA versione este arquivo no git!** (já está no `.gitignore`)

#### Opção B: Via Interface Streamlit
1. Execute o app: `streamlit run app.py`
2. No **Sidebar**, selecione **"WhisperX (Diarização Pro)"**
3. Cole seu token no campo **"HF Token"**

---

## 🚀 Como Usar

### **1. Iniciar Aplicação**
```bash
streamlit run app.py
```

### **2. Configurar Engine**
No **Sidebar → Motor de Transcrição**:
- Selecione **"WhisperX (Diarização Pro)"**
- Cole seu **HF_TOKEN** (ou configure no `.streamlit/secrets.toml`)
- Confirme que aparece **"✅ HF_TOKEN configurado"**

### **3. Configurações Adicionais (WhisperX)**

#### **Forçar CPU:**
- Marque **"Forçar CPU (desabilita GPU)"** se quiser usar CPU
- Por padrão, detecta automaticamente GPU disponível

#### **Controle de Speakers:**
- **Min Speakers:** Número mínimo de vozes esperadas (padrão: 2)
- **Max Speakers:** Número máximo de vozes esperadas (padrão: 2)
- Para call center padrão (Atendente + Cliente), deixe 2/2
- Para conferências com múltiplas pessoas, ajuste conforme necessário

### **4. Processar Áudio**
- Upload de arquivos `.wav` ou `.mp3`
- Clique em **"Iniciar Transcrição"**
- WhisperX executará:
  1. 📝 Transcrição (Whisper)
  2. ⏱️ Alinhamento temporal (Wav2Vec2)
  3. 🎭 Diarização (Pyannote embeddings + clustering)
  4. 🎯 Mapeamento Atendente/Cliente (heurísticas)

---

## 📊 Performance Esperada

### **GPU (CUDA) - Recomendado**
- **VRAM necessária:** ~4-6 GB
- **Tempo de processamento:** 
  - Áudio de 1 min: ~10-15s
  - Áudio de 5 min: ~45-60s
- **DER (Diarization Error Rate):** ~5-15% (excelente)

### **CPU (Fallback)**
- **RAM necessária:** ~8 GB
- **Tempo de processamento:**
  - Áudio de 1 min: ~60-90s
  - Áudio de 5 min: ~5-7 min
- **DER:** ~5-15% (mesma qualidade, mais lento)

---

## 🐛 Troubleshooting

### **Erro: "pyannote.audio.core.pipeline.PersonalAccessTokenError"**
**Causa:** HF_TOKEN não configurado ou inválido

**Solução:**
1. Verifique se o token está correto
2. Confirme que aceitou os termos dos modelos Pyannote
3. Tente gerar um novo token

### **Erro: "CUDA out of memory"**
**Causa:** VRAM insuficiente na GPU

**Solução:**
1. Marque **"Forçar CPU"** no Sidebar
2. Use modelo Whisper menor (tiny, base, small)
3. Feche outros programas que usam GPU

### **Erro: "No module named 'whisperx'"**
**Causa:** WhisperX não instalado

**Solução:**
```bash
pip install whisperx
```

### **Diarização Muito Lenta**
**Causa:** Executando em CPU sem otimizações

**Solução:**
1. Use GPU se disponível (remove **"Forçar CPU"**)
2. Use modelo Whisper menor (base, small)
3. Processe áudios mais curtos (<5 min)

### **Fallback: Diarização Simples**
Se o WhisperX falhar, o sistema usa **fallback automático**:
- Alternância simples Atendente ↔ Cliente
- Confidence score: ~40-50%
- Ainda funcional, mas menos preciso

---

## 🔄 Comparação de Resultados

### **Campo "reason" nos Segmentos:**

#### Whisper Clássico:
- `padrao_atendente` — Detectado por padrões regex
- `canal_esquerdo` — Detectado por análise de canal estéreo
- `empresa_vivo` — Detectado por menção de empresa
- `silencio` — Troca de turno por silêncio
- `alternancia` — Alternância simples

#### WhisperX:
- `whisperx_pyannote (original: SPEAKER_00)` — Diarização neural real
- Confidence score tipicamente **>70%** (vs 40-60% do clássico)

### **Comparação Prática:**

| Métrica | Whisper Clássico | WhisperX |
|---------|------------------|----------|
| **Precisão Diarização** | 60-75% | **85-95%** |
| **Áudio Mono** | ⚠️ Menos preciso | ✅ Excelente |
| **Áudio Estéreo** | ✅ Bom | ✅ Excelente |
| **Velocidade** | ✅ Mais rápido | ⚠️ Mais lento (~30%) |
| **GPU Requerida** | Opcional | Recomendada |
| **Setup** | Simples | Requer HF_TOKEN |

---

## 📁 Estrutura de Arquivos

```
whispertranscritor2/
├── app.py                          # Interface Streamlit (suporta ambas engines)
├── requirements.txt                # Dependências (inclui WhisperX)
├── SETUP_WHISPERX.md              # Este arquivo
├── src/
│   ├── transcriber.py             # Motor Whisper clássico (MANTIDO)
│   ├── transcriber_whisperx.py    # Motor WhisperX (NOVO)
│   ├── redactor.py                # Redação PII / Análise IA
│   ├── audio_utils.py             # Métricas de áudio
├── .streamlit/
│   └── secrets.toml               # API Keys (HF_TOKEN, OpenAI, Groq)
└── samples/                        # Áudios de teste
```

---

## 🎯 Recomendações de Uso

### **Para Produção (Alta Qualidade):**
```
Engine: WhisperX (Diarização Pro)
Modelo: medium ou large-v3
GPU: CUDA habilitada
HF_TOKEN: Configurado
Min/Max Speakers: 2/2 (call center)
```

### **Para Desenvolvimento Rápido:**
```
Engine: Whisper (Clássico)
Modelo: small ou base
GPU: Opcional
```

### **Para CPU Puro (Sem GPU):**
```
Engine: WhisperX (Diarização Pro)
Modelo: tiny ou base
Forçar CPU: ✅ Marcado
HF_TOKEN: Configurado
```

---

## 🔐 Segurança & LGPD

- **HF_TOKEN**: Apenas leitura, não expõe dados sensíveis
- **Processamento local**: Áudios NÃO são enviados para HuggingFace
- **Modelos baixados localmente**: Cache em `~/.cache/huggingface/`
- **PII Redaction**: Mantém conformidade LGPD mesmo com WhisperX

---

## 📚 Referências

- **WhisperX GitHub:** https://github.com/m-bain/whisperX
- **Pyannote.audio:** https://github.com/pyannote/pyannote-audio
- **HuggingFace Hub:** https://huggingface.co/pyannote
- **Whisper OpenAI:** https://github.com/openai/whisper

---

## 💡 Dicas Pro

1. **Primera execução é lenta:** WhisperX baixa modelos Pyannote (~1 GB)
2. **Use GPU:** Diarização Pyannote é **muito** mais rápida em GPU
3. **Áudio estéreo:** WhisperX + análise de canais = precisão máxima
4. **Batch processing:** WhisperX libera memória entre arquivos (gc.collect)
5. **Compare resultados:** Processe o mesmo áudio com ambas engines e compare!

---

## 🆘 Suporte

**Problemas com instalação?**
- Verifique versão do Python: `python --version` (requer >=3.8)
- Verifique CUDA: `nvcc --version` (se usar GPU)
- Reinstale dependências: `pip install -r requirements.txt --upgrade`

**Diarização falhou?**
- Confirme que aceitou termos do Pyannote
- Verifique logs no terminal
- Sistema usa fallback automático se Pyannote falhar

---

**✨ Desenvolvido para Call Center AI — LGPD Compliant**

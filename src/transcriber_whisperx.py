"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   🎙️ WhisperX Transcriber — Diarização Profissional com Pyannote.audio     ║
║   Motor de transcrição avançado com alinhamento temporal preciso            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import re
import json
import torch
import whisperx
import gc
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, List, Tuple
from src.audio_utils import AudioAnalyzer
from src.redactor import PIIRedactor

logger = logging.getLogger(__name__)


class CallCenterTranscriberX:
    """
    Motor de transcrição e diarização profissional para call center.
    
    Usa WhisperX com Pyannote.audio 3.x para:
    - Transcrição de alta qualidade (Whisper)
    - Alinhamento temporal preciso (Wav2Vec2)
    - Diarização neural (Pyannote embeddings + clustering)
    """

    def __init__(
        self,
        model_name: str = "small",
        regex_file: str = "regex_callcenter_br.json",
        openai_key: str = None,
        groq_key: str = None,
        hf_token: str = None,
        force_cpu: bool = False,
    ):
        """
        Inicializa o motor WhisperX.
        
        Args:
            model_name: Modelo Whisper (tiny, base, small, medium, large-v3)
            regex_file: Arquivo JSON com padrões regex para classificação
            openai_key: API key OpenAI (para redação/insights)
            groq_key: API key Groq (para redação/insights)
            hf_token: HuggingFace token (obrigatório para Pyannote)
            force_cpu: Força uso de CPU mesmo com GPU disponível
        """
        # Dispositivo
        if force_cpu:
            self.device = "cpu"
            self.compute_type = "int8"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        logger.info(f"🚀 WhisperX: Dispositivo={self.device}, Compute={self.compute_type}")
        
        # HuggingFace Token
        self.hf_token = hf_token
        if not self.hf_token:
            logger.warning("⚠️ HF_TOKEN não fornecido. Diarização pode falhar!")
        
        # Modelo Whisper via WhisperX
        logger.info(f"Carregando modelo WhisperX '{model_name}'...")
        self.model = whisperx.load_model(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
            language="pt",
        )
        self.model_name = model_name
        
        # Redactor
        self.redactor = PIIRedactor(openai_api_key=openai_key, groq_api_key=groq_key)
        
        # Carrega regex patterns
        regex_path = Path(regex_file)
        if not regex_path.exists():
            regex_path = Path(__file__).parent.parent / regex_file
        with open(regex_path, "r", encoding="utf-8") as f:
            self.patterns = json.load(f)
        
        self.regex = self._compile_regexes()
        self.known_companies = list(self.patterns["atendente"]["empresas"].keys())
    
    def _compile_regexes(self) -> dict:
        """Compila padrões regex para identificação de speakers."""
        compiled = {
            "atendente": [re.compile(p, re.IGNORECASE) for p in self.patterns["atendente"]["universal"]],
            "cliente_nordeste": [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["nordeste"]],
            "cliente_sul": [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["sul"]],
            "cliente_informal": [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["sudeste_informal"]],
            "problemas": [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["problemas_tecnicos"]],
            "intencao": [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["intencao_forte"]],
            "identidade": [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["identidade"]],
        }
        compiled["empresas"] = {}
        for emp, pats in self.patterns["atendente"]["empresas"].items():
            compiled["empresas"][emp] = [re.compile(p, re.IGNORECASE) for p in pats]
        return compiled
    
    def _score(self, text: str, patterns: list) -> int:
        """Conta quantos padrões regex matcham no texto."""
        return sum(1 for p in patterns if p.search(text))
    
    def detect_region(self, text: str) -> str:
        """Detecta região brasileira baseado em expressões regionais."""
        t = text.lower()
        if self._score(t, self.regex["cliente_nordeste"]) > 0:
            return "Nordeste"
        if self._score(t, self.regex["cliente_sul"]) > 0:
            return "Sul"
        return "Sudeste/Outros"
    
    def detect_company(self, text: str) -> str:
        """Detecta empresa mencionada na conversa."""
        t = text.lower()
        for emp in self.known_companies:
            if any(p.search(t) for p in self.regex["empresas"][emp]) or emp in t:
                return emp.capitalize()
        return "Desconhecida"
    
    def _map_speakers_to_roles(
        self,
        diarized_segments: List[Dict],
        full_text: str,
        company: str,
        region: str,
        is_stereo: bool,
        waveform: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> List[Dict]:
        """
        Mapeia SPEAKER_00, SPEAKER_01, etc. para 'Atendente' ou 'Cliente'.
        
        Estratégia:
        1. Analisa todos os segmentos de cada speaker
        2. Usa heurísticas (regex, canal, primeira fala) para classificar
        3. Atribui o speaker com maior score de atendente como "Atendente"
        4. Demais speakers são "Cliente"
        """
        # Agrupa segmentos por speaker
        from collections import defaultdict
        speaker_groups = defaultdict(list)
        for seg in diarized_segments:
            speaker_groups[seg["speaker"]].append(seg)
        
        # Calcula score para cada speaker
        speaker_scores = {}
        
        for speaker_id, segments in speaker_groups.items():
            # Concatena todo o texto desse speaker
            speaker_text = " ".join(s["text"] for s in segments).lower()
            
            # Score baseado em padrões
            score_atendente = self._score(speaker_text, self.regex["atendente"])
            score_cliente = sum(
                self._score(speaker_text, self.regex[k])
                for k in ["problemas", "intencao", "identidade", "cliente_informal"]
            )
            
            # Bonus por empresa conhecida
            emp_key = company.lower()
            if emp_key in self.regex["empresas"]:
                score_atendente += self._score(speaker_text, self.regex["empresas"][emp_key]) * 2
            
            # Bonus se é o primeiro speaker (geralmente atendente)
            first_segment = min(diarized_segments, key=lambda x: x["start"])
            if first_segment["speaker"] == speaker_id:
                score_atendente += 3
            
            # Bonus por canal estéreo (se disponível)
            if is_stereo and waveform is not None:
                left_energy = 0
                right_energy = 0
                for seg in segments:
                    mid_idx = int(((seg["start"] + seg["end"]) / 2.0) * sample_rate)
                    if 0 <= mid_idx < waveform.shape[1]:
                        half_win = int(0.05 * sample_rate)
                        ini = max(0, mid_idx - half_win)
                        fim = min(waveform.shape[1], mid_idx + half_win)
                        left_energy += float(waveform[0, ini:fim].abs().mean().item())
                        right_energy += float(waveform[1, ini:fim].abs().mean().item())
                
                # Canal esquerdo geralmente é atendente
                if left_energy > right_energy * 1.2:
                    score_atendente += 2
                elif right_energy > left_energy * 1.2:
                    score_cliente += 2
            
            # Score final
            speaker_scores[speaker_id] = {
                "atendente": score_atendente,
                "cliente": score_cliente,
                "net_score": score_atendente - score_cliente,
            }
        
        # Identifica quem é atendente (maior net_score)
        if speaker_scores:
            atendente_speaker = max(speaker_scores.items(), key=lambda x: x[1]["net_score"])[0]
        else:
            atendente_speaker = "SPEAKER_00"  # fallback
        
        logger.info(f"🎯 Mapeamento: {atendente_speaker} → Atendente")
        logger.debug(f"Speaker scores: {speaker_scores}")
        
        # Mapeia todos os segmentos
        mapped_segments = []
        for seg in diarized_segments:
            speaker_label = "Atendente" if seg["speaker"] == atendente_speaker else "Cliente"
            
            # Confiança baseada no score
            if seg["speaker"] in speaker_scores:
                scores = speaker_scores[seg["speaker"]]
                confidence = min(1.0, abs(scores["net_score"]) / 10.0 + 0.5)
            else:
                confidence = 0.5
            
            mapped_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": speaker_label,
                "confidence": round(confidence, 3),
                "reason": f"whisperx_pyannote (original: {seg['speaker']})",
                "channel": None,  # WhisperX não usa canal diretamente
            })
        
        return mapped_segments
    
    def process_audio(
        self,
        audio_path,
        redact: bool = True,
        llm_redact: bool = False,
        redaction_provider: str = "openai",
        redaction_model: str = None,
        run_insights: bool = False,
        insights_provider: str = "openai",
        insights_model: str = None,
        min_speakers: int = 2,
        max_speakers: int = 2,
    ) -> dict:
        """
        Processa áudio com WhisperX (transcrição + alinhamento + diarização).
        
        Args:
            audio_path: Caminho do arquivo de áudio
            redact: Ativa redação PII via regex
            llm_redact: Ativa redação PII via LLM
            redaction_provider: Provider LLM para redação (openai/groq)
            redaction_model: Modelo LLM para redação
            run_insights: Ativa análise de insights via LLM
            insights_provider: Provider LLM para insights
            insights_model: Modelo LLM para insights
            min_speakers: Número mínimo de speakers (padrão: 2)
            max_speakers: Número máximo de speakers (padrão: 2)
        
        Returns:
            dict: Resultado completo com transcrição, métricas, insights, etc.
        """
        audio_path = Path(audio_path)
        logger.info(f"🎙️ Processando: {audio_path.name}")
        
        # ── 1. MÉTRICAS DE ÁUDIO ──
        metrics = AudioAnalyzer.get_audio_metrics(audio_path)
        is_stereo = metrics.get("is_stereo", False)
        
        # Carrega waveform para análise de canal (se necessário)
        waveform = None
        sample_rate = 16000
        if is_stereo:
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as e:
                logger.warning(f"Não foi possível carregar waveform para análise de canal: {e}")
        
        # ── 2. TRANSCRIÇÃO WHISPER ──
        logger.info("📝 Executando transcrição Whisper...")
        audio = whisperx.load_audio(str(audio_path))
        result = self.model.transcribe(audio, batch_size=16)
        full_text = " ".join([seg["text"] for seg in result["segments"]])
        
        # Detecta região e empresa
        region = self.detect_region(full_text)
        company = self.detect_company(full_text)
        
        # ── 3. ALINHAMENTO TEMPORAL ──
        logger.info("⏱️ Alinhando timestamps...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code="pt",
                device=self.device,
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            # Libera memória
            del model_a
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Alinhamento falhou: {e}. Continuando com timestamps originais.")
        
        # ── 4. DIARIZAÇÃO ──
        diarized_segments = []
        diarization_score = 0.0
        
        if self.hf_token:
            logger.info("🎭 Executando diarização Pyannote...")
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device,
                )
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
                # Libera memória
                del diarize_model
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Extrai segmentos diarizados
                for seg in result["segments"]:
                    # WhisperX pode retornar palavras individuais ou segmentos
                    speaker = seg.get("speaker", "SPEAKER_00")
                    diarized_segments.append({
                        "start": round(float(seg["start"]), 2),
                        "end": round(float(seg["end"]), 2),
                        "text": seg["text"].strip(),
                        "speaker": speaker,
                    })
                
                # Mapeia speakers para Atendente/Cliente
                diarized_segments = self._map_speakers_to_roles(
                    diarized_segments,
                    full_text,
                    company,
                    region,
                    is_stereo,
                    waveform,
                    sample_rate,
                )
                
                # Calcula score de diarização (média das confianças)
                if diarized_segments:
                    diarization_score = sum(s["confidence"] for s in diarized_segments) / len(diarized_segments) * 100
                
            except Exception as e:
                logger.error(f"❌ Diarização falhou: {e}")
                logger.info("Usando fallback: alternância simples")
                # Fallback: alternar entre Atendente/Cliente
                prev_speaker = "Atendente"
                for i, seg in enumerate(result["segments"]):
                    speaker = "Atendente" if i % 2 == 0 else "Cliente"
                    diarized_segments.append({
                        "start": round(float(seg["start"]), 2),
                        "end": round(float(seg["end"]), 2),
                        "text": seg["text"].strip(),
                        "speaker": speaker,
                        "confidence": 0.5,
                        "reason": "fallback_alternancia",
                        "channel": None,
                    })
                diarization_score = 50.0
        else:
            # Sem HF_TOKEN: fallback simples
            logger.warning("⚠️ Sem HF_TOKEN. Usando diarização simplificada.")
            for i, seg in enumerate(result["segments"]):
                speaker = "Atendente" if i % 2 == 0 else "Cliente"
                diarized_segments.append({
                    "start": round(float(seg["start"]), 2),
                    "end": round(float(seg["end"]), 2),
                    "text": seg["text"].strip(),
                    "speaker": speaker,
                    "confidence": 0.4,
                    "reason": "no_hf_token_fallback",
                    "channel": None,
                })
            diarization_score = 40.0
        
        # ── 5. REDAÇÃO PII ──
        if redact:
            logger.info("🔒 Aplicando redação PII...")
            for seg in diarized_segments:
                if llm_redact:
                    seg["text"] = self.redactor.redact_with_llm(
                        seg["text"],
                        provider=redaction_provider,
                        model=redaction_model,
                    )
                else:
                    seg["text"] = self.redactor.redact_regex(seg["text"])
        
        # ── 6. MÉTRICAS FINAIS ──
        atendente_segs = [s for s in diarized_segments if s["speaker"] == "Atendente"]
        cliente_segs = [s for s in diarized_segments if s["speaker"] == "Cliente"]
        
        # ── 7. INSIGHTS IA ──
        insights = {}
        if run_insights:
            logger.info("🧠 Extraindo insights via LLM...")
            insights = self.redactor.analyze_with_llm(
                full_text,
                provider=insights_provider,
                model=insights_model,
            )
        
        # ── 8. RESULTADO FINAL ──
        return {
            "filename": audio_path.name,
            "metrics": metrics,
            "segments": diarized_segments,
            "full_text": full_text,
            "region": region,
            "company": company,
            "is_stereo": is_stereo,
            "total_segments": len(diarized_segments),
            "atendente_segments": len(atendente_segs),
            "cliente_segments": len(cliente_segs),
            "diarization_score": round(diarization_score, 2),
            "insights": insights,
            "timestamp": datetime.now().isoformat(),
            "engine": f"whisperx_{self.model_name}",
            "device": self.device,
        }

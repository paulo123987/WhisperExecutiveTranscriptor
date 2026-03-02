import re
import json
import torch
import whisper
import torchaudio
from pathlib import Path
from datetime import datetime
import logging
from src.audio_utils import AudioAnalyzer
from src.redactor import PIIRedactor

logger = logging.getLogger(__name__)


class CallCenterTranscriber:
    """Motor de transcrição e diarização para call center com Whisper."""

    def __init__(
        self,
        model_name: str = "small",
        regex_file: str = "regex_callcenter_br.json",
        openai_key: str = None,
        groq_key: str = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Carregando modelo Whisper '{model_name}' no dispositivo: {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)
        self.redactor = PIIRedactor(openai_api_key=openai_key, groq_api_key=groq_key)

        # Carrega regex
        regex_path = Path(regex_file)
        if not regex_path.exists():
            regex_path = Path(__file__).parent.parent / regex_file
        with open(regex_path, "r", encoding="utf-8") as f:
            self.patterns = json.load(f)

        self.regex = self._compile_regexes()
        self.known_companies = list(self.patterns["atendente"]["empresas"].keys())

    def _compile_regexes(self) -> dict:
        compiled = {
            "atendente":       [re.compile(p, re.IGNORECASE) for p in self.patterns["atendente"]["universal"]],
            "cliente_nordeste":[re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["nordeste"]],
            "cliente_sul":     [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["sul"]],
            "cliente_informal":[re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["sudeste_informal"]],
            "problemas":       [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["problemas_tecnicos"]],
            "intencao":        [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["intencao_forte"]],
            "identidade":      [re.compile(p, re.IGNORECASE) for p in self.patterns["cliente"]["identidade"]],
        }
        compiled["empresas"] = {}
        for emp, pats in self.patterns["atendente"]["empresas"].items():
            compiled["empresas"][emp] = [re.compile(p, re.IGNORECASE) for p in pats]
        return compiled

    def _score(self, text: str, patterns: list) -> int:
        return sum(1 for p in patterns if p.search(text))

    def detect_region(self, text: str) -> str:
        t = text.lower()
        if self._score(t, self.regex["cliente_nordeste"]) > 0:
            return "Nordeste"
        if self._score(t, self.regex["cliente_sul"]) > 0:
            return "Sul"
        return "Sudeste/Outros"

    def detect_company(self, text: str) -> str:
        t = text.lower()
        for emp in self.known_companies:
            if any(p.search(t) for p in self.regex["empresas"][emp]) or emp in t:
                return emp.capitalize()
        return "Desconhecida"

    def identify_speaker(
        self,
        text: str,
        index: int,
        prev_speaker: str,
        silence: float,
        channel: str = None,
        company: str = None,
        region: str = None,
    ) -> tuple:
        """
        Identifica o locutor do segmento.
        Retorna (speaker, confidence, reason).
        """
        t = text.lower()

        # 1. Canal estéreo (maior confiança)
        if channel == "esquerdo":
            return "Atendente", 1.0, "canal_esquerdo"
        if channel == "direito":
            return "Cliente", 1.0, "canal_direito"

        # 2. Primeira fala costuma ser do atendente
        if index == 0:
            return "Atendente", 0.95, "primeira_fala"

        # 3. Empresa conhecida
        emp_key = (company or "").lower()
        if emp_key in self.regex["empresas"]:
            if self._score(t, self.regex["empresas"][emp_key]) > 0:
                return "Atendente", 0.98, f"empresa_{emp_key}"

        # 4. Expressões regionais
        if region == "Nordeste" and self._score(t, self.regex["cliente_nordeste"]) > 0:
            return "Cliente", 0.93, "regiao_nordeste"
        if region == "Sul" and self._score(t, self.regex["cliente_sul"]) > 0:
            return "Cliente", 0.93, "regiao_sul"

        # 5. Pattern matching geral
        score_at = self._score(t, self.regex["atendente"])
        score_cl = sum(
            self._score(t, self.regex[k])
            for k in ["problemas", "intencao", "identidade", "cliente_informal"]
        )

        if score_at > score_cl:
            return "Atendente", min(0.9 + score_at * 0.03, 1.0), "padrao_atendente"
        if score_cl > score_at:
            return "Cliente", min(0.85 + score_cl * 0.03, 1.0), "padrao_cliente"

        # 6. Silêncio longo → troca de turno
        if silence > 2.0 and prev_speaker:
            new = "Cliente" if prev_speaker == "Atendente" else "Atendente"
            return new, 0.70, "silencio"

        # 7. Alternância simples
        if prev_speaker:
            new = "Cliente" if prev_speaker == "Atendente" else "Atendente"
            return new, 0.55, "alternancia"

        return "Atendente", 0.40, "fallback"

    def _load_waveform(self, audio_path: Path):
        """
        Carrega waveform de forma robusta, com fallback para soundfile/numpy.
        Retorna (waveform_tensor, sample_rate).
        """
        try:
            return torchaudio.load(audio_path)
        except Exception as e1:
            logger.warning(f"torchaudio.load falhou ({e1}), tentando soundfile...")
            try:
                import soundfile as sf
                import torch
                data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
                # data shape: (samples, channels) → precisamos (channels, samples)
                waveform = torch.from_numpy(data.T)
                return waveform, sr
            except Exception as e2:
                raise RuntimeError(
                    f"Não foi possível carregar o áudio '{audio_path.name}'.\n"
                    f"  torchaudio: {e1}\n"
                    f"  soundfile: {e2}\n"
                    f"Instale ffmpeg: sudo apt install ffmpeg"
                ) from e2

    def _ensure_wav(self, audio_path: Path) -> Path:
        """
        Se o arquivo de áudio não puder ser lido diretamente, converte para WAV
        temporário 16kHz mono. Retorna o caminho do arquivo a ser usado.
        """
        import tempfile, os
        try:
            # Verifica se torchaudio lê normalmente
            torchaudio.load(audio_path)
            return audio_path
        except Exception:
            pass
        # Tenta converter com soundfile → WAV temporário
        try:
            import soundfile as sf
            import torch
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)  # (channels, samples)
            # Resample para 16kHz se necessário
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            # Converte para mono
            if waveform.shape[0] > 1:
                waveform_mono = waveform.mean(dim=0, keepdim=True)
            else:
                waveform_mono = waveform
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            torchaudio.save(tmp.name, waveform_mono, 16000)
            return Path(tmp.name)
        except Exception as e:
            logger.warning(f"_ensure_wav falhou: {e}")
            return audio_path  # Tenta assim mesmo

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
    ) -> dict:
        """
        Processa um arquivo de áudio completo:
        - Análise de qualidade
        - Transcrição Whisper
        - Diarização heurística
        - Redação PII (opcional)
        - Análise de insights (opcional)
        """
        audio_path = Path(audio_path)
        metrics = AudioAnalyzer.get_audio_metrics(audio_path)

        # Garante que o arquivo está num formato compatível
        wav_path = self._ensure_wav(audio_path)
        _tmp_created = wav_path != audio_path  # arquivo temporário criado

        try:
            # — Transcrição Whisper —
            result = self.model.transcribe(str(wav_path), language="pt")
            full_text = result["text"]

            region = self.detect_region(full_text)
            company = self.detect_company(full_text)

            # — Diarização —
            waveform, sr = self._load_waveform(wav_path)
            is_stereo = waveform.shape[0] == 2
        finally:
            # Remove WAV temporário se foi criado
            if _tmp_created and wav_path.exists():
                import os; os.unlink(wav_path)

        processed_segments = []
        prev_speaker = None
        last_end = 0.0

        for i, seg in enumerate(result["segments"]):
            silence = max(0.0, float(seg["start"]) - last_end)
            channel = None

            if is_stereo:
                mid_idx = int(((seg["start"] + seg["end"]) / 2.0) * sr)
                if 0 <= mid_idx < waveform.shape[1]:
                    half_win = int(0.05 * sr)  # 50ms
                    ini = max(0, mid_idx - half_win)
                    fim = min(waveform.shape[1], mid_idx + half_win)
                    e1 = float(waveform[0, ini:fim].abs().mean().item())
                    e2 = float(waveform[1, ini:fim].abs().mean().item())
                    if e1 > e2 * 1.2:
                        channel = "esquerdo"
                    elif e2 > e1 * 1.2:
                        channel = "direito"

            text = seg["text"].strip()
            speaker, conf, reason = self.identify_speaker(
                text, i, prev_speaker, silence, channel, company, region
            )

            # — Redação PII —
            if redact:
                if llm_redact:
                    text = self.redactor.redact_with_llm(
                        text, provider=redaction_provider, model=redaction_model
                    )
                else:
                    text = self.redactor.redact_regex(text)

            processed_segments.append({
                "start": round(float(seg["start"]), 2),
                "end": round(float(seg["end"]), 2),
                "text": text,
                "speaker": speaker,
                "confidence": round(conf, 3),
                "reason": reason,
                "channel": channel,
            })
            prev_speaker = speaker
            last_end = float(seg["end"])

        # — Métricas de diarização —
        atendente_segs = [s for s in processed_segments if s["speaker"] == "Atendente"]
        cliente_segs   = [s for s in processed_segments if s["speaker"] == "Cliente"]
        avg_conf = (
            sum(s["confidence"] for s in processed_segments) / len(processed_segments)
            if processed_segments else 0.0
        )
        diarization_score = round(min(100.0, avg_conf * 100), 2)

        # — Insights IA —
        insights = {}
        if run_insights:
            insights = self.redactor.analyze_with_llm(
                full_text, provider=insights_provider, model=insights_model
            )

        return {
            "filename": audio_path.name,
            "metrics": metrics,
            "segments": processed_segments,
            "full_text": full_text,
            "region": region,
            "company": company,
            "is_stereo": is_stereo,
            "total_segments": len(processed_segments),
            "atendente_segments": len(atendente_segs),
            "cliente_segments": len(cliente_segs),
            "diarization_score": diarization_score,
            "insights": insights,
            "timestamp": datetime.now().isoformat(),
        }

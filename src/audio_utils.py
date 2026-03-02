import torchaudio
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioAnalyzer:
    """Analisador de áudio com métricas avançadas para call center."""

    @staticmethod
    def get_audio_metrics(audio_path) -> dict:
        """
        Calcula métricas completas do arquivo de áudio.
        Retorna dicionário com qualidade, ruído, canais, duração e outras métricas.
        """
        try:
            waveform, sr = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sr
            num_channels = waveform.shape[0]
            is_stereo = num_channels == 2

            # — Energia por canal —
            channel_energy = []
            channel_db = []
            for i in range(num_channels):
                energy = float(waveform[i].abs().mean().item())
                db = float(20 * np.log10(energy + 1e-10))
                channel_energy.append(round(energy, 6))
                channel_db.append(round(db, 2))

            # — Análise com librosa —
            if LIBROSA_AVAILABLE:
                y_mono, _ = librosa.load(str(audio_path), sr=sr, mono=True)

                # RMS (Root Mean Square) — nível médio do sinal
                rms_frames = librosa.feature.rms(y=y_mono)
                avg_rms = float(np.mean(rms_frames))
                max_rms = float(np.max(rms_frames))

                # SNR estimado: razão sinal/ruído
                # Estima ruído como percentil 10 do RMS (amostras mais silenciosas)
                noise_floor = float(np.percentile(rms_frames, 10) + 1e-10)
                snr_db = float(20 * np.log10((avg_rms + 1e-10) / noise_floor))
                snr_db = round(min(snr_db, 60), 2)  # cap em 60 dB

                # Peak to Average Power Ratio
                peak = float(np.max(np.abs(y_mono)))
                papr = float(10 * np.log10((peak ** 2) / (avg_rms ** 2 + 1e-10))) if avg_rms > 0 else 0.0
                papr = round(papr, 2)

                # Zero Crossing Rate — indicador de ruído/qualidade
                zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y_mono)))

                # Spectral Centroid (frequência dominante média)
                centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))

                # Score de qualidade principal (0-100)
                # Combina: RMS médio, SNR, e PAPR
                snr_score = min(100, max(0, snr_db * 1.5))       # SNR normalizado
                rms_score = min(100, max(0, avg_rms * 10000))     # RMS normalizado
                papr_score = min(100, max(0, 100 - papr * 1.5))  # PAPR: menor é melhor

                quality_score = round((snr_score * 0.5 + rms_score * 0.3 + papr_score * 0.2), 2)
                quality_score = round(min(100.0, max(0.0, quality_score)), 2)

                # Classificação do nível de ruído
                if snr_db >= 30:
                    noise_level = "Baixo 🟢"
                elif snr_db >= 15:
                    noise_level = "Médio 🟡"
                else:
                    noise_level = "Alto 🔴"

                # Classificação da qualidade
                if quality_score >= 70:
                    quality_label = "Boa 🟢"
                elif quality_score >= 40:
                    quality_label = "Regular 🟡"
                else:
                    quality_label = "Ruim 🔴"

            else:
                # Fallback sem librosa
                avg_rms = float(waveform.abs().mean().item())
                max_rms = float(waveform.abs().max().item())
                snr_db = 20.0
                papr = 10.0
                zcr = 0.0
                centroid = 0.0
                quality_score = round(min(100, max(0, avg_rms * 5000)), 2)
                noise_level = "Desconhecido ⚪"
                quality_label = "Desconhecida ⚪"

            # — Canal estéreo: verificar desequilíbrio L/R —
            stereo_balance = "N/A"
            left_dominance = False
            right_dominance = False
            if is_stereo and len(channel_energy) == 2:
                e_l, e_r = channel_energy[0], channel_energy[1]
                total_e = e_l + e_r + 1e-10
                ratio = e_l / (e_r + 1e-10)
                if ratio > 1.5:
                    stereo_balance = "Canal Esquerdo dominante 🔈"
                    left_dominance = True
                elif ratio < 0.67:
                    stereo_balance = "Canal Direito dominante 🔉"
                    right_dominance = True
                else:
                    stereo_balance = "Balanceado ⚖️"

            return {
                "duration": round(float(duration), 2),
                "sample_rate": int(sr),
                "channels": int(num_channels),
                "is_stereo": bool(is_stereo),
                "channel_energy": channel_energy,
                "channel_db": channel_db,
                "avg_rms": round(float(avg_rms), 6),
                "max_rms": round(float(max_rms), 6),
                "snr_db": round(float(snr_db), 2),
                "papr": round(float(papr), 2),
                "zcr": round(float(zcr), 4),
                "spectral_centroid_hz": round(float(centroid), 1),
                "quality_score": float(quality_score),
                "quality_label": quality_label,
                "noise_level": noise_level,
                "stereo_balance": stereo_balance,
                "left_channel_dominant": bool(left_dominance),
                "right_channel_dominant": bool(right_dominance),
            }

        except Exception as e:
            return {
                "error": str(e),
                "duration": 0,
                "channels": 1,
                "is_stereo": False,
                "quality_score": 0.0,
                "quality_label": "Erro ❌",
                "noise_level": "Erro ❌",
                "snr_db": 0.0,
                "avg_rms": 0.0,
                "max_rms": 0.0,
                "papr": 0.0,
                "stereo_balance": "N/A",
                "sample_rate": 0,
                "channel_energy": [],
                "channel_db": [],
                "zcr": 0.0,
                "spectral_centroid_hz": 0.0,
            }

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calcula o Word Error Rate (WER) entre referência e hipótese.
        Usa jiwer se disponível, senão implementação manual.
        """
        try:
            import jiwer
            return round(float(jiwer.wer(reference, hypothesis)), 4)
        except ImportError:
            pass

        # Fallback: WER manual via distância de edição
        r = reference.lower().split()
        h = hypothesis.lower().split()
        if not r:
            return 0.0

        d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.uint32)
        for i in range(len(r) + 1):
            d[i, 0] = i
        for j in range(len(h) + 1):
            d[0, j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]) + 1

        return round(float(d[len(r), len(h)]) / len(r), 4)

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Formata duração em segundos para mm:ss."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

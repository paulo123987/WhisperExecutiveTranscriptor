import re
import logging

logger = logging.getLogger(__name__)


class PIIRedactor:
    """Redator de PII (Personally Identifiable Information) via Regex e/ou LLM (LGPD)."""

    # Padrões de PII para contexto brasileiro
    PATTERNS = {
        "CPF":     r"\b\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[-\s]?\d{2}\b",
        "CNPJ":    r"\b\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[\/\s]?\d{4}[-\s]?\d{2}\b",
        "RG":      r"\b\d{1,2}[\.\s]?\d{3}[\.\s]?\d{3}[-\s]?[\dXx]\b",
        "Email":   r"[\w\.\+\-]+@[\w\.\-]+\.\w{2,6}",
        "Telefone": r"\(?\d{2}\)?[\s\-]?\d{4,5}[\s\-]?\d{4}",
        "Endereço": r"(Rua|Av\.|Avenida|Praça|Alameda|Travessa|Estrada)\s+[^,]+,\s*\d+",
        "CEP":      r"\b\d{5}[\-\s]?\d{3}\b",
        "Cartão":   r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    }

    def __init__(self, openai_api_key: str = None, groq_api_key: str = None):
        self.openai_api_key = openai_api_key
        self.groq_api_key = groq_api_key
        self._compiled = {label: re.compile(pat, re.IGNORECASE) for label, pat in self.PATTERNS.items()}

    def redact_regex(self, text: str) -> str:
        """Redação rápida via regex — sem IA."""
        redacted = text
        for label, pattern in self._compiled.items():
            redacted = pattern.sub(f"[{label.upper()} REDIGIDO]", redacted)
        return redacted

    def redact_with_llm(self, text: str, provider: str = "openai", model: str = None) -> str:
        """Redação contextual via LLM (OpenAI ou Groq). Fallback para regex."""
        prompt = (
            "Você é um especialista em privacidade de dados (LGPD).\n"
            "Identifique e substitua qualquer PII (nome completo, CPF, CNPJ, RG, e-mail, telefone, endereço, cartão)\n"
            "pelo marcador [PII REDIGIDO]. Mantenha o restante do texto intacto.\n"
            "Retorne APENAS o texto redigido, sem comentários.\n\n"
            f"Texto:\n{text}\n\nTexto Redigido:"
        )

        try:
            if provider == "openai" and self.openai_api_key:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                resp = client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content.strip()

            elif provider == "groq" and self.groq_api_key:
                from groq import Groq
                client = Groq(api_key=self.groq_api_key)
                resp = client.chat.completions.create(
                    model=model or "llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Erro na redação LLM ({provider}/{model}): {e}. Usando regex.")

        return self.redact_regex(text)

    def analyze_with_llm(self, transcription: str, provider: str = "openai", model: str = None) -> dict:
        """
        Analisa a transcrição e extrai insights via LLM.
        Retorna: motivo, sentimento, criticidade, resumo, produto, ação recomendada.
        """
        prompt = (
            "Você é um analista especializado em call center brasileiro.\n"
            "Analise a transcrição abaixo e retorne um JSON puro com as seguintes chaves:\n"
            '  "motivo_ligacao": string — motivo principal da ligação (ex: "Cancelamento de plano")\n'
            '  "sentimento_cliente": string — Positivo | Neutro | Negativo | Muito Negativo\n'
            '  "criticidade": string — Baixa | Média | Alta | Crítica\n'
            '  "resumo": string — resumo em 2 frases do que foi tratado\n'
            '  "produto_servico": string — produto ou serviço mencionado\n'
            '  "acao_recomendada": string — próxima ação recomendada para a empresa\n'
            '  "keywords": lista de strings — até 5 palavras-chave relevantes\n\n'
            "IMPORTANTE: retorne APENAS o JSON puro, sem markdown, sem explicação.\n\n"
            f"Transcrição:\n{transcription[:3000]}\n\nJSON:"
        )

        import json

        def _try_parse(raw: str) -> dict:
            try:
                # Tenta limpar markdown se presente
                clean = raw.strip().strip("```json").strip("```").strip()
                return json.loads(clean)
            except Exception:
                return {
                    "motivo_ligacao": "Não identificado",
                    "sentimento_cliente": "Neutro",
                    "criticidade": "Baixa",
                    "resumo": raw[:300] if raw else "Sem resposta",
                    "produto_servico": "N/A",
                    "acao_recomendada": "Revisar manualmente",
                    "keywords": [],
                }

        try:
            if provider == "openai" and self.openai_api_key:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                resp = client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=512,
                )
                return _try_parse(resp.choices[0].message.content.strip())

            elif provider == "groq" and self.groq_api_key:
                from groq import Groq
                client = Groq(api_key=self.groq_api_key)
                resp = client.chat.completions.create(
                    model=model or "llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=512,
                )
                return _try_parse(resp.choices[0].message.content.strip())

        except Exception as e:
            logger.warning(f"Erro na análise LLM ({provider}/{model}): {e}")

        return {
            "motivo_ligacao": "API não configurada",
            "sentimento_cliente": "Neutro",
            "criticidade": "Baixa",
            "resumo": "Configure uma API key para análise de IA.",
            "produto_servico": "N/A",
            "acao_recomendada": "Configurar API key",
            "keywords": [],
        }

from base.constants import Constants
from base.data_classes import GenerativeModel, LLMResponse
from google import genai
from google.genai import types as genai_types
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from typing import Optional
from utils.llm_logger import LLMLogger
import datetime


class ModelLoader:
    """Utility class for loading various NLP and generative models.

    Attributes:
        None (stateless utility class).
    """

    def load_hf_tokenizer(
        self,
        hugging_face_token: str,
        tokenizer_model: str = Constants.ModelNames.HuggingFace.CROSS_ENCODER_MS_MARCO_MINILM_L_6_V2,
    ) -> AutoTokenizer:
        """Load a pretrained Hugging Face tokenizer.

        Args:
            hugging_face_token (str): Authentication token for private or gated Hugging Face models.
            tokenizer_model (str, optional): Model name or path. Defaults to cross-encoder/ms-marco-MiniLM-L-6-v2.

        Returns:
            AutoTokenizer: Pretrained tokenizer compatible with the specified model.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, token=hugging_face_token)
        return tokenizer

    def load_hf_cross_encoder(
        self,
        hugging_face_token: str,
        cross_encoder_model: str = Constants.ModelNames.HuggingFace.CROSS_ENCODER_MS_MARCO_MINILM_L_6_V2,
        max_length: int = 512,
    ) -> CrossEncoder:
        """Load a pretrained cross-encoder model for semantic reranking.

        Args:
            hugging_face_token (str): Authentication token for private or gated Hugging Face models.
            cross_encoder_model (str, optional): Model name or path. Defaults to cross-encoder/ms-marco-MiniLM-L-6-v2.
            max_length (int, optional): Maximum input length for the model. Defaults to 512.

        Returns:
            CrossEncoder: Reranking model for scoring query–passage pairs.
        """
        cross_encoder = CrossEncoder(
            cross_encoder_model, max_length=max_length, token=hugging_face_token
        )
        return cross_encoder

    def load_sentence_embedding_model(
        self,
        hugging_face_token: str,
        sentence_embedding_model: str = Constants.ModelNames.HuggingFace.SENTENCE_EMBEDDING_MINILM_L6_V2,
    ) -> SentenceTransformer:
        """Load a pretrained SentenceTransformer for text embedding.

        Args:
            hugging_face_token (str): Authentication token for private or gated Hugging Face models.
            sentence_embedding_model (str, optional): Model name or path. Defaults to all-MiniLM-L6-v2.

        Returns:
            SentenceTransformer: Bi-encoder model for embedding sentences or text chunks.
        """
        sentence_transformer = SentenceTransformer(
            sentence_embedding_model, token=hugging_face_token
        )
        return sentence_transformer

    def load_gemini_generative_model(
        self,
        google_api_key: str,
        config=None,
        model_name: str = Constants.ModelNames.Gemini.GEMINI_2_5_PRO,
        llm_logger: Optional[LLMLogger] = None,
    ) -> GenerativeModel:
        """Load a Gemini generative model for text generation.

        Args:
            google_api_key (str): API key for the Google Generative AI service.
            config: Model configuration object (e.g., temperature, max_output_tokens).
            model_name (str, optional): Name of the generative model. Defaults to Gemini 2.5 Pro.
            llm_logger (LLMLogger, optional): Shared logger instance. When provided, every prompt
                and response is appended to the same timestamped JSON file.

        Returns:
            GenerativeModel: Dataclass with ``model_name`` and a ``generate`` callable
                ``(prompt, system_instructions="", response_schema=None, **kwargs) -> LLMResponse``
                that produces a standardized response object.
        """
        client = genai.Client(api_key=google_api_key)

        def generate(
            prompt: str, system_instructions: str = "", response_schema=None, **kwargs
        ) -> LLMResponse:
            date_time = f"[CURRENT DATE AND TIME: {datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')}]\nThis is the real current date. Treat any reference to 'today', 'now', 'current', or 'recent' relative to this date, not your training cutoff.\n\n"

            # Start with explicit config or an empty dictionary
            config_dict = config.__dict__.copy() if config else {}

            if system_instructions:
                config_dict["system_instruction"] = date_time + system_instructions
            else:
                config_dict["system_instruction"] = date_time

            if response_schema:
                config_dict["response_mime_type"] = "application/json"
                config_dict["response_schema"] = response_schema

            # Allow individual override via kwargs (e.g. temperature=0.2)
            config_dict.update(kwargs)

            effective_config = genai_types.GenerateContentConfig(**config_dict)

            full_prompt = prompt
            if llm_logger:
                llm_logger.log_user(full_prompt, system_instructions)

            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=effective_config,
            )

            llm_response = LLMResponse(
                text=response.text,
                parsed=getattr(response, "parsed", None),
                function_calls=response.function_calls or None,
            )
            if llm_logger:
                log_content = response.text or str(
                    [fc.name for fc in (response.function_calls or [])]
                )
                llm_logger.log_agent(log_content, system_instructions)

            return llm_response

        return GenerativeModel(model_name=model_name, generate=generate)

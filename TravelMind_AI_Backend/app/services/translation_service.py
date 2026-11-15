import logging
import asyncio
import re
from typing import Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import Settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class TranslationService:
    """
    Ultra-robust translation service with:
    - Explicit translation instructions
    - Output validation
    - Retry with different prompts on failure
    - Fallback mechanisms
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            temperature=0.1,  # Very low for deterministic translations
            timeout=30,
            max_retries=2,
        )
        self._translation_cache: Dict[str, str] = {}
        self.max_cache_size = 1000

    def _validate_translation(
        self, 
        original: str, 
        translated: str, 
        source_lang: str
    ) -> tuple[bool, str]:
        """
        Validate translation quality.
        Returns (is_valid, error_message)
        """
        # Check 1: Not empty
        if not translated or not translated.strip():
            return False, "Translation is empty"
        
        # Check 2: Not suspiciously short
        if len(translated) < len(original) / 4:
            return False, f"Translation too short ({len(translated)} vs {len(original)} chars)"
        
        # Check 3: Not identical to input (unless already English)
        if source_lang != 'en' and translated.lower() == original.lower():
            return False, "Translation identical to input"
        
        # Check 4: Check if it's still in source language
        if source_lang == 'ro':
            romanian_words = ['este', 'sunt', 'vrei', 'doriți', 'pentru', 'dacă']
            romanian_count = sum(1 for word in romanian_words if word in translated.lower())
            if romanian_count >= 2:
                return False, f"Still contains Romanian words ({romanian_count} found)"
        
        return True, ""

    async def _translate_attempt(
        self,
        text: str,
        source_language: Optional[str],
        attempt_num: int = 1
    ) -> str:
        """
        Single translation attempt with explicit instructions.
        """
        # Build ultra-clear system prompt
        if attempt_num == 1:
            system_content = f"""You are a professional translator.

TASK: Translate the following text from {source_language or 'detected language'} to English.

CRITICAL RULES:
1. Translate the ENTIRE text - do not skip any parts
2. If text contains mixed languages, translate ALL parts to English
3. Preserve meaning, tone, and structure
4. Return ONLY the translated English text
5. NO explanations, NO preambles, NO meta-commentary
6. DO NOT return the original text
7. DO NOT return anything other than the complete English translation

Now translate this text to English:"""
        else:
            # Second attempt with even MORE explicit instructions
            system_content = f"""You are a professional translator tasked with translating text to English.

The text below is in {source_language or 'a foreign language'}. Your job is to provide a complete, accurate English translation.

CRITICAL INSTRUCTIONS:
- Read the ENTIRE text carefully
- Translate EVERY word and sentence to English
- If you see English mixed with other languages, still translate everything to English
- Maintain the original meaning and tone
- Output ONLY the English translation - nothing else
- Do NOT output the original text
- Do NOT output explanations or notes

TEXT TO TRANSLATE:"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=text)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Log token usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            logger.info(
                f"Translation attempt #{attempt_num}: "
                f"Input={usage.get('input_tokens', 0)}, "
                f"Output={usage.get('output_tokens', 0)}"
            )
        
        return response.content.strip()

    async def translate(
        self,
        text: str,
        skip_detection: bool = False,
        source_language: Optional[str] = None
    ) -> str:
        """
        Translate text to English with robust validation.
        
        Args:
            text: Text to translate
            skip_detection: If True, always translate
            source_language: Source language hint (e.g., 'ro', 'fr')
            
        Returns:
            Translated English text
        """
        if not text or not text.strip():
            return text

        # Cache check
        cache_key = str(hash(text.strip().lower()))
        if cache_key in self._translation_cache:
            logger.debug("Returning cached translation")
            return self._translation_cache[cache_key]

        try:
            max_attempts = 3
            last_error = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(
                        f"Translation attempt {attempt}/{max_attempts}: "
                        f"length={len(text)}, source={source_language or 'auto'}"
                    )
                    
                    # Attempt translation
                    translated = await self._translate_attempt(
                        text, 
                        source_language,
                        attempt_num=attempt
                    )
                    
                    # Validate
                    is_valid, error_msg = self._validate_translation(
                        text, 
                        translated,
                        source_language or 'unknown'
                    )
                    
                    if is_valid:
                        logger.info(f"✅ Translation successful (attempt {attempt})")
                        logger.info(f"   Input: '{text[:80]}...'")
                        logger.info(f"   Output: '{translated[:80]}...'")
                        
                        # Cache and return
                        self._add_to_cache(cache_key, translated)
                        return translated
                    else:
                        logger.warning(f"❌ Validation failed (attempt {attempt}): {error_msg}")
                        last_error = error_msg
                        
                        if attempt < max_attempts:
                            logger.info(f"   Retrying with different prompt...")
                            await asyncio.sleep(1)  # Brief pause
                            continue
                
                except Exception as e:
                    logger.error(f"Translation attempt {attempt} failed: {e}")
                    last_error = str(e)
                    
                    if attempt < max_attempts:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # All attempts failed
            logger.error(
                f"All {max_attempts} translation attempts failed. "
                f"Last error: {last_error}"
            )
            logger.error(f"Returning original text as fallback")
            return text
            
        except Exception as e:
            logger.error(f"Translation failed catastrophically: {e}")
            return text

    def _add_to_cache(self, key: str, value: str) -> None:
        """Add to cache with size limit."""
        if len(self._translation_cache) >= self.max_cache_size:
            first_key = next(iter(self._translation_cache))
            del self._translation_cache[first_key]
        self._translation_cache[key] = value

    def clear_cache(self) -> None:
        """Clear cache."""
        self._translation_cache.clear()
        logger.info("Translation cache cleared")
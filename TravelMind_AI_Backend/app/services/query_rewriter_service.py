import openai
import logging
from typing import List, Dict, Tuple
from app.core.config import Settings

logger = logging.getLogger(__name__)


class SmartQueryRewriterService:
    """
    Query Reformulator: Minimal boosting, respects chat history context.
    Only adds clarification when truly needed for semantic search.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o-mini"
        
    async def rewrite_query(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]],
        max_history_turns: int = 5
    ) -> Tuple[str, bool]:
        """
        Reformulate query using conversation context.
        
        Returns: (reformulated_query, was_changed)
        """
        
        # Skip reformulation for simple responses
        if self._is_simple_response(query):
            return query, False
        
        if not conversation_history:
            return query, False
        
        try:
            recent = conversation_history[-(max_history_turns * 2):]
            messages = self._build_prompt(query, recent)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=100
            )
            
            reformulated = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            reformulated = reformulated.strip('"\'')
            
            if not reformulated or len(reformulated) > 300:
                return query, False
            
            if reformulated.lower() == query.lower():
                return query, False
            
            logger.info(f"[REFORMULATE] '{query}' → '{reformulated}'")
            return reformulated, True
                
        except Exception as e:
            logger.error(f"[REFORMULATE] Error: {e}")
            return query, False
    
    def _is_simple_response(self, query: str) -> bool:
        """Check if query is a simple confirmation/response that doesn't need reformulation"""
        query_lower = query.lower().strip()
        
        # Simple confirmations
        simple_responses = {
            'yes', 'yeah', 'yep', 'sure', 'ok', 'okay',
            'no', 'nope', 'nah',
            'thanks', 'thank you', 'thanks!',
            'great', 'perfect', 'awesome',
            'maybe', 'later',
        }
        
        # Check if it's a simple response
        if query_lower in simple_responses:
            return True
        
        # Check if it's very short (likely a confirmation)
        if len(query.split()) <= 2 and len(query) <= 10:
            return True
        
        return False
    
    def _build_prompt(self, query: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        
        system_prompt = """You reformulate user queries for semantic search. Keep it minimal and natural.

RULES:
1. Use chat history to understand context (location, preferences from past turns)
2. Make query standalone by adding missing context from history
3. For weak/ambiguous keywords, add ONE clarifying word maximum
4. Don't over-boost - keep it natural and concise
5. Preserve user's original intent and tone

CHAT HISTORY USAGE:
- If user mentioned location before → include it
- If user has ongoing preferences → carry them forward
- If new topic → reset and use only current query

MINIMAL KEYWORD CLARIFICATION:
Only clarify if the keyword is genuinely ambiguous for semantic search:

Examples of when to clarify (add max 1 word):
- "pool" → "swimming pool" (clarifies it's not a pool hall)
- "breakfast" → "breakfast included" (if context implies included)

Examples of when NOT to clarify:
- "terrace" → keep as "terrace" (already clear)
- "WiFi" → keep as "WiFi" (already clear)

EXAMPLES:

History: ["I'm in Borșa"]
Query: "pool and terrace"
Output: accommodation with swimming pool and terrace in Borșa

History: ["looking for a hotel"]
Query: "with breakfast"
Output: hotel with breakfast included

History: []
Query: "pool and terrace"
Output: accommodation with swimming pool and terrace

NEGATION HANDLING:
If user says "not X", convert to positive:
- "not hot tub" → "swimming pool"

OUTPUT:
- ONE reformulated query
- 5-20 words typically
- Natural language
- No quotes, no repetition"""

        history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content'][:150]}"
            for msg in history if msg.get('content')
        ])
        
        user_prompt = f"""Chat history:
{history_text}

Current query: {query}

Reformulated query:"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
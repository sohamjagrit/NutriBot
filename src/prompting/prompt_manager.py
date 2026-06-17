"""Prompt management and engineering."""

from dataclasses import dataclass
from typing import List, Optional
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SystemPrompt:
    """System prompt template."""
    name: str
    content: str
    few_shot_examples: List[tuple] = None

    def __post_init__(self):
        if self.few_shot_examples is None:
            self.few_shot_examples = []


class PromptManager:
    """Manage and generate prompts for RAG pipeline."""

    # Base system prompts
    SYSTEM_PROMPTS = {
        "standard": SystemPrompt(
            name="standard",
            content="""You are NutriBot, a nutrition assistant that answers STRICTLY from a provided knowledge base.

CRITICAL RULES:
- Answer ONLY using information found in the provided context. Do NOT use outside knowledge.
- Cite the sources you use inline, like [Source 1], [Source 2], next to the claims they support.
- If the context does not contain the information needed to answer, OR the question is not about nutrition, reply EXACTLY with:
  "I'm sorry, that's outside what I can help with. I can only answer nutrition questions using my knowledge base, and I couldn't find relevant information for this. Try asking about foods, nutrients, vitamins, or diet."
- Never invent facts, numbers, or sources. If you are unsure or the context is irrelevant, refuse using the sentence above.
- Recommend consulting a healthcare professional for personal medical concerns.
- Use clear, concise language with bullet points when listing items.""",
        ),
        "conversational": SystemPrompt(
            name="conversational",
            content="""You are a friendly nutrition expert chatbot. Your goal is to help users understand nutrition in a conversational way.

Guidelines:
- Use a warm, approachable tone while maintaining accuracy
- Ask clarifying questions to better understand user needs
- Provide personalized recommendations based on user context
- Use analogies and everyday language to explain nutrition concepts
- Build on previous conversation context to provide continuity""",
        ),
        "detailed": SystemPrompt(
            name="detailed",
            content="""You are a comprehensive nutrition information system. Provide thorough, detailed responses with scientific backing.

Guidelines:
- Include biochemical mechanisms when relevant
- Provide nutritional composition tables when applicable
- Reference research studies and findings
- Discuss both benefits and potential risks
- Organize information hierarchically (summary → details → advanced)""",
        ),
    }

    @classmethod
    def get_system_prompt(cls, prompt_type: str = "standard") -> str:
        """Get a system prompt by type.

        Args:
            prompt_type: Type of prompt (standard, conversational, detailed)

        Returns:
            System prompt content
        """
        if prompt_type not in cls.SYSTEM_PROMPTS:
            logger.warning(f"Unknown prompt type: {prompt_type}, using standard")
            prompt_type = "standard"

        return cls.SYSTEM_PROMPTS[prompt_type].content

    @classmethod
    def build_rag_prompt(
        cls,
        query: str,
        context: str,
        prompt_type: str = "standard",
        few_shot: bool = False
    ) -> str:
        """Build a RAG prompt with context.

        Args:
            query: User query
            context: Retrieved context from documents
            prompt_type: Type of system prompt
            few_shot: Whether to include few-shot examples

        Returns:
            Complete prompt ready for LLM
        """
        prompt_parts = []

        # Few-shot examples (optional)
        if few_shot:
            examples = cls._get_few_shot_examples()
            prompt_parts.append("Examples of good Q&A:")
            for q, a in examples:
                prompt_parts.append(f"\nQ: {q}")
                prompt_parts.append(f"A: {a}")
            prompt_parts.append("\n" + "="*50 + "\n")

        # Context
        prompt_parts.append("Context from nutrition knowledge base:")
        prompt_parts.append(context)
        prompt_parts.append("\n" + "="*50 + "\n")

        # Query
        prompt_parts.append(f"User Question: {query}")
        prompt_parts.append(
            "\nAnswer using ONLY the context above, citing sources inline like [Source 1]. "
            "If the context does not answer the question or the question is not about nutrition, "
            "reply with the exact refusal sentence from your instructions."
        )

        return "\n".join(prompt_parts)

    @staticmethod
    def _get_few_shot_examples() -> List[tuple]:
        """Get few-shot examples for in-context learning.

        Returns:
            List of (question, answer) tuples
        """
        return [
            (
                "What is the daily recommended protein intake?",
                "The recommended dietary allowance (RDA) for protein is 0.8g per kilogram of body weight for sedentary adults. "
                "However, active individuals or athletes may need 1.2-2.0g/kg. For a 70kg adult, this translates to 56g daily for sedentary individuals, "
                "up to 140g for athletes. Always consult with a healthcare provider for personalized recommendations."
            ),
            (
                "How much water should I drink daily?",
                "The general guideline is the '8x8' rule: 8 glasses of 8 ounces (about 2 liters) daily. However, water needs vary based on: "
                "activity level, climate, body size, and overall health. A more personalized approach is to drink enough to keep urine pale yellow. "
                "Athletes and those in hot climates may need significantly more."
            ),
            (
                "Are carbohydrates bad for weight loss?",
                "No, carbohydrates themselves aren't bad for weight loss. What matters is the type and quantity: complex carbs (whole grains, vegetables) "
                "are nutrient-dense and keep you full longer, while refined carbs (white bread, sugary drinks) lack nutrients and cause energy crashes. "
                "The key is portion control and choosing whole grain options when possible."
            ),
        ]

    @classmethod
    def register_system_prompt(cls, name: str, prompt: SystemPrompt) -> None:
        """Register a custom system prompt.

        Args:
            name: Name of the prompt
            prompt: SystemPrompt instance
        """
        cls.SYSTEM_PROMPTS[name] = prompt
        logger.info(f"Registered system prompt: {name}")

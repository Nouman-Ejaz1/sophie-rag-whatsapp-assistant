from typing import Optional, Tuple
from app.brain.research_engine import (
    research_query,
    extract_currency_pair,
    is_exchange_rate_query,
    format_research_result
)

class ResearchService:
    """
    Decoupled service encapsulating multi-source web search, news aggregation,
    and financial query parsers (USD/PKR conversion tracking).
    """
    @staticmethod
    def research(query: str) -> str:
        """Executes combined Jina, DDG, SearxNG, and Google CSE web lookup."""
        return research_query(query)

    @staticmethod
    def parse_currency(query: str) -> Optional[Tuple[str, str, float]]:
        """Extracts target currency tokens and numbers from natural Urdu/English queries."""
        return extract_currency_pair(query)

    @staticmethod
    def is_rate_query(query: str) -> bool:
        """Determines if the prompt demands currency conversion information."""
        return is_exchange_rate_query(query)

research_service = ResearchService()

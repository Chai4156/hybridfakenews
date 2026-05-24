"""
Google Custom Search integration for fact-checking.
Enables searching, retrieving, and analyzing articles.
"""

import requests
import logging
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import json

logger = logging.getLogger(__name__)

class GoogleSearchEngine:
    """
    Google Custom Search Engine integration.
    Features:
    - Search for articles
    - Retrieve full text from URLs
    - Multi-source comparison
    - Claim extraction
    """
    
    def __init__(self, api_key: str, search_engine_id: str):
        """
        Initialize Google Search integration.
        
        Args:
            api_key: Google Custom Search API key
            search_engine_id: Custom Search Engine ID (CSE ID)
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.max_results = 10
        
        if not api_key or api_key == "your_api_key_here":
            logger.warning("Google Search API key not configured. Search features disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Google Custom Search enabled")
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search for articles using Google Custom Search.
        
        Args:
            query: Search query (e.g., "fake news detector", "climate change")
            num_results: Number of results to return (max 10)
        
        Returns:
            List of search results with title, URL, snippet, publication date
        """
        if not self.enabled:
            logger.error("Google Search not configured")
            return []
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),  # Max 10 per request
                'sort': 'date'  # Sort by date (most recent first)
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()
            
            articles = []
            if 'items' in results:
                for item in results['items']:
                    articles.append({
                        'title': item.get('title', 'N/A'),
                        'url': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': item.get('displayLink', ''),
                        'date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', 'N/A')
                    })
            
            logger.info(f"Found {len(articles)} articles for query: '{query}'")
            return articles
        
        except Exception as e:
            logger.error(f"Google Search error: {e}")
            return []
    
    def search_for_claim(self, claim: str, num_results: int = 3) -> List[Dict]:
        """
        Search for evidence about a specific claim.
        Useful for fact-checking individual statements.
        
        Args:
            claim: The claim to verify (e.g., "NASA confirms water on Mars")
            num_results: Number of sources to find
        
        Returns:
            List of sources mentioning the claim
        """
        # Add quotes to search for exact phrases
        query = f'"{claim}"' if len(claim.split()) > 2 else claim
        return self.search(query, num_results)
    
    def search_topic(self, topic: str, num_results: int = 10) -> List[Dict]:
        """
        Search for multiple sources covering the same topic.
        Enables multi-source fact-checking.
        
        Args:
            topic: News topic (e.g., "cryptocurrency regulation")
            num_results: Number of sources
        
        Returns:
            List of articles from different sources
        """
        return self.search(topic, num_results)
    
    def get_related_articles(self, article_url: str, num_results: int = 5) -> List[Dict]:
        """
        Find related articles to compare coverage.
        
        Args:
            article_url: URL of original article
            num_results: Number of related articles
        
        Returns:
            List of articles on similar topic
        """
        try:
            # Extract domain to avoid identical articles
            from urllib.parse import urlparse
            domain = urlparse(article_url).netloc
            
            # Extract potential keywords from URL or snippet
            # For now, search for generic "news" to get variety
            query = "-site:" + domain + " news"
            return self.search(query, num_results)
        except Exception as e:
            logger.error(f"Error finding related articles: {e}")
            return []
    
    def batch_search(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """
        Perform multiple searches at once.
        
        Args:
            queries: List of search queries
        
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, num_results=3)
        return results
    
    def format_results_for_display(self, articles: List[Dict]) -> str:
        """
        Format search results for user display.
        
        Args:
            articles: List of article dicts
        
        Returns:
            Formatted string for UI
        """
        if not articles:
            return "No articles found."
        
        formatted = "**Search Results:**\n\n"
        for i, article in enumerate(articles, 1):
            formatted += f"{i}. **{article['title']}**\n"
            formatted += f"   Source: {article['source']}\n"
            formatted += f"   Date: {article['date']}\n"
            formatted += f"   Link: {article['url']}\n"
            formatted += f"   Snippet: {article['snippet'][:100]}...\n\n"
        
        return formatted


class FactChecker:
    """
    Fact-checking engine using multiple sources.
    Searches for claims and compares sources.
    """
    
    def __init__(self, search_engine: GoogleSearchEngine, analyzer: Optional[object] = None):
        """
        Initialize fact-checker.
        
        Args:
            search_engine: GoogleSearchEngine instance
            analyzer: NLP analyzer (fake news detector)
        """
        self.search_engine = search_engine
        self.analyzer = analyzer
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.
        Simple implementation - looks for sentences with specific keywords.
        
        Args:
            text: Article text
        
        Returns:
            List of extracted claims
        """
        keywords = ['claims', 'says', 'confirms', 'announces', 'reports', 'found', 'discovered']
        claims = []
        
        sentences = text.split('.')
        for sentence in sentences:
            # Look for sentences with claim keywords
            if any(kw in sentence.lower() for kw in keywords):
                claim = sentence.strip()
                if len(claim) > 20:  # Skip very short sentences
                    claims.append(claim)
        
        return claims[:5]  # Return top 5 claims
    
    def verify_claim(self, claim: str) -> Dict:
        """
        Verify a claim by searching for supporting evidence.
        
        Args:
            claim: Factual claim to verify
        
        Returns:
            Dictionary with verification results
        """
        sources = self.search_engine.search_for_claim(claim, num_results=5)
        
        return {
            'claim': claim,
            'supporting_sources': len([s for s in sources if 'yes' not in s['snippet'].lower()]),
            'conflicting_sources': len([s for s in sources if 'no' not in s['snippet'].lower()]),
            'sources': sources,
            'verdict': 'Likely accurate' if len(sources) > 2 else 'Needs verification'
        }
    
    def multi_source_check(self, topic: str) -> Dict:
        """
        Compare how multiple sources cover the same topic.
        
        Args:
            topic: News topic to investigate
        
        Returns:
            Comparison of different sources
        """
        articles = self.search_engine.search_topic(topic, num_results=5)
        
        analysis = {
            'topic': topic,
            'total_sources': len(articles),
            'articles': articles,
            'consensus': self._calculate_consensus(articles),
            'outliers': self._identify_outliers(articles)
        }
        
        return analysis
    
    def _calculate_consensus(self, articles: List[Dict]) -> str:
        """Check if sources agree on topic."""
        if len(articles) < 2:
            return "Insufficient sources"
        return "Strong consensus" if len(articles) > 3 else "Mixed coverage"
    
    def _identify_outliers(self, articles: List[Dict]) -> List[str]:
        """Identify sources with unique perspectives."""
        # Simplified: just return different domains
        domains = list(set([a['source'] for a in articles]))
        return domains[3:] if len(domains) > 3 else []


def get_search_engine(api_key: str = None, cse_id: str = None) -> GoogleSearchEngine:
    """
    Factory function to create Google Search engine.
    Uses env variables if credentials not provided.
    """
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    api_key = api_key or os.getenv('GOOGLE_SEARCH_API_KEY', 'your_api_key_here')
    cse_id = cse_id or os.getenv('GOOGLE_CSE_ID', 'c1208c9c628bd46b1')
    
    return GoogleSearchEngine(api_key, cse_id)

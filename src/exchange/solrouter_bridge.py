import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SolRouterBridge:
    """
    Python client for securely querying the local Node.js SolRouter Express API.
    Provides real-time private sentiment summarization before execution.
    """
    def __init__(self, host: str = "http://localhost:3000"):
        self.host = host
        self.endpoint = f"{self.host}/api/analyze"
        
    def analyze_market_context(self, query: str, context_label: str = "BTC 5M") -> Optional[Dict[str, Any]]:
        """
        Sends an analysis query to the private gpt-oss-20b model via the local SolRouter bridge.
        
        Args:
            query: The prompt directing the AI on what to search and evaluate.
            context_label: Informational metadata string for Node.js logging.
            
        Returns:
            Dictionary containing the plain-text decoded response summary and the privacy attestation ID,
            or None if the request failed.
        """
        payload = {
            "query": query,
            "marketContext": context_label
        }
        
        try:
            logger.info("Requesting private analysis via local SolRouter bridge...")
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if data.get("success"):
                logger.info("Successfully received encrypted response from SolRouter.")
                return {
                    "summary": data.get("sentimentSummary"),
                    "attestation": data.get("attestation")
                }
            else:
                logger.error("SolRouter bridge returned internal error: %s", data.get("error"))
                return None
                
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to the SolRouter bridge. Make sure 'node server.js' is running.")
            return None
        except requests.exceptions.RequestException as e:
            logger.error("SolRouter bridge request failed: %s", e)
            return None

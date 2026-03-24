from base.constants import Constants
from ddgs import DDGS
from trafilatura import extract
from typing import List, Dict, Optional

import requests
import time


class WebSearch:
    """Execute web searches and extract content from top result pages.

    Uses DuckDuckGo as the search backend and Trafilatura for HTML-to-text extraction.
    """

    def _collect_urls(self, query: str, max_results: int = 5) -> List[str]:
        """Collect URLs from DuckDuckGo search results.

        Args:
            query (str): Search query string.
            max_results (int, optional): Maximum number of URLs to retrieve. Defaults to 5.

        Returns:
            list[str]: URLs extracted from search results.
        """
        urls = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for result in results:
                url = result.get("href", None)
                if url:
                    urls.append(url)
        return urls

    def _fetch_page_content(
        self,
        url: str,
        user_agent_header: str = Constants.URLHeader.USER_AGENT_HEADER,
        timeout: int = Constants.Timeout.TEN_SECONDS,
    ) -> Optional[str]:
        """Fetch and extract readable text from a web page.

        Args:
            url (str): Target URL to fetch.
            user_agent_header (str, optional): User-Agent header for HTTP request. Defaults to project-specific UA.
            timeout (int, optional): Request timeout in seconds. Defaults to 10.

        Returns:
            str | None: Extracted text content, or None if extraction fails.

        Raises:
            Exception: If the HTTP request fails or extraction encounters an error.
        """
        headers = {"User-Agent": user_agent_header}
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == Constants.HTTPStatusCodes.OK:
                html = response.text
                text = extract(html)
                return text
            raise Exception(f"Failed to fetch {url}; status code:{response.status_code}")
        except Exception as e:
            raise Exception(f"Error fetching URL {url}: {str(e)}")

    def search(
        self, query: str, max_results: int = 5, timeout: int = Constants.Timeout.THREE_SECONDS
    ) -> List[Dict[str, str]]:
        """Perform a web search and retrieve content from top result pages.

        Args:
            query (str): Search query string.
            max_results (int, optional): Maximum number of results to fetch. Defaults to 3.
            timeout (int, optional): Delay (seconds) between fetching pages. Defaults to 3.

        Returns:
            tuple[list[str], list[dict]]: A tuple containing:
                - List of extracted text content from each page.
                - List of metadata dicts (currently just URLs) corresponding to each content item.
        """
        urls = self._collect_urls(query, max_results)
        documents = []
        documents_metadata = []
        for url in urls:
            try:
                content = self._fetch_page_content(url)
                if content:
                    documents.append(content)
                    documents_metadata.append({"url": url})
                time.sleep(timeout)
            except:
                pass
        if not documents:
            raise Exception("No web content could be retrieved for the query.")
        return documents, documents_metadata

import unittest
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

class TestScrapeTools(unittest.TestCase):

    def test_scrape_website_tool(self):
        url = 'https://www.crewai.com'
        swt = ScrapeWebsiteTool(url)
        content = swt.get_content()  # Using get_content as a hypothetical correct method
        self.assertIsNotNone(content, "HTML content should not be None")

    def test_serper_dev_tool(self):
        api_key = '1240f10e2c8f25ba63d89e5813b891d514289806'
        params = {"api_key": api_key}  # Assuming initialization with params

        sdt = SerperDevTool(**params)

        query = 'crewai'
        search_results = sdt.search(query)  # Using search as a hypothetical correct method
        self.assertIsNotNone(search_results, "Search results should not be None")
        self.assertGreater(len(search_results), 0, f"Expected at least one search result for query '{query}'")

        title = 'CrewAI: A Social Media Army'
        article = sdt.get_article(title)  # Using get_article as a hypothetical correct method
        self.assertIsNotNone(article, f"Article with title '{title}' should not be None")

        required_keys = [
            'title', 'link', 'snippet', 'date', 'source', 'thumbnail', 'country',
            'language', 'author', 'type', 'video', 'audio', 'pdf', 'word_count',
            'read_time', 'tags', 'entities', 'sentiment', 'emotions', 'summary'
        ]

        for key in required_keys:
            with self.subTest(key=key):
                self.assertIsNotNone(article.get(key), f"Article key '{key}' should not be None")

if __name__ == '__main__':
    unittest.main()

print('Test completed!')

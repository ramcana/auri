import requests
import os

def get_latest_news(query=None, country='us', max_results=5):
    """
    Fetch latest news headlines from NewsAPI.org.
    If query is provided, search for that topic; otherwise, get top headlines.
    """
    api_key = os.environ.get('NEWS_API_KEY')
    if not api_key:
        return [{"title": "Error", "summary": "NEWS_API_KEY not set in environment."}]
    base_url = 'https://newsapi.org/v2/'
    headers = {'Authorization': api_key}
    params = {'pageSize': max_results, 'language': 'en'}
    if query:
        url = base_url + 'everything'
        params['q'] = query
        # Optionally: params['sortBy'] = 'publishedAt'
    else:
        url = base_url + 'top-headlines'
        params['country'] = country
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get('articles', [])
        results = []
        for article in articles:
            title = article.get('title', 'Untitled')
            summary = article.get('description') or article.get('content') or ''
            results.append({"title": title, "summary": summary})
        if not results:
            return [{"title": "No news found", "summary": f"No articles found for query: {query}"}]
        return results
    except Exception as e:
        return [{"title": "Error", "summary": str(e)}]

# Google Custom Search Engine Integration

## Overview

Your fake news detector now includes powerful search and fact-checking capabilities powered by Google Custom Search Engine (CSE) and your custom search engine "factsearch".

### What You Can Do

| Feature | Use Case |
|---------|----------|
| **Search** | Find articles on any topic using Google CSE |
| **Search & Analyze** | Find articles and instantly analyze them for fake news |
| **Verify Claims** | Check specific factual claims against multiple sources |
| **Multi-Source Check** | Compare how different news outlets cover the same topic |

---

## Setup

### 1. Add Your Google API Key

**Get your API key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "Custom Search API"
4. Create an API key (type: Unrestricted)
5. Copy the key

**Add to .env file:**
```bash
GOOGLE_SEARCH_API_KEY=YOUR_API_KEY_HERE
GOOGLE_CSE_ID=c1208c9c628bd46b1
```

### 2. Restart Flask

```bash
cd backend
python app.py
```

Watch for this log message:
```
Google Custom Search enabled
```

---

## API Endpoints

### 1. Search Articles

**Endpoint:** `POST /api/search`

**Request:**
```json
{
  "query": "fake news detection",
  "num_results": 5
}
```

**Response:**
```json
{
  "query": "fake news detection",
  "count": 5,
  "status": "success",
  "results": [
    {
      "title": "Article Title",
      "url": "https://...",
      "source": "example.com",
      "snippet": "Article preview text...",
      "date": "2026-05-24T00:00:00Z"
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "climate change", "num_results": 5}'
```

---

### 2. Search & Analyze

**Endpoint:** `POST /api/search-and-analyze`

Searches for articles AND analyzes each one for fake news in real-time.

**Request:**
```json
{
  "query": "artificial intelligence regulation",
  "num_results": 3
}
```

**Response:**
```json
{
  "query": "artificial intelligence regulation",
  "count": 3,
  "real_count": 2,
  "fake_count": 1,
  "results": [
    {
      "title": "Article Title",
      "url": "https://...",
      "source": "example.com",
      "snippet": "...",
      "label": "Real",
      "confidence": 78.5,
      "text_length": 2500
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/search-and-analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "cryptocurrency fraud", "num_results": 3}'
```

---

### 3. Verify Claim

**Endpoint:** `POST /api/verify-claim`

Searches for evidence supporting or contradicting a specific claim.

**Request:**
```json
{
  "claim": "NASA discovered water on Mars"
}
```

**Response:**
```json
{
  "claim": "NASA discovered water on Mars",
  "supporting_sources": 8,
  "conflicting_sources": 0,
  "verdict": "Likely accurate",
  "sources": [
    {
      "title": "Source Article",
      "url": "https://...",
      "source": "nasa.gov",
      "snippet": "NASA confirms..."
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/verify-claim \
  -H "Content-Type: application/json" \
  -d '{"claim": "The moon landing was a hoax"}'
```

---

### 4. Multi-Source Check

**Endpoint:** `POST /api/multi-source-check`

Compares how different sources cover the same topic to identify consensus and outliers.

**Request:**
```json
{
  "topic": "COVID-19 vaccines effectiveness"
}
```

**Response:**
```json
{
  "topic": "COVID-19 vaccines effectiveness",
  "total_sources": 5,
  "consensus": "Strong consensus",
  "outliers": ["alternative-health.com"],
  "articles": [
    {
      "title": "Title",
      "url": "https://...",
      "source": "bbc.com",
      "snippet": "..."
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/multi-source-check \
  -H "Content-Type: application/json" \
  -d '{"topic": "election fraud 2024"}'
```

---

## Python Integration

### Use in Your Code

```python
from google_search import get_search_engine, FactChecker
from ensemble_model import EnsembleModel

# Initialize
search_engine = get_search_engine()
fact_checker = FactChecker(search_engine)

# Search
articles = search_engine.search("climate change", num_results=5)
for article in articles:
    print(f"{article['title']} - {article['source']}")

# Verify a claim
result = fact_checker.verify_claim("Bitcoin is backed by government")
print(f"Verdict: {result['verdict']}")
print(f"Supporting sources: {result['supporting_sources']}")

# Compare sources
analysis = fact_checker.multi_source_check("AI regulation")
print(f"Consensus: {analysis['consensus']}")
print(f"Outliers: {analysis['outliers']}")
```

---

## How It Works

### Architecture

```
User Request
    ↓
Google Custom Search API
    ↓
Extract Articles (title, URL, snippet)
    ↓
[For Search & Analyze]
    ├─ Fetch each article URL
    ├─ Extract text with BeautifulSoup
    ├─ Run through NLP pipeline
    └─ Return predictions
    ↓
Response to User
```

### Search Quality Settings

In your Google CSE dashboard, you can configure:

- **Search Scope**: Only news sites (recommended)
- **Safe Search**: Enable to filter explicit content
- **Country**: Limit to specific regions
- **Language**: English, etc.

**Current Settings:**
- CSE ID: `c1208c9c628bd46b1`
- Search Name: `factsearch`
- Scope: Global (all sites)

---

## Best Practices

### For Fact-Checking

1. **Use Claim Verification** for specific statements
2. **Use Multi-Source Check** for major news events
3. **Cross-reference** results with at least 3 sources
4. **Check publication dates** - more recent = more relevant

### API Rate Limits

**Google Custom Search:**
- Free: 100 queries/day
- Paid: Up to 10,000 queries/day

**DataImpulse Proxy:**
- Uses residential IPs (harder to block)
- IP rotation prevents detection

### Performance Tips

- Limit `num_results` to 3-5 for faster processing
- Use direct text input when you have content ready
- Batch searches using `/api/search` then `/api/predict`

---

## Troubleshooting

### Issue: "Google Search not configured"

**Solution:**
1. Check `.env` file has `GOOGLE_SEARCH_API_KEY=YOUR_KEY`
2. Verify API key is valid and has "Custom Search API" enabled
3. Restart Flask

### Issue: No search results returned

**Causes:**
1. API key not valid or rate limit exceeded
2. Query too specific
3. CSE not configured to search those sites

**Solution:**
- Try a simpler search term
- Check Google Cloud Console for API status
- Verify CSE settings in dashboard

### Issue: Articles fail to fetch

**Causes:**
1. URL is blocked
2. Site requires authentication
3. DataImpulse proxy not working

**Solution:**
- URL fallback to direct connection is automatic
- DataImpulse proxy provides residential IPs
- Some sites simply cannot be scraped (it's blocked)

---

## Advanced Usage

### Extract Claims from Articles

```python
from google_search import FactChecker

fact_checker = FactChecker(search_engine)

# Get article text
article_text = "NASA announced water on Mars..."

# Extract factual claims
claims = fact_checker.extract_claims(article_text)

# Verify each claim
for claim in claims:
    result = fact_checker.verify_claim(claim)
    print(f"Claim: {claim}")
    print(f"Verdict: {result['verdict']}")
```

### Batch Analysis

```python
topics = ["climate change", "cryptocurrency", "AI safety"]

for topic in topics:
    analysis = fact_checker.multi_source_check(topic)
    print(f"{topic}: {analysis['consensus']}")
```

---

## What's Included

**New Files:**
- `backend/google_search.py` - Google CSE integration, fact-checker
- Updated `.env` - Google API credentials
- Updated `backend/app.py` - 4 new API endpoints

**New Classes:**
- `GoogleSearchEngine` - Search functionality
- `FactChecker` - Claim verification and multi-source analysis

**New Endpoints:**
- `/api/search` - Find articles
- `/api/search-and-analyze` - Search + fake news detection
- `/api/verify-claim` - Check factual claims
- `/api/multi-source-check` - Compare sources

---

## Security Reminders

⚠️ **DO NOT:**
- Commit `.env` to git (contains API keys)
- Share your API key publicly
- Use fake data to train models

✅ **DO:**
- Keep API keys in `.env` only
- Monitor API usage in Google Cloud Console
- Use legitimate sources for verification

---

## Next Steps

1. **Get API Key:**
   - https://developers.google.com/custom-search/v1/overview

2. **Test Endpoints:**
   - Use curl or Postman to test
   - Check `backend/prediction.log` for debugging

3. **Integrate Frontend:**
   - Add search UI component to React
   - Call `/api/search-and-analyze` endpoint

4. **Monitor Usage:**
   - Google Cloud Console: API quotas
   - `.env`: API keys secure

---

## Support

- **Google Custom Search Docs:** https://developers.google.com/custom-search
- **CSE Control Panel:** https://cse.google.com/cse
- **Flask API Logs:** `backend/prediction.log`

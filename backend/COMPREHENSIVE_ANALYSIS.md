# Comprehensive Analysis Endpoint

## Overview

The `/api/comprehensive-analysis` endpoint implements your complete fake news detection workflow in a single unified request.

## What It Does (5-Step Pipeline)

```
1. Extract & Clean Input
        ↓
2. Run Ensemble Prediction (RoBERTa + SBERT + NB)
        ↓
3. Automatically Search for Related Articles
        ↓
4. Analyze Related Articles for Verification
        ↓
5. Generate Comprehensive Reasoning
        ↓
Return unified result with prediction + verification
```

## API Endpoint

**URL:** `POST /api/comprehensive-analysis`

**Purpose:** Complete fake news analysis with automatic source verification

## Request Format

```json
{
  "input_type": "text",
  "input_value": "Article text or URL content",
  "search_results": 5
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_type` | string | Yes | - | "text" or "url" |
| `input_value` | string | Yes | - | Article text (if text) or URL (if url) |
| `search_results` | integer | No | 5 | Number of related articles to find |

## Response Format

```json
{
  "status": "success",
  "input_type": "text",
  "source_url": null,
  
  "prediction": {
    "label": "Real",
    "confidence": 82.5,
    "reasoning": "Text shows neutral reporting with cited sources. Cross-verification with 3 related articles shows 100% consensus as real content.",
    "chunks_analyzed": 2
  },
  
  "verification": {
    "related_articles_found": 5,
    "articles_analyzed": 3,
    "real_count": 3,
    "fake_count": 0,
    "consensus": "Real",
    "articles": [
      {
        "title": "Climate change: 2024 hottest year on record",
        "source": "bbc.com",
        "label": "Real",
        "confidence": 89.2
      }
    ]
  },
  
  "model": "ensemble (RoBERTa 0.8 + SBERT-NB 0.2)",
  "text_preview": "Article text preview...",
  "processing_steps": 5
}
```

## How It Works

### Step 1: Extract & Clean Text
- If URL input: Fetch content with BeautifulSoup + proxy
- If text input: Use provided text
- Clean: Remove noise, tokenize, lemmatize, remove stopwords

### Step 2: Run Initial Ensemble Prediction
- Split into 256-token chunks
- Run RoBERTa (80% weight) inference
- Run SBERT embeddings + Naive Bayes (20% weight)
- Soft vote: 0.8×RoBERTa + 0.2×SBERT-NB
- Get confidence score

### Step 3: Automatically Search
- Extract first 100 chars as search query
- Call Google Custom Search API
- Return top N articles (default 5)

### Step 4: Verify Against Sources
- For each related article:
  - Fetch URL content
  - Clean & chunk text
  - Run ensemble prediction
  - Store label + confidence
- Count real vs fake sources
- Calculate consensus

### Step 5: Generate Reasoning
- Analyze initial prediction confidence
- Add verification context
- Show how related sources align
- Explain potential discrepancies

## Usage Examples

### Example 1: Analyze Text (Your Original Use Case)

**Request:**
```bash
curl -X POST http://localhost:5000/api/comprehensive-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "text",
    "input_value": "NASA announced discovery of water on Mars. The space agency confirmed presence of H2O molecules in soil samples collected by rovers.",
    "search_results": 5
  }'
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "label": "Real",
    "confidence": 88.3,
    "reasoning": "Text demonstrates factual reporting with specific citations. Cross-verification with 3 related articles shows 100% consensus as real content. Strong evidence from official sources (nasa.gov, science journals)."
  },
  "verification": {
    "related_articles_found": 5,
    "articles_analyzed": 3,
    "real_count": 3,
    "fake_count": 0,
    "consensus": "Real"
  }
}
```

---

### Example 2: Analyze URL

**Request:**
```bash
curl -X POST http://localhost:5000/api/comprehensive-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "url",
    "input_value": "https://www.bbc.com/news/science_environment/article",
    "search_results": 5
  }'
```

**Response:**
```json
{
  "status": "success",
  "input_type": "url",
  "source_url": "https://www.bbc.com/news/science_environment/article",
  "prediction": {
    "label": "Real",
    "confidence": 85.7
  },
  "verification": {
    "related_articles_found": 5,
    "articles_analyzed": 3,
    "real_count": 3,
    "consensus": "Real"
  }
}
```

---

### Example 3: Suspicious Content

**Request:**
```bash
curl -X POST http://localhost:5000/api/comprehensive-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "text",
    "input_value": "The government is hiding UFO alien bases in Area 51. No official sources confirm this because they are part of the coverup."
  }'
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "label": "Fake",
    "confidence": 73.2,
    "reasoning": "Text uses conspiracy language and makes unverified claims without credible sources. Cross-verification with 3 related articles shows mixed signals (2 real, 1 fake/speculation). Contradicts consensus from official sources."
  },
  "verification": {
    "related_articles_found": 5,
    "articles_analyzed": 3,
    "real_count": 2,
    "fake_count": 1,
    "consensus": "Mixed/Fake"
  }
}
```

---

## How It Matches Your Original Idea

| Your Idea | Implementation |
|-----------|-----------------|
| Text/link input | ✅ `input_type` and `input_value` parameters |
| Process & analyze content | ✅ Step 1: Extract & clean |
| Run searches for similar articles | ✅ Step 3: Google CSE search |
| Verify facts using results | ✅ Step 4: Analyze related articles |
| Feed to RoBERTa + SBERT + NB | ✅ Step 2 & 4: Ensemble inference |
| Return prediction + confidence | ✅ Unified response with label + % confidence |
| Single comprehensive result | ✅ One endpoint, one response |

---

## Response Fields Explained

### prediction
```json
{
  "label": "Real|Fake",           // Binary classification
  "confidence": 85.7,              // Percentage (0-100)
  "reasoning": "...",              // Human-readable explanation
  "chunks_analyzed": 2             // Number of text chunks
}
```

### verification
```json
{
  "related_articles_found": 5,     // Total articles from search
  "articles_analyzed": 3,          // Analyzed (faster, top 3 only)
  "real_count": 3,                 // Real articles found
  "fake_count": 0,                 // Fake articles found
  "consensus": "Real|Mixed|Fake",  // Overall consensus
  "articles": [...]                // Article details
}
```

---

## Performance Notes

**Time Breakdown (Typical):**
- Step 1 (Extract): 1-3s
- Step 2 (Predict): 2-4s
- Step 3 (Search): 1-2s
- Step 4 (Verify): 5-10s (analyzes 3 articles)
- Step 5 (Reasoning): <1s

**Total: ~10-20 seconds** (mostly URL fetching)

---

## Error Handling

### Bad Request (400)
```json
{
  "error": "No JSON body provided"
}
```

### URL Not Accessible (502)
```json
{
  "error": "Failed to fetch URL: Connection timeout"
}
```

### Processing Error (500)
```json
{
  "error": "Analysis failed: Invalid text format"
}
```

---

## When to Use

### Use `/api/comprehensive-analysis` when:
- ✅ User provides single article for complete analysis
- ✅ You need prediction + verification together
- ✅ User wants one-click analysis (simple UI)
- ✅ You need context-aware reasoning

### Use individual endpoints when:
- ✅ User wants to search without analyzing
- ✅ User wants to verify specific claims only
- ✅ You're building advanced features
- ✅ You need fine-grained control

---

## Integration with Frontend

**React Component Example:**

```jsx
async function analyzeArticle(text) {
  const response = await fetch('http://localhost:5000/api/comprehensive-analysis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input_type: 'text',
      input_value: text,
      search_results: 5
    })
  });
  
  const result = await response.json();
  
  return {
    prediction: result.prediction.label,
    confidence: result.prediction.confidence,
    reasoning: result.prediction.reasoning,
    verification: result.verification,
    sources: result.verification.articles
  };
}
```

---

## Summary

The **comprehensive-analysis** endpoint is the primary endpoint you requested - it implements your complete workflow:

```
Input → Process → Search → Verify → Predict → Report
```

All in one unified response with:
- Initial prediction (RoBERTa + SBERT + NB)
- Related articles (Google CSE)
- Source verification (Article analysis)
- Comprehensive reasoning (All data combined)
- Confidence scores (Percentage-based)

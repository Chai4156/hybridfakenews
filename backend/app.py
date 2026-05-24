from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse
import numpy as np
from ensemble_model import EnsembleModel
from clean_news import clean_text
from vectorizer import chunk_text
from google_search import get_search_engine, FactChecker
try:
    from config import PROXY_ENABLED, DATAIMPULSE_LOGIN, DATAIMPULSE_PASSWORD, DATAIMPULSE_HOST, DATAIMPULSE_PORT
except ImportError:
    PROXY_ENABLED = False

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
cors_origins = os.getenv('CORS_ORIGINS')
if cors_origins:
    origins = [origin.strip() for origin in cors_origins.split(',') if origin.strip()]
    CORS(app, origins=origins)
else:
    CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load ensemble model
ensemble = EnsembleModel()
ensemble.load_models()

# Initialize Google Search Engine
search_engine = get_search_engine()
fact_checker = FactChecker(search_engine, ensemble)


def extract_text_from_url(url):
    is_safe, reason = validate_fetch_url(url)
    if not is_safe:
        raise ValueError(reason)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    # Setup proxy if enabled
    proxies = None
    if PROXY_ENABLED:
        proxy_url = f'http://{DATAIMPULSE_LOGIN}:{DATAIMPULSE_PASSWORD}@{DATAIMPULSE_HOST}:{DATAIMPULSE_PORT}'
        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f'Using DataImpulse proxy: {DATAIMPULSE_HOST}:{DATAIMPULSE_PORT}')
    
    # Setup session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    try:
        response = session.get(url, timeout=20, headers=headers, proxies=proxies, allow_redirects=True)
        response.raise_for_status()
        logger.info(f'Successfully fetched URL: {url}')
    except Exception as e:
        logger.error(f"Request failed: {e}")
        if proxies:
            logger.warning('Proxy request failed, retrying without proxy...')
            try:
                response = session.get(url, timeout=20, headers=headers, allow_redirects=True)
                response.raise_for_status()
                logger.info(f'Successfully fetched URL without proxy: {url}')
            except Exception as e2:
                logger.error(f"Fallback request also failed: {e2}")
                raise
        else:
            raise
    
    soup = BeautifulSoup(response.text, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Try to get paragraphs first, then fallback to all text
    paragraphs = soup.find_all('p')
    if paragraphs:
        text = ' '.join(p.get_text(strip=True) for p in paragraphs)
    else:
        # Fallback: get all text from body
        body = soup.find('body')
        text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
    
    return text.strip() if text else "No content extracted"


def _is_disallowed_ip(ip) -> bool:
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast


def _host_resolves_to_private(hostname: str):
    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True, 'Host could not be resolved'
    for info in addrinfo:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if _is_disallowed_ip(ip):
            return True, 'URL resolves to a local or private address'
    return False, None


def validate_fetch_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False, 'Only http/https URLs are allowed'
    if not parsed.hostname:
        return False, 'Invalid URL'

    hostname = parsed.hostname.lower()
    if hostname == 'localhost' or hostname.endswith('.local'):
        return False, 'Local URLs are not allowed'

    try:
        ip = ipaddress.ip_address(hostname)
        if _is_disallowed_ip(ip):
            return False, 'Local or private IPs are not allowed'
    except ValueError:
        is_private, reason = _host_resolves_to_private(hostname)
        if is_private:
            return False, reason

    return True, None


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    input_type = data.get('input_type', 'text')
    input_value = data.get('input_value', '').strip()

    if not input_value:
        return jsonify({'error': 'Input cannot be empty'}), 400

    try:
        if input_type == 'url':
            logger.info(f"Fetching URL: {input_value}")
            raw_text = extract_text_from_url(input_value)
            if not raw_text:
                return jsonify({'error': 'No readable text found at URL'}), 422
        else:
            raw_text = input_value

        # Step 1: Clean text (remove noise, lowercase, tokenize, lemmatize)
        logger.info("Step 1: Cleaning text...")
        cleaned_text = clean_text(raw_text)
        
        if not cleaned_text:
            return jsonify({'error': 'Text cleaning resulted in empty content'}), 422

        # Step 2: Handle long texts with chunking
        logger.info("Step 2: Processing text chunks...")
        chunks = chunk_text(cleaned_text, max_tokens=256)
        logger.info(f"Text split into {len(chunks)} chunk(s)")
        
        # Step 3: Get ensemble predictions for all chunks
        logger.info("Step 3: Running ensemble model inference...")
        predictions, confidences, probs = ensemble.predict(chunks, batch_size=1)
        
        # Aggregate predictions: use average confidence and majority vote
        final_prediction = int(sum(predictions) > len(predictions) / 2)  # Majority vote
        final_confidence = np.mean(confidences)
        final_probs = np.mean(probs, axis=0)  # Average probabilities
        
        label = 'Real' if final_prediction == 1 else 'Fake'
        logger.info(f"Ensemble Prediction: {label} ({final_confidence:.4f}) for input type: {input_type}")

        # Step 4: Generate reasoning
        logger.info("Step 4: Generating prediction reasoning...")
        reasoning = ensemble.generate_reasoning(
            cleaned_text,
            final_prediction,
            final_confidence,
            probs[0],  # RoBERTa probabilities (0.8 weight)
            probs[0] if len(probs) > 0 else [0.5, 0.5]  # SBERT-NB
        )

        return jsonify({
            'label': label,
            'confidence': round(float(final_confidence) * 100, 2),
            'input_type': input_type,
            'text_preview': raw_text[:200] + '...' if len(raw_text) > 200 else raw_text,
            'reasoning': reasoning,
            'model': 'ensemble (RoBERTa 0.8 + SBERT-NB 0.2)',
            'chunks_analyzed': len(chunks)
        })

    except ValueError as e:
        logger.warning(f"Invalid URL: {e}")
        return jsonify({'error': str(e)}), 400
    except requests.RequestException as e:
        logger.error(f"URL fetch failed: {e}")
        return jsonify({'error': f'Failed to fetch URL: {str(e)}'}), 502
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})


@app.route('/api/search', methods=['POST'])
def search():
    """
    Search for articles using Google Custom Search Engine.
    Request: {"query": "fake news", "num_results": 5}
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        num_results = int(data.get('num_results', 5))
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400
        
        logger.info(f"Searching for: '{query}'")
        articles = search_engine.search(query, num_results)
        
        return jsonify({
            'query': query,
            'results': articles,
            'count': len(articles),
            'status': 'success' if articles else 'no_results'
        })
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-and-analyze', methods=['POST'])
def search_and_analyze():
    """
    Search for articles and analyze them for fake news.
    Request: {"query": "topic to search", "num_results": 3}
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        num_results = int(data.get('num_results', 3))
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400
        
        logger.info(f"Searching and analyzing: '{query}'")
        articles = search_engine.search(query, num_results)
        
        # Analyze each article
        analyzed_articles = []
        for article in articles:
            try:
                # Extract text from URL
                article_text = extract_text_from_url(article['url'])
                
                # Clean and predict
                cleaned = clean_text(article_text)
                chunks = chunk_text(cleaned, max_tokens=256)
                predictions, confidences, _ = ensemble.predict(chunks, batch_size=1)
                
                final_prediction = int(sum(predictions) > len(predictions) / 2)
                final_confidence = float(np.mean(confidences))
                
                analyzed_articles.append({
                    'title': article['title'],
                    'url': article['url'],
                    'source': article['source'],
                    'snippet': article['snippet'],
                    'label': 'Real' if final_prediction == 1 else 'Fake',
                    'confidence': round(final_confidence * 100, 2),
                    'text_length': len(article_text)
                })
            except Exception as e:
                logger.warning(f"Could not analyze {article['url']}: {e}")
                analyzed_articles.append({
                    'title': article['title'],
                    'url': article['url'],
                    'source': article['source'],
                    'error': 'Could not fetch/analyze'
                })
        
        return jsonify({
            'query': query,
            'results': analyzed_articles,
            'count': len(analyzed_articles),
            'real_count': sum(1 for a in analyzed_articles if a.get('label') == 'Real'),
            'fake_count': sum(1 for a in analyzed_articles if a.get('label') == 'Fake')
        })
    
    except Exception as e:
        logger.error(f"Search and analyze error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/verify-claim', methods=['POST'])
def verify_claim():
    """
    Verify a specific claim by searching for evidence.
    Request: {"claim": "The Earth is flat"}
    """
    try:
        data = request.get_json()
        claim = data.get('claim', '').strip()
        
        if not claim:
            return jsonify({'error': 'Claim required'}), 400
        
        logger.info(f"Verifying claim: '{claim}'")
        verification = fact_checker.verify_claim(claim)
        
        return jsonify(verification)
    
    except Exception as e:
        logger.error(f"Claim verification error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/multi-source-check', methods=['POST'])
def multi_source_check():
    """
    Compare how multiple sources cover the same topic.
    Request: {"topic": "climate change"}
    """
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        
        if not topic:
            return jsonify({'error': 'Topic required'}), 400
        
        logger.info(f"Running multi-source check on: '{topic}'")
        analysis = fact_checker.multi_source_check(topic)
        
        return jsonify(analysis)
    
    except Exception as e:
        logger.error(f"Multi-source check error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/comprehensive-analysis', methods=['POST'])
def comprehensive_analysis():
    """
    UNIFIED ENDPOINT - Complete fake news analysis workflow
    
    1. Process text/URL input
    2. Automatically search for related articles
    3. Verify claims against sources
    4. Run ensemble prediction
    5. Return comprehensive analysis
    
    Request: {
        "input_type": "text|url",
        "input_value": "article text or URL",
        "search_results": 5
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        input_type = data.get('input_type', 'text')
        input_value = data.get('input_value', '').strip()
        search_results = int(data.get('search_results', 5))

        if not input_value:
            return jsonify({'error': 'Input cannot be empty'}), 400

        logger.info(f"[COMPREHENSIVE ANALYSIS] Starting unified workflow for {input_type} input")
        
        # STEP 1: Extract and clean text
        logger.info("[Step 1/5] Extracting and cleaning text...")
        if input_type == 'url':
            raw_text = extract_text_from_url(input_value)
            source_url = input_value
            if not raw_text:
                return jsonify({'error': 'No readable text found at URL'}), 422
        else:
            raw_text = input_value
            source_url = None

        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            return jsonify({'error': 'Text cleaning resulted in empty content'}), 422

        # STEP 2: Run initial NLP prediction
        logger.info("[Step 2/5] Running initial ensemble prediction...")
        chunks = chunk_text(cleaned_text, max_tokens=256)
        predictions, confidences, probs = ensemble.predict(chunks, batch_size=1)
        
        initial_prediction = int(sum(predictions) > len(predictions) / 2)
        initial_confidence = float(np.mean(confidences))
        initial_label = 'Real' if initial_prediction == 1 else 'Fake'
        
        # STEP 3: Search for related articles automatically
        logger.info(f"[Step 3/5] Searching for related articles (limit: {search_results})...")
        search_query = raw_text[:100]  # Use first 100 chars as search query
        try:
            related_articles = search_engine.search(search_query, num_results=search_results)
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            related_articles = []
        
        # STEP 4: Analyze related articles for verification
        logger.info("[Step 4/5] Analyzing related articles for verification...")
        verification_data = {
            'real_count': 0,
            'fake_count': 0,
            'articles_analyzed': 0,
            'articles': []
        }
        
        if related_articles:
            for article in related_articles[:3]:  # Analyze top 3 for speed
                try:
                    article_text = extract_text_from_url(article['url'])
                    article_cleaned = clean_text(article_text)
                    article_chunks = chunk_text(article_cleaned, max_tokens=256)
                    article_preds, article_conf, _ = ensemble.predict(article_chunks, batch_size=1)
                    
                    article_prediction = int(sum(article_preds) > len(article_preds) / 2)
                    article_label = 'Real' if article_prediction == 1 else 'Fake'
                    
                    if article_prediction == 1:
                        verification_data['real_count'] += 1
                    else:
                        verification_data['fake_count'] += 1
                    
                    verification_data['articles_analyzed'] += 1
                    verification_data['articles'].append({
                        'title': article['title'],
                        'source': article['source'],
                        'label': article_label,
                        'confidence': round(float(np.mean(article_conf)) * 100, 2)
                    })
                except Exception as e:
                    logger.warning(f"Could not analyze article {article['url']}: {e}")
                    continue
        
        # STEP 5: Generate comprehensive reasoning
        logger.info("[Step 5/5] Generating comprehensive analysis...")
        
        # Build context-aware reasoning
        reasoning = ensemble.generate_reasoning(
            cleaned_text,
            initial_prediction,
            initial_confidence,
            probs[0] if len(probs) > 0 else [0.5, 0.5],
            probs[0] if len(probs) > 0 else [0.5, 0.5]
        )
        
        # Add verification context to reasoning
        if verification_data['articles_analyzed'] > 0:
            real_pct = (verification_data['real_count'] / verification_data['articles_analyzed']) * 100
            verification_context = f"\n\nCross-verification with {verification_data['articles_analyzed']} related articles: {real_pct:.0f}% are classified as real content. "
            
            if real_pct > 70:
                verification_context += "This supports the initial prediction."
            elif real_pct < 30:
                verification_context += "This contradicts the initial prediction - be cautious."
            else:
                verification_context += "Mixed signals from related content."
            
            reasoning += verification_context
        
        logger.info("[COMPLETE] Comprehensive analysis finished successfully")
        
        # UNIFIED RESPONSE
        return jsonify({
            'status': 'success',
            'input_type': input_type,
            'source_url': source_url,
            
            # PRIMARY PREDICTION
            'prediction': {
                'label': initial_label,
                'confidence': round(initial_confidence * 100, 2),
                'reasoning': reasoning,
                'chunks_analyzed': len(chunks)
            },
            
            # VERIFICATION DATA
            'verification': {
                'related_articles_found': len(related_articles),
                'articles_analyzed': verification_data['articles_analyzed'],
                'real_count': verification_data['real_count'],
                'fake_count': verification_data['fake_count'],
                'consensus': 'Real' if verification_data['real_count'] > verification_data['fake_count'] else 'Mixed/Fake',
                'articles': verification_data['articles']
            },
            
            # METADATA
            'model': 'ensemble (RoBERTa 0.8 + SBERT-NB 0.2)',
            'text_preview': raw_text[:200] + '...' if len(raw_text) > 200 else raw_text,
            'processing_steps': 5
        })
    
    except ValueError as e:
        logger.warning(f"Invalid URL: {e}")
        return jsonify({'error': str(e)}), 400
    except requests.RequestException as e:
        logger.error(f"URL fetch failed: {e}")
        return jsonify({'error': f'Failed to fetch URL: {str(e)}'}), 502
    except Exception as e:
        logger.error(f"Comprehensive analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/<path:path>')
def serve_frontend(path=''):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=debug, port=port)

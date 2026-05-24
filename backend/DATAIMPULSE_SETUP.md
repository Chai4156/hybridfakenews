# DataImpulse Proxy Integration Guide

## ✅ Setup Complete

Your fake news detection system now has DataImpulse web proxy integration enabled.

### Configuration Details

**Proxy Credentials (from your DataImpulse account):**
- **Host**: `gw.dataimpulse.com`
- **Port**: `823`
- **Login**: `57dca839482c5fbd929d`
- **Password**: `bbl84g4848el2707`

**Status**: 
- ✅ Flask backend running with proxy enabled
- ✅ Text predictions working
- ✅ URL scraping with proxy fallback implemented
- ⚠️ Some sites still block proxy IPs (site-level restrictions)

---

## How It Works

### Architecture

```
Frontend (React)
    ↓
Backend Flask API (/api/predict)
    ↓
extract_text_from_url() → Attempts with proxy first
    ↓
    ├─ Success: Returns content
    └─ Fails: Retries without proxy
    ↓
NLP Pipeline (clean → chunk → predict → reason)
    ↓
Response to Frontend
```

### Configuration Files

**[config.py](./config.py)**
- Loads proxy credentials from `.env` file
- Prints proxy status on startup
- Enables IP rotation via DataImpulse's rotating proxy mode

**[.env](../.env)**
- Contains sensitive credentials
- ⚠️ **DO NOT commit to git** (add to `.gitignore`)
- Format: `KEY=value` pairs

**[app.py](./app.py)**
- `extract_text_from_url()` function uses proxy
- Implements retry logic with exponential backoff
- Falls back to direct connection if proxy fails

---

## Testing Your Proxy

### Test 1: Direct Terminal Test

```bash
# From backend directory
python -c "
import requests
from config import DATAIMPULSE_LOGIN, DATAIMPULSE_PASSWORD, DATAIMPULSE_HOST, DATAIMPULSE_PORT

proxy_url = f'http://{DATAIMPULSE_LOGIN}:{DATAIMPULSE_PASSWORD}@{DATAIMPULSE_HOST}:{DATAIMPULSE_PORT}'
proxies = {'http': proxy_url, 'https': proxy_url}

try:
    response = requests.get('https://www.bbc.com/news', proxies=proxies, timeout=10)
    print(f'✓ Proxy working! Status: {response.status_code}')
except Exception as e:
    print(f'✗ Proxy error: {e}')
"
```

### Test 2: Frontend Test

1. Open http://localhost:5173
2. Switch to **URL Input** tab
3. Try these URLs in order:

| URL | Expected | Notes |
|-----|----------|-------|
| https://www.bbc.com/news | ✅ Works | Permissive site |
| https://www.reuters.com | ✅ Works | Major news site |
| https://kmsnews.org/... | ⚠️ May fail | Aggressive bot protection |

---

## Proxy Settings & Optimization

### DataImpulse Configuration Options

From your control panel, you can configure:

**1. IP Rotation Settings** (Already enabled)
   - Current: Rotating mode (each request gets new IP)
   - Alternative: Sticky mode (same IP for session)

**2. Geographic Targeting**
   - Select specific countries to reduce blocks
   - Useful if target site has geo-restrictions

**3. Anonymous Filter**
   - Enable for more anonymous-looking requests
   - May impact speed slightly

**4. User-Agent Rotation**
   - Built into our headers (not DataImpulse setting)
   - Rotates through browser-like User-Agents

### Recommended Settings for News Scraping

In your DataImpulse dashboard, consider:

```
Type: Rotating (current)
Protocol: HTTP/HTTPS ✓
Countries: Any (or specific)
Anonymous Filter: Disabled (unless sites block)
Rotation Interval: Default
```

---

## Troubleshooting

### Issue: "Connection aborted" error still appears

**Causes & Solutions:**

1. **Site actively blocks residential IPs**
   - Solution: The proxy will automatically fall back to direct connection
   - Some sites actively block all proxy traffic

2. **DataImpulse account limit reached**
   - Check: Dashboard → Usage → GB remaining
   - Solution: Add more GB or wait for monthly reset

3. **Credentials incorrect**
   - Check: [.env](../.env) file values match DataImpulse dashboard
   - Re-enter credentials if uncertain

4. **Proxy temporarily offline**
   - Solution: Fallback to direct connection is built-in
   - Try again in a few minutes

### Issue: Proxy is enabled but making requests slow

**Solution:**
- Disable proxy for development: Set `PROXY_ENABLED=False` in [.env](../.env)
- Keep enabled for production (bypass blocks)

---

## Advanced Usage

### Disable/Enable Proxy at Runtime

Edit [.env](../.env):
```bash
# Disable proxy
PROXY_ENABLED=False

# Re-enable proxy
PROXY_ENABLED=True
```

Then restart Flask:
```bash
# Kill current process (Ctrl+C in terminal)
cd backend && python app.py
```

### Monitor Proxy Requests

Check logs during requests:
```bash
# Watch for these log messages:
# [CONFIG] Proxy enabled: True
# Using DataImpulse proxy: gw.dataimpulse.com:823
# Successfully fetched URL: https://...
# Request failed: ... (means fallback to direct)
```

### Switch to Sticky IP Mode

If you need the same IP for a scraping session:

1. Go to DataImpulse dashboard
2. Change from "Rotating" to "Sticky"
3. Note the sticky IP address
4. Update [config.py](./config.py) if needed

---

## API Response Format

When proxy succeeds:
```json
{
  "label": "Real/Fake",
  "confidence": 85.5,
  "reasoning": "Model explanation...",
  "text_preview": "First 100 chars...",
  "chunks_analyzed": 3,
  "model_info": "RoBERTa + SBERT"
}
```

When proxy fails (fallback):
```json
{
  "label": "Real/Fake",
  "confidence": 82.3,
  "reasoning": "Model explanation...",
  "note": "Fetched without proxy"
}
```

---

## Performance Notes

- **With proxy**: +3-5 seconds per URL (routing overhead)
- **Without proxy**: 1-2 seconds per URL
- **Fallback mechanism**: Automatic, no user action needed
- **Caching**: Text is cached during session to avoid re-fetching

---

## Security Reminders

⚠️ **DO NOT:**
- Commit `.env` file to git
- Share credentials in emails
- Use in public repositories
- Log credentials in error messages

✅ **DO:**
- Add `.env` to `.gitignore`
- Keep credentials updated in DataImpulse
- Monitor usage in dashboard
- Rotate credentials periodically if concerned

---

## Next Steps

1. **For immediate testing:**
   ```bash
   curl -x "http://57dca839482c5fbd929d:bbl84g4848el2707@gw.dataimpulse.com:823" \
        "https://www.bbc.com/news"
   ```

2. **For production deployment:**
   - Add `.env` file to deployment secrets
   - Monitor proxy usage in DataImpulse dashboard
   - Set up alerts if GB usage is high

3. **For better blocking bypass:**
   - Contact DataImpulse support if specific sites still block
   - They can whitelist IPs for your account
   - Consider residential proxy for sensitive targets

---

## Files Modified

- `backend/app.py` - URL scraping with proxy support
- `backend/config.py` - NEW - Configuration loader
- `backend/.env` - NEW - Your credentials (not in git)
- `backend/.env.example` - Template for setup

---

## Support

For DataImpulse support: https://dataimpulse.com/documentation
For code issues: Check Flask logs in terminal

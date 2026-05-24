"""
Configuration for web scraping and proxy settings.
Add your DataImpulse credentials here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ===== PROXY CONFIGURATION =====
# Set these to enable DataImpulse proxy

# Option 1: Using environment variables (recommended)
PROXY_ENABLED = os.getenv('PROXY_ENABLED', 'False').lower() == 'true'
PROXY_PROVIDER = os.getenv('PROXY_PROVIDER', 'dataimpulse')  # 'dataimpulse', 'custom', etc.

# DataImpulse credentials
DATAIMPULSE_API_KEY = os.getenv('DATAIMPULSE_API_KEY', '')
DATAIMPULSE_ENDPOINT = os.getenv('DATAIMPULSE_ENDPOINT', 'http://proxy.dataimpulse.com:8080')
DATAIMPULSE_USERNAME = os.getenv('DATAIMPULSE_USERNAME', '')
DATAIMPULSE_LOGIN = os.getenv('DATAIMPULSE_LOGIN', '')
DATAIMPULSE_PASSWORD = os.getenv('DATAIMPULSE_PASSWORD', '')
DATAIMPULSE_HOST = os.getenv('DATAIMPULSE_HOST', 'gw.dataimpulse.com')
DATAIMPULSE_PORT = int(os.getenv('DATAIMPULSE_PORT', '823'))

# Proxy rotation settings
ROTATE_PROXY_IPS = os.getenv('ROTATE_PROXY_IPS', 'True').lower() == 'true'
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '20'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))

print(f"[CONFIG] Proxy enabled: {PROXY_ENABLED}")
if PROXY_ENABLED:
    print(f"[CONFIG] Using {PROXY_PROVIDER} proxy with IP rotation: {ROTATE_PROXY_IPS}")

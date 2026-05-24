import { useState, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || ''

const EXAMPLES = [
  { label: 'Try real news', text: 'The Federal Reserve raised interest rates by 25 basis points on Wednesday, citing continued progress on inflation while noting risks to the economic outlook.' },
  { label: 'Try fake news',  text: 'BREAKING: Scientists EXPOSED for hiding SECRET data proving the earth is FLAT. Government officials caught deleting evidence — share before it\'s removed!' },
]

export default function App() {
  const [tab, setTab] = useState('text')
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const textareaRef = useRef(null)

  const handleSubmit = async () => {
    if (!input.trim()) { setError('Please enter some text or a URL.'); return }
    setError('')
    setResult(null)
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/api/comprehensive-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          input_type: tab, 
          input_value: input.trim(),
          search_results: 5
        })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Analysis failed')
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleExample = (text) => {
    setTab('text')
    setInput(text)
    setResult(null)
    setError('')
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSubmit()
  }

  const isFake = result?.prediction?.label === 'Fake'

  return (
    <div style={styles.wrapper}>
      <div style={styles.grid} />

      <header style={styles.header}>
        <div style={styles.logoRow}>
          <span style={styles.logoIcon}>◈</span>
          <span style={styles.logoText}>FACT CHECK</span>
        </div>
        <p style={styles.tagline}>AI-powered fake news detection with source verification — powered by RoBERTa + SBERT</p>
        <div style={styles.badgeRow}>
          <span style={styles.badge}>NLP</span>
          <span style={styles.badge}>Ensemble Learning</span>
          <span style={styles.badge}>Fact Verification</span>
        </div>
      </header>

      <main style={styles.main}>
        <div style={styles.card}>
          <div style={styles.tabRow}>
            {['text', 'url'].map(t => (
              <button
                key={t}
                onClick={() => { setTab(t); setResult(null); setInput(''); setError('') }}
                style={{ ...styles.tab, ...(tab === t ? styles.tabActive : {}) }}
              >
                {t === 'text' ? '▤ Text Input' : '⎋ URL Input'}
              </button>
            ))}
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>
              {tab === 'text' ? 'Paste or type news article text:' : 'Enter article URL:'}
            </label>

            {tab === 'text' ? (
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Paste news content here... (Ctrl+Enter to analyze)"
                style={styles.textarea}
                rows={7}
              />
            ) : (
              <input
                type="url"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                placeholder="https://example.com/news-article"
                style={styles.urlInput}
              />
            )}
          </div>

          {error && <div style={styles.errorBox}>⚠ {error}</div>}

          <button
            onClick={handleSubmit}
            disabled={loading}
            style={{ ...styles.analyzeBtn, ...(loading ? styles.analyzeBtnDisabled : {}) }}
          >
            {loading ? (
              <span style={styles.spinnerRow}>
                <span style={styles.spinner} /> Analyzing & Verifying...
              </span>
            ) : '→ Analyze with Verification'}
          </button>

          <div style={styles.exampleRow}>
            {EXAMPLES.map(ex => (
              <button key={ex.label} onClick={() => handleExample(ex.text)} style={styles.exampleBtn}>
                {ex.label}
              </button>
            ))}
          </div>
        </div>

        {result && (
          <>
            <div style={{ ...styles.resultCard, ...(isFake ? styles.resultFake : styles.resultReal) }}>
              <div style={styles.resultHeader}>
                <span style={styles.resultIcon}>{isFake ? '✕' : '✓'}</span>
                <div>
                  <div style={{ ...styles.resultLabel, color: isFake ? 'var(--red)' : 'var(--green)' }}>
                    {isFake ? 'Likely Fake News' : 'Likely Reliable Content'}
                  </div>
                  <div style={styles.resultSub}>
                    {isFake
                      ? 'Patterns consistent with misinformation detected.'
                      : 'No significant fake news patterns detected.'}
                  </div>
                </div>
              </div>

              <div style={styles.confRow}>
                <span style={styles.confLabel}>Confidence</span>
                <span style={{ ...styles.confValue, color: isFake ? 'var(--red)' : 'var(--green)' }}>
                  {result.prediction.confidence}%
                </span>
              </div>

              <div style={styles.barTrack}>
                <div style={{
                  ...styles.barFill,
                  width: `${result.prediction.confidence}%`,
                  background: isFake ? 'var(--red)' : 'var(--green)'
                }} />
              </div>

              {result.prediction.reasoning && (
                <div style={styles.reasoningBox}>
                  <span style={styles.reasoningLabel}>ANALYSIS & REASONING</span>
                  <p style={styles.reasoningText}>{result.prediction.reasoning}</p>
                </div>
              )}

              {result.text_preview && (
                <div style={styles.preview}>
                  <span style={styles.previewLabel}>ANALYZED TEXT</span>
                  <p style={styles.previewText}>"{result.text_preview}"</p>
                </div>
              )}
            </div>

            {result.verification && (
              <div style={styles.verificationCard}>
                <div style={styles.verificationHeader}>
                  <span style={styles.verificationIcon}>🔍</span>
                  <div>
                    <div style={styles.verificationTitle}>Source Verification</div>
                    <div style={styles.verificationSub}>Cross-checked against {result.verification.articles_analyzed} related sources</div>
                  </div>
                </div>

                <div style={styles.statsRow}>
                  <div style={styles.statBox}>
                    <div style={styles.statValue}>{result.verification.real_count}</div>
                    <div style={styles.statLabel}>Real Sources</div>
                  </div>
                  <div style={styles.statBox}>
                    <div style={styles.statValue}>{result.verification.fake_count}</div>
                    <div style={styles.statLabel}>Fake Sources</div>
                  </div>
                  <div style={styles.statBox}>
                    <div style={{...styles.statValue, color: result.verification.consensus === 'Real' ? 'var(--green)' : 'var(--red)'}}>
                      {result.verification.consensus}
                    </div>
                    <div style={styles.statLabel}>Consensus</div>
                  </div>
                </div>

                {result.verification.articles && result.verification.articles.length > 0 && (
                  <div style={styles.articlesSection}>
                    <span style={styles.articlesLabel}>RELATED ARTICLES ANALYZED</span>
                    <div style={styles.articlesList}>
                      {result.verification.articles.map((article, idx) => (
                        <div key={idx} style={styles.articleItem}>
                          <div style={styles.articleMeta}>
                            <span style={{...styles.articleLabel, color: article.label === 'Real' ? 'var(--green)' : 'var(--red)'}}>
                              {article.label === 'Real' ? '✓' : '✕'} {article.label}
                            </span>
                            <span style={styles.articleSource}>{article.source}</span>
                          </div>
                          <div style={styles.articleTitle}>{article.title}</div>
                          <div style={styles.articleConfidence}>Confidence: {article.confidence.toFixed(1)}%</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            <p style={styles.disclaimer}>
              ℹ This analysis is performed by an AI system combining NLP predictions with multi-source verification. Always cross-check with trusted news organizations.
            </p>
          </>
        )}

        <section style={styles.infoGrid}>
          {INFO_CARDS.map(card => (
            <div key={card.title} style={styles.infoCard}>
              <div style={styles.infoIcon}>{card.icon}</div>
              <div style={styles.infoTitle}>{card.title}</div>
              <div style={styles.infoText}>{card.text}</div>
            </div>
          ))}
        </section>
      </main>

      <footer style={styles.footer}>
        <span>Fake News Detector with Source Verification</span>
        <span style={{ color: 'var(--muted)' }}>RoBERTa + SBERT-NB Ensemble · Google Custom Search · Accuracy: 92.5%</span>
      </footer>
    </div>
  )
}

const INFO_CARDS = [
  { icon: '⬡', title: 'RoBERTa Transformer', text: 'Fine-tuned transformer with bidirectional context understanding for nuanced language analysis.' },
  { icon: '⬡', title: 'SBERT + Naive Bayes', text: 'Semantic embeddings with probabilistic classifier for interpretable, fast text feature extraction.' },
  { icon: '⬡', title: 'Hybrid Ensemble', text: 'Soft-voting ensemble combines both models — balancing speed with deep semantic accuracy.' },
  { icon: '⬡', title: 'URL Scraping', text: 'Paste any news URL and the system extracts article text automatically via BeautifulSoup.' },
]

const styles = {
  wrapper: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
  },
  grid: {
    position: 'fixed',
    inset: 0,
    backgroundImage: `linear-gradient(rgba(232,255,0,0.03) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(232,255,0,0.03) 1px, transparent 1px)`,
    backgroundSize: '48px 48px',
    pointerEvents: 'none',
    zIndex: 0,
  },
  header: {
    position: 'relative',
    zIndex: 1,
    textAlign: 'center',
    padding: '60px 24px 40px',
    borderBottom: '1px solid #1a1a2a',
  },
  logoRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '12px',
    marginBottom: '12px',
  },
  logoIcon: { fontSize: '2rem', color: 'var(--accent)' },
  logoText: {
    fontFamily: 'var(--font-head)',
    fontSize: '3rem',
    fontWeight: 800,
    letterSpacing: '0.15em',
    color: 'var(--accent)',
  },
  tagline: {
    color: 'var(--muted)',
    fontSize: '0.85rem',
    marginBottom: '16px',
    letterSpacing: '0.05em',
  },
  badgeRow: { display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' },
  badge: {
    background: '#1a1a2a',
    border: '1px solid var(--border)',
    color: 'var(--muted)',
    fontSize: '0.7rem',
    padding: '4px 10px',
    letterSpacing: '0.08em',
    borderRadius: '2px',
  },
  main: {
    flex: 1,
    position: 'relative',
    zIndex: 1,
    maxWidth: '780px',
    width: '100%',
    margin: '0 auto',
    padding: '48px 24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '28px',
  },
  card: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    padding: '32px',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  tabRow: { display: 'flex', gap: '2px', borderBottom: '1px solid var(--border)', paddingBottom: '16px' },
  tab: {
    background: 'none',
    border: 'none',
    color: 'var(--muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8rem',
    cursor: 'pointer',
    padding: '8px 16px',
    letterSpacing: '0.05em',
    transition: 'color 0.2s',
  },
  tabActive: {
    color: 'var(--accent)',
    borderBottom: '2px solid var(--accent)',
  },
  inputGroup: { display: 'flex', flexDirection: 'column', gap: '8px' },
  label: { color: 'var(--muted)', fontSize: '0.75rem', letterSpacing: '0.08em' },
  textarea: {
    width: '100%',
    background: '#0d0d14',
    border: '1px solid var(--border)',
    color: 'var(--text)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.85rem',
    padding: '14px 16px',
    resize: 'vertical',
    outline: 'none',
    lineHeight: 1.7,
    transition: 'border-color 0.2s',
  },
  urlInput: {
    width: '100%',
    background: '#0d0d14',
    border: '1px solid var(--border)',
    color: 'var(--text)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.85rem',
    padding: '14px 16px',
    outline: 'none',
  },
  errorBox: {
    background: 'rgba(255,59,92,0.1)',
    border: '1px solid rgba(255,59,92,0.3)',
    color: 'var(--red)',
    fontSize: '0.8rem',
    padding: '10px 14px',
  },
  analyzeBtn: {
    background: 'var(--accent)',
    border: 'none',
    color: '#000',
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    fontSize: '0.9rem',
    letterSpacing: '0.08em',
    padding: '14px 28px',
    cursor: 'pointer',
    width: '100%',
    transition: 'opacity 0.2s, transform 0.1s',
  },
  analyzeBtnDisabled: { opacity: 0.5, cursor: 'not-allowed' },
  spinnerRow: { display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' },
  spinner: {
    display: 'inline-block',
    width: '14px',
    height: '14px',
    border: '2px solid #000',
    borderTopColor: 'transparent',
    borderRadius: '50%',
    animation: 'spin 0.7s linear infinite',
  },
  exampleRow: { display: 'flex', gap: '8px', flexWrap: 'wrap' },
  exampleBtn: {
    background: 'none',
    border: '1px solid var(--border)',
    color: 'var(--muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.72rem',
    padding: '6px 12px',
    cursor: 'pointer',
    letterSpacing: '0.05em',
    transition: 'border-color 0.2s, color 0.2s',
  },
  resultCard: {
    border: '1px solid',
    padding: '28px 32px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    animation: 'fadeIn 0.4s ease',
  },
  resultFake: {
    background: 'rgba(255,59,92,0.05)',
    borderColor: 'rgba(255,59,92,0.3)',
  },
  resultReal: {
    background: 'rgba(0,229,160,0.05)',
    borderColor: 'rgba(0,229,160,0.3)',
  },
  resultHeader: { display: 'flex', gap: '16px', alignItems: 'flex-start' },
  resultIcon: { fontSize: '1.6rem', lineHeight: 1, marginTop: '2px' },
  resultLabel: { fontFamily: 'var(--font-head)', fontSize: '1.4rem', fontWeight: 700 },
  resultSub: { color: 'var(--muted)', fontSize: '0.8rem', marginTop: '4px' },
  confRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  confLabel: { color: 'var(--muted)', fontSize: '0.75rem', letterSpacing: '0.08em' },
  confValue: { fontFamily: 'var(--font-head)', fontSize: '1.6rem', fontWeight: 800 },
  barTrack: { height: '4px', background: 'rgba(255,255,255,0.07)', borderRadius: '2px', overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: '2px', transition: 'width 0.8s ease' },
  preview: {
    background: 'rgba(0,0,0,0.2)',
    padding: '14px 16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  previewLabel: { color: 'var(--muted)', fontSize: '0.65rem', letterSpacing: '0.12em' },
  previewText: { color: 'var(--text)', fontSize: '0.8rem', lineHeight: 1.6, fontStyle: 'italic' },
  reasoningBox: {
    background: 'rgba(0,0,0,0.2)',
    padding: '14px 16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    borderLeft: '3px solid var(--accent)',
  },
  reasoningLabel: { color: 'var(--accent)', fontSize: '0.65rem', letterSpacing: '0.12em', fontWeight: 600 },
  reasoningText: { color: 'var(--text)', fontSize: '0.8rem', lineHeight: 1.6 },
  disclaimer: { color: 'var(--muted)', fontSize: '0.72rem', lineHeight: 1.5 },
  verificationCard: {
    border: '1px solid rgba(0,229,160,0.3)',
    background: 'rgba(0,229,160,0.02)',
    padding: '28px 32px',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
    animation: 'fadeIn 0.4s ease',
  },
  verificationHeader: { display: 'flex', gap: '16px', alignItems: 'flex-start' },
  verificationIcon: { fontSize: '1.6rem', lineHeight: 1, marginTop: '2px' },
  verificationTitle: { fontFamily: 'var(--font-head)', fontSize: '1.2rem', fontWeight: 700, color: 'var(--text)' },
  verificationSub: { color: 'var(--muted)', fontSize: '0.75rem', marginTop: '4px' },
  statsRow: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' },
  statBox: {
    background: 'rgba(0,0,0,0.2)',
    padding: '14px 16px',
    textAlign: 'center',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  statValue: { fontFamily: 'var(--font-head)', fontSize: '1.4rem', fontWeight: 800, color: 'var(--accent)' },
  statLabel: { color: 'var(--muted)', fontSize: '0.65rem', letterSpacing: '0.08em' },
  articlesSection: { display: 'flex', flexDirection: 'column', gap: '12px' },
  articlesLabel: { color: 'var(--muted)', fontSize: '0.65rem', letterSpacing: '0.12em', fontWeight: 600 },
  articlesList: { display: 'flex', flexDirection: 'column', gap: '12px' },
  articleItem: {
    background: 'rgba(0,0,0,0.2)',
    padding: '12px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    borderLeft: '2px solid rgba(0,229,160,0.5)',
  },
  articleMeta: { display: 'flex', gap: '12px', alignItems: 'center' },
  articleLabel: { fontWeight: 600, fontSize: '0.75rem', letterSpacing: '0.05em' },
  articleSource: { color: 'var(--muted)', fontSize: '0.7rem' },
  articleTitle: { color: 'var(--text)', fontSize: '0.8rem', fontWeight: 500, lineHeight: 1.4 },
  articleConfidence: { color: 'var(--muted)', fontSize: '0.7rem' },
  disclaimer: { color: 'var(--muted)', fontSize: '0.72rem', lineHeight: 1.5 },
  infoGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
    gap: '1px',
    background: 'var(--border)',
    border: '1px solid var(--border)',
  },
  infoCard: {
    background: 'var(--surface)',
    padding: '24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  infoIcon: { color: 'var(--accent)', fontSize: '1.2rem' },
  infoTitle: { fontFamily: 'var(--font-head)', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text)' },
  infoText: { color: 'var(--muted)', fontSize: '0.75rem', lineHeight: 1.6 },
  footer: {
    position: 'relative',
    zIndex: 1,
    borderTop: '1px solid var(--border)',
    padding: '20px 24px',
    display: 'flex',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: '8px',
    fontSize: '0.72rem',
    color: 'var(--muted)',
    letterSpacing: '0.05em',
  },
}

// ═══════════════════════════════════════════════════════════════
//  PhishGuard AI — Frontend JavaScript (app.js)
//  Handles: UI interactions, heuristic scanning, ML API calls
// ═══════════════════════════════════════════════════════════════

// ─── Constants ──────────────────────────────────────────────────────────────

const SUSPICIOUS_KW = [
  'login','signin','verify','account','secure','update','confirm',
  'banking','paypal','amazon','google','microsoft','apple','netflix','password',
  'credential','urgent','suspended','limited','click','free','winner','prize','bitcoin'
];

const TRUSTED = [
  'google.com','microsoft.com','apple.com','amazon.com','paypal.com',
  'facebook.com','twitter.com','instagram.com','linkedin.com','github.com',
  'youtube.com','wikipedia.org','stackoverflow.com','reddit.com'
];

const SUSP_TLD = [
  '.tk','.ml','.ga','.cf','.gq','.xyz','.top',
  '.work','.click','.link','.download','.zip','.review'
];

const BRANDS = [
  'paypal','apple','google','microsoft','amazon','netflix',
  'bankofamerica','citibank','wellsfargo'
];

const EMAIL_PATTERNS = [
  [/click here (immediately|now|asap)/i,                         'Urgency click trap phrase'],
  [/your account (will be|has been|is) (suspended|terminated|blocked|locked)/i, 'Account threat language'],
  [/verify your (account|identity|information|email|password)/i, 'Identity verification scam phrase'],
  [/update your (payment|billing|credit card|bank)/i,            'Payment update phishing phrase'],
  [/you (have|'ve) won/i,                                         'Prize/lottery bait language'],
  [/urgent (action|response|attention) required/i,               'Urgency manipulation tactic'],
  [/confirm your (password|credentials|login)/i,                 'Credential theft phrase'],
  [/limited time offer/i,                                        'FOMO manipulation phrase'],
  [/act (now|immediately|fast)/i,                                'Urgency pressure phrase'],
  [/dear (customer|user|member|valued|sir|madam|friend)/i,       'Generic non-personalized greeting'],
  [/your (account|password) (expires|expiring|will expire)/i,    'Expiry fear tactic'],
  [/wire transfer/i,                                             'Wire transfer fraud indicator'],
  [/send (me )?money/i,                                          'Direct money request'],
  [/final notice/i,                                              'False urgency — "final notice"'],
  [/your (order|package|shipment|delivery) (has been|is being) (held|stopped|delayed)/i, 'Package scam pattern'],
];

// ─── Feature Extraction ─────────────────────────────────────────────────────

/**
 * Calculate Shannon entropy of a string
 * @param {string} s
 * @returns {number}
 */
function entropy(s) {
  if (!s) return 0;
  const freq = {};
  for (const c of s) freq[c] = (freq[c] || 0) + 1;
  return -Object.values(freq).reduce((sum, n) => {
    const p = n / s.length;
    return sum + p * Math.log2(p);
  }, 0);
}

/**
 * Safe URL parser
 * @param {string} url
 * @returns {URL|null}
 */
function parseURL(url) {
  try {
    const u = url.includes('://') ? url : 'http://' + url;
    return new URL(u);
  } catch { return null; }
}

/**
 * Extract 15 URL features for scoring
 * @param {string} url
 * @returns {Object|null}
 */
function extractURLFeatures(url) {
  const p = parseURL(url);
  if (!p) return null;

  const domain   = p.hostname.replace(/^www\./, '').toLowerCase();
  const full     = url.toLowerCase();
  const subdomain = domain.split('.').slice(0, -2).join('.');

  const isTrusted    = TRUSTED.some(t => domain === t || domain.endsWith('.' + t));
  const brandInSub   = BRANDS.some(b => subdomain.includes(b)) && !isTrusted;
  const isIP         = /^\d+\.\d+\.\d+\.\d+$/.test(p.hostname);
  const suspTLD      = SUSP_TLD.some(t => domain.endsWith(t));
  const kwCount      = SUSPICIOUS_KW.filter(k => full.includes(k)).length;
  const hexEnc       = /%[0-9a-fA-F]{2}/.test(url);
  const domainEnt    = entropy(domain);
  const subDepth     = domain.split('.').length - 2;
  const numDigits    = (domain.match(/\d/g) || []).length;
  const doubleSlash  = url.slice(8).includes('//');
  const qParams      = [...p.searchParams.keys()].length;

  return {
    url_length:        url.length,
    has_https:         p.protocol === 'https:',
    has_ip:            isIP,
    has_at:            url.includes('@'),
    subdomain_depth:   Math.max(0, subDepth),
    domain_entropy:    domainEnt.toFixed(3),
    keyword_count:     kwCount,
    suspicious_tld:    suspTLD,
    is_trusted:        isTrusted,
    brand_in_subdomain: brandInSub,
    hex_encoding:      hexEnc,
    has_port:          !!p.port,
    digit_count:       numDigits,
    query_params:      qParams,
    double_slash:      doubleSlash,
    _raw: { isIP, brandInSub, suspTLD, kwCount, hexEnc, domainEnt, subDepth, isTrusted, numDigits, qParams, doubleSlash }
  };
}

// ─── Heuristic Scoring ──────────────────────────────────────────────────────

/**
 * Score a URL 0–100 using weighted heuristics
 * @param {Object} f - extracted features
 * @returns {{ score: number, reasons: string[], confidence: number }}
 */
function scoreURL(f) {
  if (!f) return { score: 50, reasons: ['Could not parse URL'], confidence: 40 };

  const r = f._raw;
  let score = 0;
  const reasons = [];

  const add = (cond, weight, msg) => { if (cond) { score += weight; reasons.push(msg); } };

  add(r.isIP,              25, 'IP address used instead of domain name');
  add(f.has_at,            20, 'Contains @ symbol — URL redirection trick');
  add(r.brandInSub,        22, 'Trusted brand name spoofed in subdomain');
  add(r.suspTLD,           18, 'Suspicious top-level domain detected');
  add(f.url_length > 75,   10, `Unusually long URL (${f.url_length} characters)`);
  add(f.subdomain_depth > 3, 10, `Deep subdomain nesting (${f.subdomain_depth} levels)`);
  add(r.hexEnc,            12, 'Hexadecimal encoding detected in URL');
  add(r.kwCount >= 3,      15, `${r.kwCount} phishing keywords found in URL`);
  add(r.kwCount === 1 || r.kwCount === 2, 6, `Phishing keyword(s) detected (${r.kwCount})`);
  add(r.domainEnt > 3.8,   10, `High domain entropy (${parseFloat(f.domain_entropy).toFixed(2)}) — randomized chars`);
  add(!f.has_https,         8, 'No HTTPS — unencrypted connection');
  add(f.has_port,          12, 'Non-standard port detected');
  add(r.numDigits > 3,      7, `${r.numDigits} digits in domain name`);
  add(r.doubleSlash,       10, 'Double slash redirect trick detected');
  add(r.qParams > 5,        5, `Excessive query parameters (${r.qParams})`);

  if (r.isTrusted) score = Math.max(0, score - 30);
  score = Math.min(100, score);

  const confidence = Math.min(95, 55 + reasons.length * 5);
  return { score, reasons, confidence };
}

// ─── Email Analysis ──────────────────────────────────────────────────────────

/**
 * Analyze email text for phishing indicators
 * @param {string} text
 * @returns {{ score, reasons, confidence, features, urlResults }}
 */
function analyzeEmail(text) {
  const lower   = text.toLowerCase();
  const reasons = [];

  // Pattern matching
  EMAIL_PATTERNS.forEach(([re, msg]) => { if (re.test(text)) reasons.push(msg); });

  // Urgency words
  const urgencyWords = ['urgent','immediately','asap','expires','expire','last chance','act fast','today only','final notice'];
  const urgCount = urgencyWords.filter(w => lower.includes(w)).length;
  if (urgCount >= 2) reasons.push(`${urgCount} urgency trigger words detected`);

  // Threat words
  const threatWords = ['suspended','terminated','blocked','locked','deleted','unauthorized','illegal','violation','fraud'];
  const threatCount = threatWords.filter(w => lower.includes(w)).length;
  if (threatCount >= 1) reasons.push(`${threatCount} threat/fear word(s) used`);

  // Money bait
  const moneyWords = ['free','win','won','prize','cash','bitcoin','transfer','payment','reward','gift'];
  const moneyCount = moneyWords.filter(w => lower.includes(w)).length;
  if (moneyCount >= 2) reasons.push(`${moneyCount} money-bait words found`);

  // Punctuation abuse
  const excCount = (text.match(/!/g) || []).length;
  if (excCount > 3) reasons.push(`Excessive exclamation marks (${excCount})`);

  // Caps ratio
  const capsRatio = (text.match(/[A-Z]/g) || []).length / Math.max(text.length, 1);
  if (capsRatio > 0.15) reasons.push(`Excessive capitalization (${(capsRatio * 100).toFixed(0)}%)`);

  // Misspellings
  const misspell = ['acount','securty','verifiy','infomation','recieve','pasword','secutiry','exprie','confirmaton'];
  const misspCount = misspell.filter(w => lower.includes(w)).length;
  if (misspCount) reasons.push(`${misspCount} common phishing misspelling(s)`);

  // Embedded URLs
  const urlMatches = text.match(/https?:\/\/[^\s<>"]+/g) || [];
  const urlResults = urlMatches.slice(0, 5).map(u => {
    const f  = extractURLFeatures(u);
    const { score } = scoreURL(f);
    return { url: u, score };
  });
  const suspURLs = urlResults.filter(u => u.score > 40).length;
  if (suspURLs > 0) reasons.push(`${suspURLs} suspicious URL(s) embedded in email`);

  // Final score
  let score = 0;
  score += Math.min(40, reasons.length * 12);
  score += suspURLs * 15;
  score = Math.min(100, score);

  const confidence = Math.min(95, 50 + reasons.length * 6);

  const features = {
    word_count:     text.split(/\s+/).length,
    url_count:      urlMatches.length,
    suspicious_urls: suspURLs,
    urgency_words:  urgCount,
    threat_words:   threatCount,
    money_words:    moneyCount,
    exclamations:   excCount,
    caps_pct:       (capsRatio * 100).toFixed(1) + '%',
    misspellings:   misspCount,
  };

  return { score, reasons, confidence, features, urlResults };
}

// ─── Verdict Helpers ────────────────────────────────────────────────────────

/**
 * Map risk score to a verdict object
 * @param {number} score
 */
function getVerdict(score) {
  if (score >= 70) return { label:'PHISHING',   level:'danger',  emoji:'🚨', color:'var(--danger)', msg:'High risk — do NOT click or respond.' };
  if (score >= 40) return { label:'SUSPICIOUS', level:'warning', emoji:'⚠️',  color:'var(--warn)',   msg:'Proceed with extreme caution.' };
  if (score >= 20) return { label:'LOW RISK',   level:'caution', emoji:'🔍', color:'var(--caution)', msg:'Looks relatively safe, stay alert.' };
  return               { label:'SAFE',          level:'safe',    emoji:'✅', color:'var(--safe)',    msg:'No significant threats detected.' };
}

function bgColor(v) {
  if (v.level === 'danger')  return 'rgba(255,59,92,0.1)';
  if (v.level === 'warning') return 'rgba(255,176,32,0.1)';
  if (v.level === 'caution') return 'rgba(56,217,245,0.1)';
  return 'rgba(29,219,138,0.1)';
}

function borderColor(v) {
  if (v.level === 'danger')  return 'rgba(255,59,92,0.35)';
  if (v.level === 'warning') return 'rgba(255,176,32,0.35)';
  if (v.level === 'caution') return 'rgba(56,217,245,0.35)';
  return 'rgba(29,219,138,0.35)';
}

// ─── Render Result ───────────────────────────────────────────────────────────

/**
 * Render a result card into a container element
 * @param {HTMLElement} container
 * @param {Object} data - { score, reasons, confidence, features, urlResults, ml_details? }
 */
function renderResult(container, data) {
  const { score, reasons, confidence, features, urlResults, ml_details } = data;
  const v = getVerdict(score);

  const r    = 46, cx = 55, cy = 55;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;

  const pillsHTML = Object.entries(features).map(([k, val]) => `
    <div class="pill">
      <div class="pill-label">${k.replace(/_/g, ' ')}</div>
      <div class="pill-val">${val}</div>
    </div>
  `).join('');

  const indicatorsHTML = reasons.length
    ? reasons.map(r => `
        <div class="indicator" style="background:${bgColor(v)};border-left-color:${v.color}">
          <span class="indicator-arrow" style="color:${v.color}">›</span>
          <span class="indicator-text">${r}</span>
        </div>
      `).join('')
    : `<div class="indicator" style="background:rgba(29,219,138,0.08);border-left-color:var(--safe)">
        <span class="indicator-arrow" style="color:var(--safe)">✓</span>
        <span class="indicator-text">No threat indicators found</span>
      </div>`;

  const urlTableHTML = urlResults && urlResults.length ? `
    <div class="url-table">
      <div class="indicators-title" style="margin-top:20px">🔗 EMBEDDED URLS</div>
      ${urlResults.map(u => {
        const uv = getVerdict(u.score);
        return `<div class="url-row">
          <span class="v-pill" style="background:${bgColor(uv)};border:1px solid ${borderColor(uv)};color:${uv.color}">${uv.label}</span>
          <span class="url-score" style="color:${uv.color}">${u.score}</span>
          <span class="url-text">${u.url.slice(0, 70)}${u.url.length > 70 ? '…' : ''}</span>
        </div>`;
      }).join('')}
    </div>
  ` : '';

  // Optional ML details block
  const mlDetailsHTML = ml_details ? `
    <div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border2)">
      <div class="indicators-title">🤖 ML MODEL DETAILS</div>
      <div class="ml-feature-grid">
        ${Object.entries(ml_details.features || {}).map(([k,v]) => `
          <div class="ml-feature-item">
            <div class="ml-feature-name">${k.replace(/_/g,' ')}</div>
            <div class="ml-feature-val">${typeof v === 'number' ? v.toFixed(3) : v}</div>
            ${ml_details.importances && ml_details.importances[k]
              ? `<div class="ml-feature-importance">importance: ${ml_details.importances[k].toFixed(3)}</div>`
              : ''}
          </div>
        `).join('')}
      </div>
      ${ml_details.model_name ? `<div style="font-size:10px;color:var(--muted);margin-top:8px">Model: <span style="color:var(--accent);font-family:var(--mono)">${ml_details.model_name}</span></div>` : ''}
    </div>
  ` : '';

  container.innerHTML = `
    <div class="result card"
         style="background:linear-gradient(135deg,${bgColor(v)},var(--surface));
                border-color:${borderColor(v)};
                box-shadow:0 0 40px ${v.color}18">

      <div class="result-header">
        <div>
          <div class="verdict-badge" style="background:${bgColor(v)};border:1px solid ${borderColor(v)};color:${v.color}">
            <span>${v.emoji}</span>
            <span>${v.label}</span>
          </div>
          <div class="verdict-msg">${v.msg}</div>
        </div>
        <div class="ring-wrap">
          <svg width="110" height="110">
            <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--surface2)" stroke-width="9"/>
            <circle cx="${cx}" cy="${cy}" r="${r}" fill="none"
              stroke="${v.color}" stroke-width="9"
              stroke-dasharray="${dash} ${circ}"
              stroke-linecap="round"
              style="filter:drop-shadow(0 0 8px ${v.color});transition:stroke-dasharray 1s ease"/>
          </svg>
          <div class="ring-center">
            <span class="ring-score" style="color:${v.color}">${score}</span>
            <span class="ring-label">RISK</span>
          </div>
        </div>
      </div>

      <div class="conf-row">
        <span class="conf-label">Model Confidence</span>
        <span class="conf-val" style="color:${v.color};font-family:var(--mono)">${confidence}%</span>
      </div>
      <div class="conf-bar">
        <div class="conf-fill" style="width:${confidence}%;background:linear-gradient(90deg,${v.color}88,${v.color})"></div>
      </div>

      <div class="pills">${pillsHTML}</div>

      <div class="indicators-title">⚡ THREAT INDICATORS</div>
      ${indicatorsHTML}
      ${urlTableHTML}
      ${mlDetailsHTML}

      <div style="font-family:var(--mono);font-size:10px;color:var(--muted);text-align:right;margin-top:16px">
        Analyzed · ${new Date().toLocaleTimeString()}
      </div>
    </div>
  `;

  updateCounter();
}

// ─── URL Scanner ─────────────────────────────────────────────────────────────

function scanURL() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) return;

  const btn       = document.getElementById('url-btn');
  const container = document.getElementById('url-result');

  btn.disabled    = true;
  btn.textContent = '⏳ Scanning…';
  container.innerHTML = `<div class="card scanning"><div class="spinner"></div>Analyzing URL…</div>`;

  setTimeout(() => {
    const features = extractURLFeatures(url);
    const { score, reasons, confidence } = scoreURL(features);

    const displayFeatures = features ? {
      url_length:      features.url_length,
      has_https:       features.has_https,
      domain_entropy:  features.domain_entropy,
      keyword_count:   features.keyword_count,
      subdomain_depth: features.subdomain_depth,
      has_ip:          features.has_ip,
      is_trusted:      features.is_trusted,
      suspicious_tld:  features.suspicious_tld,
    } : {};

    renderResult(container, { score, reasons, confidence, features: displayFeatures, urlResults: [] });
    btn.disabled    = false;
    btn.textContent = '🔍 SCAN';
  }, 600);
}

// ─── Email Scanner ───────────────────────────────────────────────────────────

function scanEmail() {
  const text = document.getElementById('email-input').value.trim();
  if (!text) return;

  const btn       = document.getElementById('email-btn');
  const container = document.getElementById('email-result');

  btn.disabled    = true;
  btn.textContent = '⏳ Analyzing…';
  container.innerHTML = `<div class="card scanning"><div class="spinner"></div>Analyzing email content…</div>`;

  setTimeout(() => {
    const result = analyzeEmail(text);
    renderResult(container, result);
    btn.disabled    = false;
    btn.textContent = '🔍 ANALYZE EMAIL';
  }, 800);
}

// ─── Batch Scanner ───────────────────────────────────────────────────────────

function scanBatch() {
  const lines = document.getElementById('batch-input').value
    .split('\n').map(l => l.trim()).filter(Boolean);
  if (!lines.length) return;

  const btn       = document.getElementById('batch-btn');
  const container = document.getElementById('batch-result');

  btn.disabled    = true;
  btn.textContent = `⏳ Scanning ${lines.length} URLs…`;
  container.innerHTML = `<div class="card scanning"><div class="spinner"></div>Running batch analysis…</div>`;

  setTimeout(() => {
    const results = lines.slice(0, 20).map(url => {
      const f = extractURLFeatures(url);
      const { score, reasons } = scoreURL(f);
      const v = getVerdict(score);
      return { url, score, verdict: v, top: reasons[0] || 'No issues found' };
    });

    const phishCount = results.filter(r => r.score >= 70).length;
    const suspCount  = results.filter(r => r.score >= 40 && r.score < 70).length;
    const safeCount  = results.filter(r => r.score < 40).length;

    container.innerHTML = `
      <div class="card result" style="animation:slideUp 0.4s ease">
        <div style="display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap">
          <div style="background:rgba(255,59,92,0.1);border:1px solid rgba(255,59,92,0.3);border-radius:10px;padding:14px 20px;text-align:center;flex:1;min-width:80px">
            <div style="font-family:var(--mono);font-size:24px;font-weight:700;color:var(--danger)">${phishCount}</div>
            <div style="font-size:9px;color:var(--muted);letter-spacing:1.5px">PHISHING</div>
          </div>
          <div style="background:rgba(255,176,32,0.1);border:1px solid rgba(255,176,32,0.3);border-radius:10px;padding:14px 20px;text-align:center;flex:1;min-width:80px">
            <div style="font-family:var(--mono);font-size:24px;font-weight:700;color:var(--warn)">${suspCount}</div>
            <div style="font-size:9px;color:var(--muted);letter-spacing:1.5px">SUSPICIOUS</div>
          </div>
          <div style="background:rgba(29,219,138,0.1);border:1px solid rgba(29,219,138,0.3);border-radius:10px;padding:14px 20px;text-align:center;flex:1;min-width:80px">
            <div style="font-family:var(--mono);font-size:24px;font-weight:700;color:var(--safe)">${safeCount}</div>
            <div style="font-size:9px;color:var(--muted);letter-spacing:1.5px">SAFE</div>
          </div>
        </div>
        <div class="indicators-title">BATCH RESULTS — ${results.length} URLs</div>
        ${results.sort((a, b) => b.score - a.score).map(r => `
          <div class="batch-row">
            <span class="v-pill" style="background:${bgColor(r.verdict)};border:1px solid ${borderColor(r.verdict)};color:${r.verdict.color}">${r.verdict.label}</span>
            <span style="font-family:var(--mono);font-weight:700;font-size:15px;color:${r.verdict.color};min-width:36px">${r.score}</span>
            <span style="font-family:var(--mono);font-size:11px;color:var(--muted);flex:1;word-break:break-all">${r.url}</span>
          </div>
        `).join('')}
      </div>
    `;

    btn.disabled    = false;
    btn.textContent = `⚡ SCAN ${lines.length} URLs`;
    updateCounter();
  }, 900);
}

// ─── ML Backend Integration ──────────────────────────────────────────────────

/**
 * Scan a URL using the Python ML backend
 */
async function scanWithML() {
  const url      = document.getElementById('ml-url-input').value.trim();
  const endpoint = document.getElementById('ml-endpoint').value.trim();
  if (!url) return;

  const btn       = document.getElementById('ml-btn');
  const container = document.getElementById('ml-result');
  const status    = document.getElementById('ml-status');

  btn.disabled    = true;
  btn.textContent = '⏳ Running ML…';
  container.innerHTML = `<div class="card scanning"><div class="spinner"></div>Calling Python ML backend…</div>`;

  try {
    const response = await fetch(endpoint, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ url }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();

    // data shape from app.py:
    // { score, verdict, confidence, reasons, features, importances, model_name }
    const score      = Math.round(data.score * 100);
    const confidence = Math.round((data.probability || data.confidence || 0.5) * 100);
    const reasons    = data.reasons || [];
    const features   = data.features || {};

    renderResult(container, {
      score,
      reasons,
      confidence,
      features: {
        url_length:    features.url_length || 0,
        has_https:     features.has_https || false,
        domain_entropy: (features.domain_entropy || 0).toFixed(3),
        keyword_count: features.keyword_count || 0,
        is_ip:         features.is_ip || false,
        suspicious_tld: features.suspicious_tld || false,
      },
      urlResults: [],
      ml_details: {
        features:    data.features,
        importances: data.importances,
        model_name:  data.model_name,
      }
    });

    status.className = 'ml-status ok';
    status.textContent = `✓ ML backend responded — model: ${data.model_name || 'RandomForest'}`;
    status.classList.remove('hidden');

  } catch (err) {
    container.innerHTML = `<div class="error">
      ❌ Could not reach ML backend at <code>${endpoint}</code><br><br>
      Make sure you have run: <code>python app.py</code><br>
      Error: ${err.message}
    </div>`;
    status.className = 'ml-status fail';
    status.textContent = `✗ Backend offline — ${err.message}`;
    status.classList.remove('hidden');
  }

  btn.disabled    = false;
  btn.textContent = '🤖 ML SCAN';
}

/**
 * Ping the ML backend health endpoint
 */
async function checkMLHealth() {
  const endpoint = document.getElementById('ml-endpoint').value.replace('/predict', '/health');
  const status   = document.getElementById('ml-status');

  status.className    = 'ml-status';
  status.textContent  = '⏳ Pinging…';
  status.classList.remove('hidden');

  try {
    const res = await fetch(endpoint, { method: 'GET' });
    const data = await res.json();
    status.className   = 'ml-status ok';
    status.textContent = `✓ Backend online — ${JSON.stringify(data)}`;
  } catch (e) {
    status.className   = 'ml-status fail';
    status.textContent = `✗ Backend offline — ${e.message}`;
  }
}

// ─── UI Helpers ──────────────────────────────────────────────────────────────

function switchTab(id, el) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  el.classList.add('active');
}

function setURL(u) {
  document.getElementById('url-input').value = u;
  document.getElementById('url-input').focus();
}

function loadSample() {
  document.getElementById('email-input').value = `Subject: URGENT: Your Account Has Been Suspended

Dear Customer,

Your Amazon account has been suspended due to unauthorized activity. Click here immediately to verify your account:

http://amazon-security-alert.tk/verify?token=xK9a2mP&ref=account

You must act now or your account will be permanently deleted within 24 hours! This is your final notice.

Update your billing information here to avoid suspension:
http://paypa1-update.ml/payment?confirm=1&user=12345

We have detected multiple violations. Urgent action required!

Do not ignore this email. Your account expires today!!!

Regards,
Amazon Security Team`;
}

// ─── Counter Animation ───────────────────────────────────────────────────────

let scanCount = 18472;

function updateCounter() {
  scanCount += Math.floor(Math.random() * 3) + 1;
  const el = document.getElementById('s1');
  if (el) el.textContent = scanCount.toLocaleString();
}

window.addEventListener('load', () => {
  let n = 0;
  const target = 18472;
  const step   = Math.ceil(target / 60);
  const iv     = setInterval(() => {
    n = Math.min(n + step, target);
    document.getElementById('s1').textContent = n.toLocaleString();
    if (n >= target) clearInterval(iv);
  }, 16);
});

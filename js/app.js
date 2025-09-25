/* -----------------------------
     نمونهٔ دیتابیس (قابل جایگزینی با فایل JSON)
     هر رکورد می‌تواند هر فیلدی داشته باشد؛ الگوریتم روی همهٔ متن‌ها کار می‌کند.
     ----------------------------- */
  const SAMPLE_DB = [
    { id: 1, title: "بازیابی رمز عبور", content: "برای بازیابی رمز، روی فراموشی رمز کلیک کنید و ایمیل خود را وارد نمایید. لینک بازیابی ارسال می‌شود." , tags: ["رمز", "بازیابی"] },
    { id: 2, title: "راهنمای نصب محصول", content: "مراحل نصب نرم‌افزار: دانلود بسته، اجرای نصب‌کننده، وارد کردن کلید لایسنس، پیکربندی اولیه.", tags: ["نصب", "راهنما"] },
    { id: 3, title: "ثبت‌نام و احراز هویت", content: "برای ساخت حساب جدید به صفحه ثبت‌نام بروید و مشخصات را وارد کنید. ایمیل را تایید کنید.", tags: ["ثبت‌نام", "حساب"] },
    { id: 4, title: "سوالات متداول درباره پرداخت", content: "پرداخت‌ها امن هستند. فاکتور پس از پرداخت ایمیل می‌شود. روش‌های پرداخت: کارت، درگاه، کیف‌پول." , tags: ["پرداخت", "فاکتور"] },
    { id: 5, title: "مشکلات ورود", content: "اگر با خطای ورود مواجه شدید، کش مرورگر را پاک کنید یا از بازیابی رمز استفاده کنید. پشتیبانی تماس بگیرید." , tags: ["ورود", "خطا"] }
  ];

  /* ------------- متن‌پردازی و توکنایز -------------- */
  function normalizeText(str){
    if(!str) return "";
    // تبدیل حروف معمول و حذف نویسه‌های غیرضروری
    str = String(str).toLowerCase();
    // چند تبدیل مخصوص فارسی (نمونه)
    str = str.replace(/آ/g,'ا').replace(/أ|إ/g,'ا').replace(/ي/g,'ی').replace(/ئ/g,'ی').replace(/ۀ/g,'ه');
    // حذف علامت‌ها
    str = str.replace(/[،،\.,\/#!$%\^&\*;:{}=\-_`~()؟\?«»"']/g,' ');
    // حذف اضافی فضا
    str = str.replace(/\s+/g,' ').trim();
    return str;
  }

  function tokenize(text){
    text = normalizeText(text);
    // گرفتن توکن‌های یونیکد حروف و عدد (هر زبان)
    const tokens = Array.from(text.matchAll(/[\p{L}\p{N}_]+/gu)).map(m=>m[0]);
    return tokens;
  }

  /* ------------- ساخت TF-IDF -------------- */
  let DB = JSON.parse(JSON.stringify(SAMPLE_DB)); // clone
  let vocab = new Map(); // token -> index
  let docVectors = []; // array of {tfidf: Map(token, value), rawText, doc}
  let idf = new Map();
  function buildIndex(db){
    DB = db;
    vocab = new Map();
    idf = new Map();
    const docTokensList = [];

    // جمع‌آوری توکن‌ها و TF
    DB.forEach((doc, di) => {
      const text = Object.values(doc).filter(v=>typeof v === 'string' || typeof v === 'number').join(' ');
      const tokens = tokenize(text);
      docTokensList.push(tokens);
      const seen = new Set();
      tokens.forEach(t => {
        if(!vocab.has(t)) vocab.set(t, vocab.size);
        if(!seen.has(t)){ idf.set(t, (idf.get(t)||0)+1); seen.add(t); }
      });
    });

    // تبدیل idf به لگاریتمی
    const N = DB.length;
    for(const [t,df] of idf.entries()){
      idf.set(t, Math.log((N)/(1+df)) ); // smoothed
    }

    // ساخت بردارهای TF-IDF
    docVectors = DB.map((doc, di) => {
      const text = Object.values(doc).filter(v=>typeof v === 'string' || typeof v === 'number').join(' ');
      const tokens = docTokensList[di];
      const tf = new Map();
      tokens.forEach(t => tf.set(t, (tf.get(t)||0) + 1));
      // نرمال‌سازی TF
      const tfidf = new Map();
      let norm = 0;
      for(const [t,count] of tf.entries()){
        const val = (count / tokens.length) * (idf.get(t) || 0.0);
        tfidf.set(t, val);
        norm += val*val;
      }
      norm = Math.sqrt(norm) || 1;
      // نرمال‌سازی بردار
      for(const k of tfidf.keys()) tfidf.set(k, tfidf.get(k)/norm);
      return { doc, rawText: text, tfidf, tokens };
    });
  }

  /* ------------- شباهت کسینوسی -------------- */
  function cosineSim(mapA, mapB){
    // maps token -> value (sparse)
    let sum = 0;
    // iterate smaller map
    if(mapA.size < mapB.size){
      for(const [k,v] of mapA.entries()){
        if(mapB.has(k)) sum += v * mapB.get(k);
      }
    } else {
      for(const [k,v] of mapB.entries()){
        if(mapA.has(k)) sum += v * mapA.get(k);
      }
    }
    return sum; // since vectors normalized -> بین 0 و 1
  }

  /* ------------- فاصله لِونشتاین ساده (فازی) -------------- */
  function levenshtein(a,b){
    if(a===b) return 0;
    const al = a.length, bl = b.length;
    if(al === 0) return bl;
    if(bl === 0) return al;
    let v0 = new Array(bl+1).fill(0).map((_,i)=>i);
    let v1 = new Array(bl+1).fill(0);
    for(let i=0;i<al;i++){
      v1[0] = i+1;
      for(let j=0;j<bl;j++){
        const cost = a[i] === b[j] ? 0 : 1;
        v1[j+1] = Math.min(v1[j]+1, v0[j+1]+1, v0[j]+cost);
      }
      [v0,v1] = [v1,v0];
    }
    return v0[bl];
  }

  /* ------------- تبدیل کوئری به بردار و امتیازدهی -------------- */
  function queryVector(query){
    const tokens = tokenize(query);
    const tf = new Map();
    tokens.forEach(t => tf.set(t, (tf.get(t)||0)+1));
    const tfidf = new Map();
    let norm = 0;
    for(const [t,count] of tf.entries()){
      const val = (count / tokens.length) * (idf.get(t) || 0.0);
      tfidf.set(t, val);
      norm += val*val;
    }
    norm = Math.sqrt(norm) || 1;
    for(const k of tfidf.keys()) tfidf.set(k, tfidf.get(k)/norm);
    return { tokens, tfidf };
  }

  /* ------------- نیت‌شناس ساده -------------- */
  const INTENT_KEYWORDS = {
    "ثبت‌نام": ["ثبت‌نام","عضویت","حساب جدید","ثبت نام","ایجاد حساب"],
    "ورود": ["ورود","لاگین","login","sign in","وارد شدن"],
    "بازیابی رمز": ["بازیابی","فراموشی رمز","فراموشی","رمز عبور","reset password"],
    "نصب": ["نصب","install","راه‌اندازی","راه اندازی","setup"],
    "پرداخت": ["پرداخت","فاکتور","صورتحساب","هزینه","پرداختی"]
  };

  function detectIntent(query){
    const q = normalizeText(query);
    const scores = {};
    for(const [intent, kws] of Object.entries(INTENT_KEYWORDS)){
      for(const kw of kws){
        if(q.includes(normalizeText(kw))) scores[intent] = (scores[intent]||0) + 2;
      }
    }
    // کلمات توکنی هم وزن می‌گیرند
    const tokens = tokenize(q);
    for(const [intent,kws] of Object.entries(INTENT_KEYWORDS)){
      for(const t of tokens){
        for(const kw of kws){
          if(normalizeText(kw).includes(t) || t.includes(normalizeText(kw))) scores[intent] = (scores[intent]||0)+0.5;
        }
      }
    }
    // انتخاب بهترین
    const entries = Object.entries(scores);
    if(entries.length === 0) return null;
    entries.sort((a,b)=>b[1]-a[1]);
    return entries[0][0];
  }

  /* ------------- اجرای جستجو -------------- */
  function search(query, mode='combined', topK=10){
    const qv = queryVector(query);
    const results = [];
    DB.forEach((doc, idx) => {
      const dv = docVectors[idx];
      // cosine
      const cos = cosineSim(qv.tfidf, dv.tfidf) || 0;
      // fuzzy: حداقل فاصله لِونشتاین بین توکن‌های query و توکن‌های doc
      let bestLevScore = 0;
      for(const qt of qv.tokens){
        for(const dt of dv.tokens){
          // فقط برای رشته‌های کوتاه/متوسط
          const lev = levenshtein(qt, dt);
          const maxLen = Math.max(qt.length, dt.length, 1);
          const score = 1 - (lev / maxLen); // 1 بهترین
          if(score > bestLevScore) bestLevScore = score;
        }
      }
      // exact phrase boost
      const phrase = normalizeText(query);
      const exactBoost = dv.rawText.includes(phrase) ? 0.35 : 0;
      // ترکیب وزن‌ها
      let score = 0;
      if(mode === 'tfidf') score = cos;
      else if(mode === 'fuzzy') score = bestLevScore;
      else score = (cos * 0.7) + (bestLevScore * 0.25) + exactBoost;

      // clamp
      score = Math.max(0, Math.min(1, score));
      results.push({ doc, score, cos, fuzzy: bestLevScore, exactBoost });
    });

    results.sort((a,b)=>b.score - a.score);
    return results.slice(0, topK);
  }

  /* ------------- های‌لایت کلمات در متن -------------- */
  function highlightMatches(text, query){
    const tokens = Array.from(new Set(tokenize(query)));
    if(tokens.length === 0) return escapeHtml(text);
    // مرتب کن بر اساس طول برای جلوگیری از برش‌های ناخواسته
    tokens.sort((a,b)=>b.length - a.length);
    let out = escapeHtml(text);
    for(const t of tokens){
      if(t.length < 1) continue;
      const re = new RegExp(escapeRegExp(t), 'gi');
      out = out.replace(re, match => `<span class="highlight">${match}</span>`);
    }
    return out;
  }

  /* ------------- کمکی‌ها -------------- */
  function escapeHtml(unsafe){
    return String(unsafe)
      .replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  }
  function escapeRegExp(s){ return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

  /* ------------- UI binding -------------- */
  const queryInput = document.getElementById('query');
  const searchBtn = document.getElementById('searchBtn');
  const resultsEl = document.getElementById('results');
  const resultCount = document.getElementById('resultCount');
  const detectedIntentEl = document.getElementById('detectedIntent');
  const explainBtn = document.getElementById('explainBtn');
  const showDbBtn = document.getElementById('showDbBtn');
  const jsonFile = document.getElementById('jsonFile');
  const resetDbBtn = document.getElementById('resetDb');
  const modeBtns = document.querySelectorAll('.modeBtn');
  const modeLabel = document.getElementById('modeLabel');

  let currentMode = 'combined';
  modeBtns.forEach(b=> b.addEventListener('click', (e)=>{
    modeBtns.forEach(x=> x.classList.remove('bg-violet-600','text-white'));
    modeBtns.forEach(x=> x.classList.add('bg-slate-800','text-slate-200'));
    b.classList.remove('bg-slate-800','text-slate-200');
    b.classList.add('bg-violet-600','text-white');
    currentMode = b.getAttribute('data-mode');
    modeLabel.innerText = currentMode === 'combined' ? 'ترکیبی (TF-IDF + فازی)' : (currentMode==='tfidf' ? 'TF-IDF' : 'فازی');
  }));

  searchBtn.addEventListener('click', runSearch);
  queryInput.addEventListener('keydown', (e)=>{ if(e.key === 'Enter') runSearch(); });

  explainBtn.addEventListener('click', ()=>{
    // فقط نمایش باکس‌های توضیحی برای هر نتیجه
    document.querySelectorAll('.explain').forEach(el => el.classList.toggle('hidden'));
  });

  showDbBtn.addEventListener('click', ()=>{
    const pretty = JSON.stringify(DB, null, 2);
    const w = window.open();
    w.document.write('<pre style="direction:ltr">'+escapeHtml(pretty)+'</pre>');
  });

  jsonFile.addEventListener('change', (e)=>{
    const f = e.target.files?.[0];
    if(!f) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try{
        const parsed = JSON.parse(ev.target.result);
        if(!Array.isArray(parsed)) throw new Error('فایل JSON باید آرایه‌ای از اشیا باشد.');
        buildIndex(parsed);
        alert('JSON بارگذاری و ایندکس‌سازی شد. حالا جستجو کن.');
      } catch(err){
        alert('خطا در خواندن JSON: ' + err.message);
      }
    };
    reader.readAsText(f, 'utf-8');
  });

  resetDbBtn.addEventListener('click', ()=>{
    DB = JSON.parse(JSON.stringify(SAMPLE_DB));
    buildIndex(DB);
    alert('دیتابیس به نمونهٔ پیش‌فرض بازگردانده شد.');
  });

  function runSearch(){
    const q = queryInput.value.trim();
    if(!q) { resultsEl.innerHTML = '<div class="text-slate-400 text-sm">لطفاً یک عبارت جستجو وارد کنید.</div>'; resultCount.innerText = '0'; detectedIntentEl.innerText = '—'; return; }
    const intent = detectIntent(q);
    detectedIntentEl.innerText = intent || 'نامشخص';
    const res = search(q, currentMode, 12);
    resultCount.innerText = res.length;
    resultsEl.innerHTML = '';
    if(res.length === 0) {
      resultsEl.innerHTML = `<div class="text-slate-400 text-sm">نتیجه‌ای نیافت شد.</div>`;
      return;
    }
    res.forEach(r=>{
      const scorePct = Math.round(r.score*100);
      const card = document.createElement('div');
      card.className = 'p-4 rounded-xl border border-transparent hover:border-violet-500 transition flex flex-col';
      card.innerHTML = `
        <div class="flex items-start gap-3">
          <div class="flex-1">
            <div class="text-sm text-slate-300">${escapeHtml(r.doc.title || r.doc.name || 'بدون عنوان')}</div>
            <div class="mt-2 text-xs text-slate-200 leading-relaxed">${highlightMatches(r.doc.content || r.doc.description || '', queryInput.value)}</div>
            <div class="mt-3 flex items-center gap-2 text-xs text-slate-400">
              <div>امتیاز: <span class="font-medium text-white">${scorePct}%</span></div>
              <div>|</div>
              <div>شباهت TF-IDF: ${Math.round(r.cos*100)}%</div>
              <div>|</div>
              <div>فازی: ${Math.round(r.fuzzy*100)}%</div>
            </div>
          </div>
          <div class="w-24 text-right">
            <div class="text-xs text-slate-400">id: ${r.doc.id ?? '-'}</div>
            <div class="mt-3 text-xs"><button data-id="${r.doc.id}" class="copyBtn px-2 py-1 rounded-md bg-slate-800">کپی محتوا</button></div>
          </div>
        </div>
        <div class="mt-3 explain hidden text-xs text-slate-300 bg-slate-900 p-3 rounded-md">
          <div>دلایل امتیازدهی:</div>
          <ul class="mt-2 list-disc list-inside">
            <li>کسینوس TF-IDF: ${ (r.cos).toFixed(3) }</li>
            <li>شباهت فازی (Levenshtein): ${ (r.fuzzy).toFixed(3) }</li>
            <li>تقویت عبارات دقیق: ${ r.exactBoost.toFixed(3) }</li>
          </ul>
        </div>
      `;
      resultsEl.appendChild(card);
    });

    // copy handlers
    document.querySelectorAll('.copyBtn').forEach(b=>{
      b.addEventListener('click', (e)=>{
        const id = b.getAttribute('data-id');
        const doc = DB.find(d=>String(d.id) === String(id));
        if(doc){ navigator.clipboard.writeText(JSON.stringify(doc,null,2)); alert('محتوا کپی شد'); }
      });
    });
  }

  // index initial
  buildIndex(DB);

  // نمونهٔ پیش‌جستجو برای نمایش
  queryInput.value = "نحوهٔ بازیابی رمز عبور";

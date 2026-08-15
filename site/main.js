/* memoranda project page — theme toggle, sticky nav, lightbox. No dependencies. */
(function () {
  'use strict';

  /* ---------- theme ---------- */
  var root = document.documentElement;
  var STORAGE_KEY = 'memoranda-theme';

  function systemTheme() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function currentTheme() {
    return root.getAttribute('data-theme') || systemTheme();
  }
  function applyTheme(theme, persist) {
    root.setAttribute('data-theme', theme);
    if (persist) {
      try { localStorage.setItem(STORAGE_KEY, theme); } catch (e) { /* private mode */ }
    }
    var buttons = document.querySelectorAll('.theme-toggle');
    for (var i = 0; i < buttons.length; i++) {
      var b = buttons[i];
      var label = theme === 'dark' ? 'Light' : 'Dark';
      b.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
      b.setAttribute('aria-label', 'Switch to ' + label.toLowerCase() + ' theme');
      var span = b.querySelector('.label');
      if (span) span.textContent = label;
    }
  }
  (function initTheme() {
    var stored = null;
    try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) { /* ignore */ }
    // ?theme=light|dark previews a theme without persisting it.
    var m = /[?&]theme=(light|dark)\b/.exec(window.location.search);
    if (m) stored = m[1];
    if (stored === 'dark' || stored === 'light') {
      applyTheme(stored, false);
    } else {
      // Do not stamp the attribute so prefers-color-scheme keeps driving; still label the button.
      applyTheme(systemTheme(), false);
      root.removeAttribute('data-theme');
    }
    var toggles = document.querySelectorAll('.theme-toggle');
    for (var i = 0; i < toggles.length; i++) {
      toggles[i].addEventListener('click', function () {
        applyTheme(currentTheme() === 'dark' ? 'light' : 'dark', true);
      });
    }
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      var onChange = function () {
        var s = null;
        try { s = localStorage.getItem(STORAGE_KEY); } catch (e) { /* ignore */ }
        if (!s) { applyTheme(systemTheme(), false); root.removeAttribute('data-theme'); }
      };
      if (mq.addEventListener) mq.addEventListener('change', onChange);
      else if (mq.addListener) mq.addListener(onChange);
    }
  })();

  /* ---------- sticky nav ---------- */
  var nav = document.querySelector('.sitenav');
  var masthead = document.querySelector('.masthead');
  function updateNav() {
    if (!nav || !masthead) return;
    var threshold = masthead.getBoundingClientRect().bottom + window.scrollY - 8;
    if (window.scrollY > threshold) nav.classList.add('is-visible');
    else nav.classList.remove('is-visible');
  }
  window.addEventListener('scroll', updateNav, { passive: true });
  window.addEventListener('resize', updateNav);
  updateNav();

  /* ---------- lightbox ---------- */
  var lb = document.getElementById('lightbox');
  if (!lb) return;
  var lbImg = lb.querySelector('img');
  var lbCap = lb.querySelector('.lb-cap');
  var btnClose = lb.querySelector('.lb-close');
  var btnPrev = lb.querySelector('.lb-prev');
  var btnNext = lb.querySelector('.lb-next');
  var links = Array.prototype.slice.call(document.querySelectorAll('a.figlink'));
  var index = -1;
  var lastFocus = null;

  function captionFor(link) {
    var explicit = link.getAttribute('data-caption');
    if (explicit) return explicit;
    var img = link.querySelector('img');
    return img ? img.getAttribute('alt') || '' : '';
  }
  function show(i) {
    if (i < 0) i = links.length - 1;
    if (i >= links.length) i = 0;
    index = i;
    var link = links[index];
    var img = link.querySelector('img');
    lbImg.src = link.getAttribute('href');
    lbImg.alt = img ? img.getAttribute('alt') || '' : '';
    lbCap.textContent = (index + 1) + ' / ' + links.length + ' — ' + captionFor(link);
  }
  function open(i) {
    lastFocus = document.activeElement;
    lb.classList.add('is-open');
    lb.setAttribute('aria-hidden', 'false');
    document.body.classList.add('lb-locked');
    show(i);
    btnClose.focus();
  }
  function close() {
    lb.classList.remove('is-open');
    lb.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('lb-locked');
    lbImg.removeAttribute('src');
    if (lastFocus && typeof lastFocus.focus === 'function') lastFocus.focus();
  }

  links.forEach(function (link, i) {
    link.addEventListener('click', function (ev) {
      if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button === 1) return; // let new-tab work
      ev.preventDefault();
      open(i);
    });
  });
  btnClose.addEventListener('click', close);
  btnPrev.addEventListener('click', function () { show(index - 1); });
  btnNext.addEventListener('click', function () { show(index + 1); });
  lb.addEventListener('click', function (ev) {
    if (ev.target === lb) close(); // backdrop click
  });
  document.addEventListener('keydown', function (ev) {
    if (!lb.classList.contains('is-open')) return;
    if (ev.key === 'Escape') { ev.preventDefault(); close(); return; }
    if (ev.key === 'ArrowLeft') { ev.preventDefault(); show(index - 1); return; }
    if (ev.key === 'ArrowRight') { ev.preventDefault(); show(index + 1); return; }
    if (ev.key === 'Tab') {
      // keep focus inside the dialog
      var focusables = [btnClose, btnPrev, btnNext];
      var first = focusables[0], last = focusables[focusables.length - 1];
      if (ev.shiftKey && document.activeElement === first) { ev.preventDefault(); last.focus(); }
      else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first.focus(); }
      else if (focusables.indexOf(document.activeElement) === -1) { ev.preventDefault(); first.focus(); }
    }
  });
})();

/* Alliance Briefing : interactions et animations de l'interface.
 *
 * Les effets reprennent des composants React Bits (reactbits.dev), réécrits en JavaScript
 * natif pour rester dans les pages Jinja sans chaîne de build :
 *   - Orb          : sphère WebGL de l'en-tête (shader d'origine, couleurs ajustées) ;
 *   - BlurText     : apparition mot à mot du message d'accueil ;
 *   - CountUp      : chiffres qui défilent jusqu'à leur valeur ;
 *   - SpotlightCard: halo qui suit le curseur sur les cartes (voir app.css).
 * Sans JavaScript, la page reste complète : seuls les effets et les panneaux disparaissent.
 */
(() => {
  "use strict";
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.documentElement.classList.add("js");

  /* ---------- Orb (React Bits) ---------- */
  const VERT = `
    attribute vec2 position; varying vec2 vUv;
    void main() { vUv = position * 0.5 + 0.5; gl_Position = vec4(position, 0.0, 1.0); }`;
  const FRAG = `
    precision highp float;
    uniform float iTime; uniform vec3 iResolution; uniform float hue; uniform float hover; uniform float rot;
    varying vec2 vUv;
    vec3 rgb2yiq(vec3 c) { return vec3(dot(c, vec3(.299, .587, .114)), dot(c, vec3(.596, -.274, -.322)), dot(c, vec3(.211, -.523, .312))); }
    vec3 yiq2rgb(vec3 c) { return vec3(c.x + .956 * c.y + .621 * c.z, c.x - .272 * c.y - .647 * c.z, c.x - 1.106 * c.y + 1.703 * c.z); }
    vec3 adjustHue(vec3 color, float deg) {
      float r = deg * 3.14159265 / 180.0; vec3 yiq = rgb2yiq(color);
      float cA = cos(r), sA = sin(r);
      return yiq2rgb(vec3(yiq.x, yiq.y * cA - yiq.z * sA, yiq.y * sA + yiq.z * cA));
    }
    vec3 hash33(vec3 p3) {
      p3 = fract(p3 * vec3(.1031, .11369, .13787)); p3 += dot(p3, p3.yxz + 19.19);
      return -1.0 + 2.0 * fract(vec3(p3.x + p3.y, p3.x + p3.z, p3.y + p3.z) * p3.zyx);
    }
    float snoise3(vec3 p) {
      const float K1 = .333333333, K2 = .166666667;
      vec3 i = floor(p + (p.x + p.y + p.z) * K1); vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
      vec3 e = step(vec3(0.0), d0 - d0.yzx); vec3 i1 = e * (1.0 - e.zxy); vec3 i2 = 1.0 - e.zxy * (1.0 - e);
      vec3 d1 = d0 - (i1 - K2); vec3 d2 = d0 - (i2 - K1); vec3 d3 = d0 - .5;
      vec4 h = max(.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
      vec4 n = h * h * h * h * vec4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));
      return dot(vec4(31.316), n);
    }
    vec4 extractAlpha(vec3 c) { float a = max(max(c.r, c.g), c.b); return vec4(c / (a + 1e-5), a); }
    const vec3 base1 = vec3(.611765, .262745, .996078);
    const vec3 base2 = vec3(.298039, .760784, .913725);
    const vec3 base3 = vec3(.062745, .078431, .600000);
    const float innerRadius = .6; const float noiseScale = .65;
    float light1(float i, float a, float d) { return i / (1.0 + d * a); }
    float light2(float i, float a, float d) { return i / (1.0 + d * d * a); }
    vec4 draw(vec2 uv) {
      vec3 c1 = adjustHue(base1, hue), c2 = adjustHue(base2, hue), c3 = adjustHue(base3, hue);
      float ang = atan(uv.y, uv.x); float len = length(uv); float invLen = len > 0.0 ? 1.0 / len : 0.0;
      float n0 = snoise3(vec3(uv * noiseScale, iTime * .5)) * .5 + .5;
      float r0 = mix(mix(innerRadius, 1.0, .4), mix(innerRadius, 1.0, .6), n0);
      float d0 = distance(uv, (r0 * invLen) * uv);
      float v0 = light1(1.0, 10.0, d0); v0 *= smoothstep(r0 * 1.05, r0, len);
      float cl = cos(ang + iTime * 2.0) * .5 + .5;
      float a = iTime * -1.0; vec2 pos = vec2(cos(a), sin(a)) * r0; float d = distance(uv, pos);
      float v1 = light2(1.5, 5.0, d); v1 *= light1(1.0, 50.0, d0);
      float v2 = smoothstep(1.0, mix(innerRadius, 1.0, n0 * .5), len);
      float v3 = smoothstep(innerRadius, mix(innerRadius, 1.0, .5), len);
      vec3 col = mix(c1, c2, cl); col = mix(c3, col, v0); col = (col + v1) * v2 * v3;
      return extractAlpha(clamp(col, 0.0, 1.0));
    }
    void main() {
      vec2 frag = vUv * iResolution.xy; vec2 center = iResolution.xy * .5; float size = min(iResolution.x, iResolution.y);
      vec2 uv = (frag - center) / size * 2.0;
      float s = sin(rot), c = cos(rot); uv = vec2(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
      uv.x += hover * .1 * sin(uv.y * 10.0 + iTime); uv.y += hover * .1 * sin(uv.x * 10.0 + iTime);
      vec4 col = draw(uv); gl_FragColor = vec4(col.rgb * col.a, col.a);
    }`;

  const orbs = [];
  function mountOrb(host) {
    const canvas = host.querySelector("canvas");
    const gl = canvas && canvas.getContext("webgl", { premultipliedAlpha: true, alpha: true, antialias: true });
    if (!gl) return;
    const shader = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const prog = gl.createProgram();
    gl.attachShader(prog, shader(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "position");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const u = (n) => gl.getUniformLocation(prog, n);
    const U = { time: u("iTime"), res: u("iResolution"), hue: u("hue"), hover: u("hover"), rot: u("rot") };
    gl.uniform1f(U.hue, -12);

    const orb = { host, hover: 0, target: 0, rot: 0, speed: 1, boost: 0 };
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = canvas.clientHeight * dpr;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform3f(U.res, canvas.width, canvas.height, canvas.width / canvas.height);
    };
    resize();
    new ResizeObserver(resize).observe(canvas);
    host.parentElement.addEventListener("pointermove", (ev) => {
      const r = host.getBoundingClientRect();
      const x = (ev.clientX - r.left) / r.width - 0.5, y = (ev.clientY - r.top) / r.height - 0.5;
      orb.target = Math.hypot(x, y) < 0.4 ? 1 : 0;
    });
    host.parentElement.addEventListener("pointerleave", () => { orb.target = 0; });

    let last = performance.now(), t = 0;
    const frame = (now) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      orb.hover += (Math.max(orb.target, orb.boost) - orb.hover) * 0.08;
      orb.rot += dt * (0.3 * orb.hover + orb.boost * 2);
      t += dt * (1 + orb.boost * 2.5);
      gl.uniform1f(U.time, t);
      gl.uniform1f(U.hover, orb.hover * 0.5);
      gl.uniform1f(U.rot, orb.rot);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (!reduced) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
    orbs.push(orb);
  }

  /* ---------- BlurText (React Bits) ---------- */
  function blurText(el) {
    const words = el.textContent.trim().split(/\s+/);
    el.setAttribute("aria-label", el.textContent.trim());
    el.textContent = "";
    words.forEach((word, i) => {
      const span = document.createElement("span");
      span.className = "word";
      span.setAttribute("aria-hidden", "true");
      span.textContent = i < words.length - 1 ? word + " " : word;
      el.append(span);
      span.animate(
        [
          { filter: "blur(12px)", opacity: 0, transform: "translateY(-14px)" },
          { filter: "blur(4px)", opacity: 0.5, transform: "translateY(3px)", offset: 0.5 },
          { filter: "blur(0)", opacity: 1, transform: "none" },
        ],
        { duration: 700, delay: 120 + i * 140, easing: "cubic-bezier(.2,.8,.2,1)", fill: "both" }
      );
    });
  }

  /* ---------- CountUp (React Bits) ---------- */
  function countUp(el) {
    const to = Number(el.dataset.count);
    if (!Number.isFinite(to) || to === 0) return;
    const duration = 1400, start = performance.now() + 300;
    el.textContent = "0";
    const tick = (now) => {
      const p = Math.min(Math.max((now - start) / duration, 0), 1);
      el.textContent = Math.round(to * (1 - Math.pow(1 - p, 3))).toLocaleString("fr-FR");
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    // Valeur finale garantie, même si l'onglet est en arrière-plan (rAF suspendu).
    setTimeout(() => { el.textContent = to.toLocaleString("fr-FR"); }, duration + 600);
  }

  /* ---------- Panneaux de détail ---------- */
  function openDrawer(id) {
    const dialog = document.getElementById(id);
    if (!dialog || dialog.open) return;
    dialog.showModal();
    dialog.scrollTop = 0;
  }
  function closeDrawer(dialog) {
    if (reduced) return dialog.close();
    dialog.classList.add("is-closing");
    dialog.addEventListener("animationend", () => { dialog.classList.remove("is-closing"); dialog.close(); }, { once: true });
  }

  document.addEventListener("click", (ev) => {
    const opener = ev.target.closest("[data-open]");
    if (opener) { ev.preventDefault(); return openDrawer(opener.dataset.open); }
    const closer = ev.target.closest("[data-close]");
    if (closer) return closeDrawer(closer.closest("dialog"));
    if (ev.target.matches("dialog.drawer")) closeDrawer(ev.target); // clic sur le fond
  });
  document.addEventListener("cancel", (ev) => {
    if (ev.target.matches("dialog.drawer")) { ev.preventDefault(); closeDrawer(ev.target); }
  }, true);

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-orb]").forEach(mountOrb);

    // Halo des cartes (SpotlightCard).
    document.querySelectorAll("[data-spotlight]").forEach((card, i) => {
      card.style.setProperty("--i", i);
      card.addEventListener("pointermove", (ev) => {
        const r = card.getBoundingClientRect();
        card.style.setProperty("--mx", `${ev.clientX - r.left}px`);
        card.style.setProperty("--my", `${ev.clientY - r.top}px`);
      });
    });

    // Filtres de « L'essentiel du jour » : n'affiche que la rubrique choisie, sans changer l'ordre.
    document.querySelectorAll(".filters").forEach((group) => {
      const card = group.closest(".card");
      group.addEventListener("click", (ev) => {
        const btn = ev.target.closest("[data-filter]");
        if (!btn || btn.disabled) return;
        group.querySelectorAll("[data-filter]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
        card.querySelectorAll("[data-filtered] > .entry").forEach((li) => {
          const section = li.querySelector("[data-section]").dataset.section;
          li.hidden = btn.dataset.filter !== "all" && section !== btn.dataset.filter;
        });
      });
    });

    // Changer de jour dès qu'une date est choisie.
    document.querySelectorAll("[data-autosubmit]").forEach((input) => {
      input.addEventListener("change", () => input.value && input.form.requestSubmit());
    });

    // Génération : bouton occupé, sphère qui s'accélère.
    document.querySelectorAll("form[data-generate]").forEach((form) => {
      form.addEventListener("submit", () => {
        const btn = form.querySelector("button");
        btn.setAttribute("aria-busy", "true");
        const label = btn.querySelector("[data-label]");
        if (label) label.textContent = "Analyse en cours…";
        orbs.forEach((o) => { o.boost = 1; });
      });
    });

    // Séquence d'entrée unique : message d'accueil, puis cartes et chiffres.
    if (!reduced) {
      document.querySelectorAll("[data-split]").forEach(blurText);
      document.querySelectorAll("[data-count]").forEach(countUp);
    }
    requestAnimationFrame(() => document.body.classList.add("is-ready"));
  });
})();

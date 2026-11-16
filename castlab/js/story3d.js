/* Landing story: one 3D crankbait that changes with scroll position across five chapters.
   Built procedurally (no model files). Decorative; the data shown in the app is never faked here.
   Honours prefers-reduced-motion (no idle motion, scroll still moves the lure). */
import * as THREE from 'three';
import { RoomEnvironment } from '../vendor/RoomEnvironment.js';

const host = document.getElementById('authView'), canvas = host && host.querySelector('.story3d');
if (canvas) { try { start(); } catch (e) { canvas.remove(); } }

function start() {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05; renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100); camera.position.set(0, 0, 10);
  const pm = new THREE.PMREMGenerator(renderer); scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  const key = new THREE.DirectionalLight(0xfff1dc, 2.2); key.position.set(4, 6, 6); scene.add(key);
  const rim = new THREE.DirectionalLight(0x9fd0ff, 1.4); rim.position.set(-6, 2, -4); scene.add(rim);

  /* ---------- paint patterns (canvas textures). u: around the body (0.75 = back, 0.25 = belly), v: tail -> nose ---------- */
  function paint(kind) {
    const W = 1024, H = 512, c = document.createElement('canvas'); c.width = W; c.height = H; const g = c.getContext('2d');
    const P = { shad: ['#173247', '#8fa6b6', '#f1f4f5'], chartreuse: ['#2f6a1c', '#d4ef37', '#fbf6b8'], firetiger: ['#28591a', '#b9dc32', '#ff6b1f'] }[kind];
    for (let x = 0; x < W; x++) { const u = x / W, d = Math.min(Math.abs(u - 0.25), 1 - Math.abs(u - 0.25)), k = Math.min(1, d / 0.5);
      const col = k < 0.55 ? mix(P[0], P[1], k / 0.55) : mix(P[1], P[2], (k - 0.55) / 0.45); g.fillStyle = col; g.fillRect(x, 0, 1, H); }
    if (kind === 'firetiger') { g.fillStyle = 'rgba(15,20,10,.85)'; for (let i = 0; i < 7; i++) { const y = 70 + i * 52; for (const side of [0.5, 1.0]) { g.beginPath(); g.ellipse(side * W - W / 4 + 0, y, 140, 9, 0, 0, Math.PI * 2); g.fill(); } } }
    if (kind === 'chartreuse') { g.fillStyle = 'rgba(10,15,8,.75)'; for (let i = 0; i < 90; i++) { const x = W * (0.05 + 0.4 * Math.random()), y = 60 + Math.random() * (H - 140); g.beginPath(); g.arc(x, y, 3 + Math.random() * 3, 0, 7); g.fill(); } }
    g.strokeStyle = 'rgba(255,255,255,.18)'; g.lineWidth = 1.2;           // scales
    for (let y = 40; y < H - 60; y += 14) for (let x = (y / 14 % 2) * 9; x < W; x += 18) { g.beginPath(); g.arc(x, y, 9, 0.15 * Math.PI, 0.85 * Math.PI); g.stroke(); }
    g.fillStyle = 'rgba(0,0,0,.25)'; [0, 0.5].forEach(s => g.fillRect(W * s, H * 0.25, W * 0.5, 2));   // lateral line
    g.fillStyle = 'rgba(255,90,60,.55)'; [0.75].forEach(s => { g.beginPath(); g.ellipse(W * s, H * 0.86, 70, 16, 0, 0, 7); g.fill(); });   // gill flash
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
  }
  function mix(a, b, k) { const A = parseInt(a.slice(1), 16), B = parseInt(b.slice(1), 16), ch = s => Math.round(((A >> s) & 255) * (1 - k) + ((B >> s) & 255) * k); return 'rgb(' + ch(16) + ',' + ch(8) + ',' + ch(0) + ')'; }
  const patterns = ['shad', 'chartreuse', 'firetiger'], tex = patterns.map(paint), names = ['Shad pattern', 'Chartreuse', 'Firetiger'];

  /* ---------- lure ---------- */
  const lure = new THREE.Group(), body = new THREE.Group(); lure.add(body); scene.add(lure);
  const prof = new THREE.SplineCurve([[0, -1.25], [0.16, -1.15], [0.32, -0.9], [0.45, -0.45], [0.52, 0.05], [0.5, 0.5], [0.4, 0.88], [0.22, 1.14], [0, 1.25]].map(p => new THREE.Vector2(p[0], p[1]))).getPoints(80);
  prof[0].x = 0; prof[prof.length - 1].x = 0;
  const bodyMat = new THREE.MeshPhysicalMaterial({ map: tex[0], roughness: 0.32, metalness: 0.25, clearcoat: 1, clearcoatRoughness: 0.04, iridescence: 0.55, iridescenceIOR: 1.6, sheen: 0.2 });
  const bodyMesh = new THREE.Mesh(new THREE.LatheGeometry(prof, 96), bodyMat); bodyMesh.scale.set(1, 1, 0.62); bodyMesh.rotation.z = -Math.PI / 2; body.add(bodyMesh);
  const lipShape = new THREE.Shape(); lipShape.moveTo(-0.24, 0); lipShape.lineTo(0.24, 0); lipShape.quadraticCurveTo(0.34, 0.5, 0, 0.72); lipShape.quadraticCurveTo(-0.34, 0.5, -0.24, 0);
  const lip = new THREE.Mesh(new THREE.ExtrudeGeometry(lipShape, { depth: 0.035, bevelEnabled: true, bevelSize: 0.012, bevelThickness: 0.012, bevelSegments: 3 }), new THREE.MeshPhysicalMaterial({ color: 0xd9c49a, transmission: 0.55, roughness: 0.08, thickness: 0.4, ior: 1.5, clearcoat: 1, attenuationColor: new THREE.Color(0xb08a4a), attenuationDistance: 0.6 }));
  lip.rotation.set(0, Math.PI / 2, 0); lip.rotateX(-1.05); lip.position.set(1.12, -0.14, 0); body.add(lip);
  const eyeW = new THREE.MeshPhysicalMaterial({ color: 0xf2c94c, metalness: 0.6, roughness: 0.2, clearcoat: 1 }), eyeB = new THREE.MeshStandardMaterial({ color: 0x0b0b0b, roughness: 0.1 });
  [-1, 1].forEach(s => { const e = new THREE.Mesh(new THREE.SphereGeometry(0.1, 32, 16), eyeW); e.scale.set(1, 1, 0.45); e.position.set(0.86, 0.12, s * 0.27); body.add(e); const p = new THREE.Mesh(new THREE.SphereGeometry(0.05, 24, 12), eyeB); p.position.set(0.88, 0.12, s * 0.31); body.add(p); });
  const steel = new THREE.MeshPhysicalMaterial({ color: 0xc9ccd1, metalness: 1, roughness: 0.22, clearcoat: 0.5 });
  const ring = (r, tube) => new THREE.Mesh(new THREE.TorusGeometry(r, tube, 12, 40), steel);
  const tie = ring(0.06, 0.012); tie.position.set(1.02, 0.22, 0); tie.rotation.y = Math.PI / 2; body.add(tie);
  function treble(x, y) {
    const g = new THREE.Group(), sr = ring(0.07, 0.012); sr.rotation.y = Math.PI / 2; g.add(sr);
    const shank = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.42, 10), steel); shank.position.y = -0.27; g.add(shank);
    for (let k = 0; k < 3; k++) { const a = k * 2 * Math.PI / 3, c = new THREE.CatmullRomCurve3([new THREE.Vector3(0, -0.46, 0), new THREE.Vector3(0.12, -0.52, 0), new THREE.Vector3(0.2, -0.4, 0), new THREE.Vector3(0.17, -0.27, 0)]);
      const h = new THREE.Mesh(new THREE.TubeGeometry(c, 24, 0.013, 8), steel); h.rotation.y = a; g.add(h);
      const barb = new THREE.Mesh(new THREE.ConeGeometry(0.022, 0.07, 8), steel); barb.position.set(0.17 * Math.cos(-a), -0.24, 0.17 * Math.sin(-a)); g.add(barb); }
    g.position.set(x, y, 0); return g;
  }
  const h1 = treble(0.15, -0.5), h2 = treble(-1.2, -0.12); body.add(h1, h2);
  const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 9, 0)]);
  const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0x2b2b2b, transparent: true, opacity: 0.55 })); scene.add(line);

  /* ---------- chapter 1: the 48-hour score curve as a glowing tube ---------- */
  const curvePts = []; for (let i = 0; i <= 60; i++) { const x = -4.2 + i * 8.4 / 60, s = 0.5 + 0.32 * Math.sin(i / 60 * Math.PI * 4 - 0.6) + 0.12 * Math.sin(i / 60 * Math.PI * 11); curvePts.push(new THREE.Vector3(x, (s - 0.5) * 2.6, -0.6 + 0.9 * Math.sin(i / 60 * Math.PI))); }
  const path = new THREE.CatmullRomCurve3(curvePts), tubeGeo = new THREE.TubeGeometry(path, 300, 0.035, 10), col = [];
  const pos = tubeGeo.attributes.position, cA = new THREE.Color(0x1668a8), cB = new THREE.Color(0xe4572e); for (let i = 0; i < pos.count; i++) { const k = (pos.getY(i) / 2.6) + 0.5, c = cA.clone().lerp(cB, Math.max(0, Math.min(1, k))); col.push(c.r, c.g, c.b); }
  tubeGeo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  const tube = new THREE.Mesh(tubeGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0 })); scene.add(tube);

  /* ---------- chapter 2: a school of baitfish ---------- */
  const fp = new THREE.SplineCurve([[0, -0.5], [0.09, -0.42], [0.012, -0.36], [0.06, -0.1], [0.075, 0.15], [0.045, 0.4], [0, 0.5]].map(p => new THREE.Vector2(p[0], p[1]))).getPoints(30); fp[0].x = 0.0001;
  const fishGeo = new THREE.LatheGeometry(fp, 12); fishGeo.scale(1, 1, 0.45); fishGeo.rotateZ(-Math.PI / 2);
  const N = 70, school = new THREE.InstancedMesh(fishGeo, new THREE.MeshPhysicalMaterial({ color: 0xd8e1e6, metalness: 0.85, roughness: 0.25, iridescence: 0.8, transparent: true, opacity: 0 }), N); scene.add(school);
  const seeds = Array.from({ length: N }, (_, i) => ({ a: i / N * Math.PI * 2 + Math.random() * 0.3, r: 1.7 + Math.random() * 0.8, y: (Math.random() - 0.5) * 1.6, sp: 0.35 + Math.random() * 0.25, s: 0.55 + Math.random() * 0.5 }));
  const dummy = new THREE.Object3D();

  /* ---------- chapter 4: water surface with ripples ---------- */
  const water = new THREE.Mesh(new THREE.PlaneGeometry(40, 40, 1, 1), new THREE.ShaderMaterial({ transparent: true, depthWrite: false,
    uniforms: { uT: { value: 0 }, uA: { value: 0 }, uR: { value: 0 }, uC: { value: new THREE.Vector2(0, 0) } },
    vertexShader: 'varying vec2 vP; void main(){ vP = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: `uniform float uT; uniform float uA; uniform float uR; uniform vec2 uC; varying vec2 vP;
      void main(){ float d = length(vP - uC); vec3 deep = vec3(0.04,0.13,0.2), lit = vec3(0.16,0.42,0.6);
        float rings = 0.0; for (int k = 0; k < 4; k++) { float r = uR - float(k) * 0.55; if (r > 0.0) rings += smoothstep(0.06, 0.0, abs(d - r)) * (1.0 - r / 7.0); }
        float sheen = 0.5 + 0.5 * sin(vP.x * 1.3 + uT * 0.6) * sin(vP.y * 1.1 - uT * 0.5);
        vec3 c = mix(deep, lit, smoothstep(14.0, 0.0, d) * 0.6 + sheen * 0.08) + vec3(0.75,0.9,1.0) * rings * 0.6;
        gl_FragColor = vec4(c, uA * smoothstep(19.0, 8.0, d)); }` }));
  water.rotation.x = -Math.PI / 2; water.position.y = -1.35; scene.add(water);

  /* ---------- scroll -> chapter position ---------- */
  const chapters = Array.from(host.querySelectorAll('.chapter')), patLabel = document.getElementById('storyPattern');
  const K = [ // per chapter: position, rotation, scale (desktop, lure on the right)
    { p: [2.0, 0.15, 0], r: [0.25, -0.55, 0.18], s: 1.25 },
    { p: [1.4, 0.1, 0], r: [0, 0, 0], s: 0.62 },
    { p: [1.9, 0.0, 0], r: [0.15, 0.35, 0.05], s: 0.95 },
    { p: [2.05, 0.1, 0.4], r: [0.05, -0.05, -0.04], s: 1.38 },
    { p: [0.35, -1.15, 0], r: [0.3, -0.4, -0.95], s: 0.95 } ];
  const lerp = (a, b, k) => a + (b - a) * k, ease = k => k * k * (3 - 2 * k);
  function chapterT() {
    const mid = innerHeight * 0.5, tops = chapters.map(c => c.getBoundingClientRect().top);
    let i = 0; while (i < tops.length - 1 && tops[i + 1] <= mid) i++;
    const h = (i < tops.length - 1 ? tops[i + 1] : tops[i] + chapters[i].offsetHeight) - tops[i];
    return Math.max(0, Math.min(chapters.length - 1, i + (mid - tops[i]) / h - 0.5));
  }
  let raf = 0, last = performance.now(), clock = 0, lastPat = -1, mobile = false;
  function size() { const w = innerWidth, h = innerHeight; renderer.setSize(w, h, false); camera.aspect = w / h; mobile = w < 760; camera.position.set(0, mobile ? -0.6 : 0, mobile ? 13 : 10); camera.updateProjectionMatrix(); }
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now; if (!reduce) clock += dt;
    const t = chapterT(), i = Math.min(3, Math.floor(t)), k = ease(t - i), A = K[i], B = K[Math.min(4, i + 1)];
    const w1 = Math.max(0, 1 - Math.abs(t - 1) * 1.6), w2 = Math.max(0, 1 - Math.abs(t - 2) * 1.6), w4 = Math.max(0, Math.min(1, (t - 3.3) / 0.7));
    let px = lerp(A.p[0], B.p[0], k), py = lerp(A.p[1], B.p[1], k), pz = lerp(A.p[2], B.p[2], k);
    let rx = lerp(A.r[0], B.r[0], k), ry = lerp(A.r[1], B.r[1], k), rz = lerp(A.r[2], B.r[2], k); const sc = lerp(A.s, B.s, k);
    if (w1 > 0) { const u = Math.max(0.02, Math.min(0.98, (t - 0.5))), onC = path.getPointAt(u), tan = path.getTangentAt(u); const cx = 0.9; px = lerp(px, onC.x * 0.55 + cx, w1); py = lerp(py, onC.y * 0.55, w1); pz = lerp(pz, onC.z * 0.55, w1); rz = lerp(rz, Math.atan2(tan.y, tan.x), w1); ry = lerp(ry, -Math.atan2(tan.z, tan.x) * 0.6, w1); }
    if (mobile) { px = 0.15 * (px - 1.6); py += 1.3; }
    lure.position.set(px, py + (reduce ? 0 : Math.sin(clock * 1.4) * 0.06 * (1 - w4)), pz);
    lure.rotation.set(rx + (reduce ? 0 : Math.sin(clock * 0.9) * 0.05), ry + (reduce ? 0 : (t < 0.6 ? clock * 0.35 : 0) * (1 - Math.min(1, t / 0.6))), rz + (reduce ? 0 : Math.sin(clock * 2.1) * 0.03 * (1 - w4)));
    lure.scale.setScalar(sc * (mobile ? 0.68 : 1));
    const camY = (mobile ? -0.6 : 0) + 5.5 * w4; camera.position.y = camY; camera.lookAt(0, (mobile ? -0.6 : 0) - 1.6 * w4, 0);
    const pc = [239, 231, 218], nc = [15, 42, 61]; host.style.backgroundColor = 'rgb(' + pc.map((v, j) => Math.round(v + (nc[j] - v) * w4)).join(',') + ')'; host.classList.toggle('deep', w4 > 0.5);
    const tip = new THREE.Vector3(1.02, 0.22, 0); lure.localToWorld(tip); lineGeo.attributes.position.setXYZ(0, tip.x, tip.y, tip.z); lineGeo.attributes.position.setXYZ(1, tip.x + (w4 ? -1.5 * w4 : 0), 9, tip.z); lineGeo.attributes.position.needsUpdate = true;
    tube.material.opacity = w1; tube.position.set(mobile ? 0 : 0.9, mobile ? 1.3 : 0, 0); tube.scale.setScalar(0.55); tube.visible = w1 > 0.01;
    school.material.opacity = w2; school.visible = w2 > 0.01;
    if (school.visible) seeds.forEach((s, j) => { const a = s.a + clock * s.sp, r = s.r * (0.7 + 0.3 * w2); dummy.position.set(px + Math.cos(a) * r, py + s.y + Math.sin(clock * 1.3 + j) * 0.08, pz + Math.sin(a) * r * 0.6); dummy.rotation.set(0, -a - Math.PI / 2, 0); dummy.scale.setScalar(s.s * (0.4 + 0.6 * w2)); dummy.updateMatrix(); school.setMatrixAt(j, dummy.matrix); });
    school.instanceMatrix.needsUpdate = true;
    const pat = t < 2.6 ? 0 : Math.min(2, Math.floor((t - 2.6) * 3)); if (pat !== lastPat) { lastPat = pat; bodyMat.map = tex[pat]; bodyMat.needsUpdate = true; if (patLabel) patLabel.textContent = names[pat]; }
    water.material.uniforms.uA.value = w4 * 0.92; water.material.uniforms.uT.value = clock; water.material.uniforms.uR.value = w4 > 0 ? (reduce ? 2.5 : (clock * 1.6) % 7) : 0; water.material.uniforms.uC.value.set(px, -pz); water.position.y = mobile ? -0.4 : -1.35; water.visible = w4 > 0.01;
    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  const running = () => !host.hidden && !document.hidden;
  function sync() { cancelAnimationFrame(raf); raf = 0; if (running()) { size(); last = performance.now(); raf = requestAnimationFrame(frame); } }
  addEventListener('resize', () => { if (running()) size(); });
  new MutationObserver(sync).observe(host, { attributes: true, attributeFilter: ['hidden'] });
  document.addEventListener('visibilitychange', sync); sync();
}

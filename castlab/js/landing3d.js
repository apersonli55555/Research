/* Landing-page scene: a living depth chart. A procedural lake basin drawn as contour lines, with a water
   level that slowly rises and falls. Decorative only (no real place, no data claims). ES module (three.js).
   Falls back to the static CSS contour background if WebGL is unavailable; honours reduced-motion. */
import * as THREE from 'three';

const host = document.getElementById('authView');
if (host) { try { start(); } catch (e) { /* keep the CSS background */ } }

function start() {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(1.75, window.devicePixelRatio || 1));
  const canvas = renderer.domElement; canvas.className = 'land3d'; canvas.setAttribute('aria-hidden', 'true'); host.prepend(canvas); host.classList.add('has3d');
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  const geo = new THREE.PlaneGeometry(30, 30, 260, 260); geo.rotateX(-Math.PI / 2);
  const uniforms = { uTime: { value: 0 }, uLevel: { value: 0 } };
  const mat = new THREE.ShaderMaterial({
    uniforms, transparent: true,
    vertexShader: `
      uniform float uTime; uniform float uLevel; varying float vH; varying vec2 vP;
      float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float noise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
        return mix(mix(hash(i), hash(i+vec2(1,0)), f.x), mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y); }
      float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int k = 0; k < 5; k++) { s += a * noise(p); p = p * 2.03 + 11.7; a *= 0.5; } return s; }
      float terrain(vec2 p){ float d = length((p - vec2(1.5, 0.5)) * vec2(0.75, 1.05)); float basin = smoothstep(0.5, 7.0, d) * 3.2 - 1.3; return basin + (fbm(p * 0.28) - 0.5) * 2.4; }
      void main(){ vec3 pos = position; float h = terrain(pos.xz); vH = h; vP = pos.xz;
        float y = h; if (h < uLevel) y = uLevel + 0.02 * sin(pos.x * 2.2 + uTime * 0.9) * sin(pos.z * 1.9 - uTime * 0.7);
        pos.y = y; gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0); }`,
    fragmentShader: `
      uniform float uTime; uniform float uLevel; varying float vH; varying vec2 vP;
      float line(float v, float w){ float d = abs(fract(v - 0.5) - 0.5) / fwidth(v); return 1.0 - smoothstep(0.0, w, d); }
      void main(){
        vec3 deep = vec3(0.059, 0.165, 0.239), ink = vec3(0.62, 0.83, 0.98), col; float a = 1.0;
        if (vH < uLevel) { float dpt = uLevel - vH;
          col = mix(vec3(0.10, 0.36, 0.56), vec3(0.04, 0.14, 0.24), smoothstep(0.0, 1.5, dpt));
          col += ink * 0.10 * line(dpt * 3.0 + 0.12 * sin(uTime * 0.5 + vP.x * 0.6), 1.2);
          col += vec3(0.75, 0.90, 1.0) * (1.0 - smoothstep(0.0, 0.07, dpt)) * 0.55; }
        else { float up = vH - uLevel; col = deep + vec3(0.02, 0.05, 0.07) * smoothstep(0.0, 3.0, up);
          col += ink * 0.30 * line(vH * 4.0, 1.25); col += ink * 0.30 * line(vH * 0.8, 1.6); }
        float fade = 1.0 - smoothstep(9.0, 15.0, length(vP));
        gl_FragColor = vec4(mix(deep, col, fade), a); }`
  });
  scene.add(new THREE.Mesh(geo, mat));
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches; let raf = 0, t0 = performance.now();
  function size() { const w = host.clientWidth, h = host.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); frame(performance.now(), true); }
  function frame(now, once) {
    const t = reduce ? 6 : (now - t0) / 1000; uniforms.uTime.value = t; uniforms.uLevel.value = -0.05 + 0.42 * Math.sin(t * 0.22);
    const ang = 0.5 + t * 0.025, R = host.clientWidth < 760 ? 15 : 12.5; camera.position.set(Math.sin(ang) * R + 1.5, 7.2, Math.cos(ang) * R + 0.5); camera.lookAt(1.5, -0.4, 0.5);
    renderer.render(scene, camera);
    if (!once && !reduce) raf = requestAnimationFrame(frame);
  }
  const running = () => !host.hidden && !document.hidden;
  function sync() { cancelAnimationFrame(raf); raf = 0; if (running()) { size(); if (!reduce) raf = requestAnimationFrame(frame); } }
  new ResizeObserver(() => { if (running()) size(); }).observe(host);
  new MutationObserver(sync).observe(host, { attributes: true, attributeFilter: ['hidden'] });
  document.addEventListener('visibilitychange', sync); sync();
}

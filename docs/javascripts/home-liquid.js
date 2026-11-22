/* Homepage hero: the liquid "O" rendered through a WebGL shader.
   The <img> underneath is the static fallback (reduced motion, no WebGL, context loss);
   the canvas only fades in once it has drawn a frame. */
(function () {
  "use strict";

  var stage = document.querySelector("[data-liquid-stage]");
  if (!stage) return;
  var img = stage.querySelector("img");
  var canvas = stage.querySelector("canvas");
  var hero = stage.closest("section") || stage;
  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (reducedMotion.matches) return;

  var TEX_SIZE = 1024;
  var TEX_PAD = 48; // transparent margin so displaced lookups never smear the image edge
  var O_HEIGHT = 0.8; // O height as a fraction of canvas height
  var O_CENTER_Y = -0.05; // leaves headroom above the O for droplets
  var MAX_RIPPLES = 6;
  var MAX_DROPS = 5;
  var GRAVITY = 0.42; // canvas-heights per second squared
  var MAX_DPR = 1.75;
  var IDLE_AFTER_MS = 3000; // with no pointer input for this long, drop to ~30 fps
  var IDLE_FRAME_MS = 30; // just under two 60 Hz frames, so timestamp jitter can't skip a third
  var ACTIVE_FRAME_MS = 9.5; // caps 120/144 Hz near 60/72 fps; 60 to 100 Hz displays draw every frame
  var MAX_TILT_X_DEG = 5;
  var MAX_TILT_Y_DEG = 7;
  var TILT_PERSPECTIVE_PX = 1400;
  var MAX_DT_S = 0.05;
  var RIPPLE_HIT_ALPHA = 0.4; // the pointer only trails ripples over the letter's body
  var TILT_EASE = 0.06; // fraction of the gap to the pointer target closed per 60 Hz frame
  var RIPPLE_SPACING = 0.06; // canvas heights the pointer must travel before it trails another ripple
  var CLOCK_OFFSET_S = 12; // starts the noise mid-flow rather than from its symmetric origin
  var RIPPLE_LIFE_S = 3;
  var RIPPLE_AMP_TRAIL = 0.45;
  var RIPPLE_AMP_CLICK = 1.2;
  var RIPPLE_AMP_LANDING = 0.7;
  var RIPPLE_AMP_INTRO = 1.0;
  var FIRST_DROP_S = 1.2;
  var DROP_GAP_MIN_S = 0.9;
  var DROP_GAP_RANGE_S = 1.4;
  var INTRO_S = 1.6;
  var SOFTWARE_RENDERER = /swiftshader|llvmpipe|softpipe|basic render|software/i;

  var VERT = "attribute vec2 aPos;varying vec2 vUv;void main(){vUv=aPos*.5+.5;gl_Position=vec4(aPos,0.,1.);}";

  var FRAG = [
    "precision highp float;",
    "varying vec2 vUv;",
    "uniform sampler2D uTex;",
    "uniform float uTime;",
    "uniform float uIntro;",
    "uniform vec2 uRes;",
    "uniform vec4 uBox;",
    "uniform vec2 uLight;",
    "uniform vec4 uRip[" + MAX_RIPPLES + "];",
    "uniform vec4 uDrop[" + MAX_DROPS + "];",
    "float hash(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}",
    "float noise(vec2 p){vec2 i=floor(p),f=fract(p);vec2 u=f*f*(3.-2.*f);",
    " return mix(mix(hash(i),hash(i+vec2(1.,0.)),u.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),u.x),u.y);}",
    "float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*noise(p);p=p*2.03+vec2(1.7,9.2);a*=.5;}return v;}",
    "vec2 toTex(vec2 p){return vec2((p.x-uBox.x)/uBox.z+.5,.5-(p.y-uBox.y)/uBox.w);}",
    "vec4 tex(vec2 uv,float b){vec2 m=step(vec2(0.),uv)*step(uv,vec2(1.));return texture2D(uTex,uv,b)*m.x*m.y;}",
    "void main(){",
    " vec2 p=(vUv-.5)*vec2(uRes.x/uRes.y,1.);",
    " float t=uTime;",
    " float br=(1.+.014*sin(t*.8)+.006*sin(t*1.9))*mix(.9,1.,uIntro);",
    " vec2 pc=uBox.xy+(p-uBox.xy)/br;",
    // Droplets: metaballs summed with a blurred copy of the letter's alpha, so a drop
    // necks off the rim as it leaves and melts back in when it lands.
    " float f=tex(toTex(pc),4.).a*.95;",
    " float best=9.;vec2 dv=vec2(0.);",
    " for(int i=0;i<" + MAX_DROPS + ";i++){",
    "  vec4 d=uDrop[i]; if(d.z<=0.) continue;",
    "  vec2 v=pc-d.xy; f+=d.z*d.z/max(dot(v,v),1e-6);",
    "  float s=length(v)/d.z; if(s<best){best=s;dv=v/d.z;}",
    " }",
    // Pixels outside the texture box with no drop skip the displacement: it can't reach past TEX_PAD to the
    // letter. Texture lookups stay outside the branch; their mip level comes from the whole 2x2 pixel quad
    // and is undefined if part of it branched away.
    " vec2 bu=toTex(pc);",
    " vec2 disp=vec2(0.);float cs=.5;",
    " if(f>=1.||(bu.x>=0.&&bu.x<=1.&&bu.y>=0.&&bu.y<=1.)){",
    // Slow domain-warped flow: the whole letter reads as moving liquid rather than a wobble.
    "  vec2 q=pc*2.6;",
    "  vec2 w=vec2(fbm(q+vec2(t*.10,-t*.07)),fbm(q+vec2(-t*.08,t*.11)+7.3));",
    "  float n=fbm(q*1.7+(w-.5)*2.5+vec2(0.,-t*.15));",
    "  cs=fbm(pc*8.+(w-.5)*3.+vec2(t*.22,t*.16));",
    "  disp=(w-.5)*.026+(n-.5)*.012+(1.-uIntro)*(n-.5)*.08;",
    "  for(int i=0;i<" + MAX_RIPPLES + ";i++){",
    "   vec4 r=uRip[i]; if(r.w<=0.) continue;",
    "   vec2 d=pc-r.xy; float dist=max(length(d),1e-4);",
    "   float k=dist-r.z*.34;",
    "   float wave=sin(k*80.)*exp(-k*k*300.)*exp(-r.z*1.5)*r.w;",
    "   disp+=d/dist*wave*.012;",
    "  }",
    " }",
    " vec2 tuv=toTex(pc+disp);",
    " vec4 c=tex(tuv,0.);",
    // Split R and B along the displacement for a refraction fringe.
    " c.r=tex(toTex(pc+disp*1.2),0.).r;",
    " c.b=tex(toTex(pc+disp*.8),0.).b;",
    " float e=2./" + TEX_SIZE + ".;",
    " float hl=tex(tuv-vec2(e,0.),1.5).a,hr=tex(tuv+vec2(e,0.),1.5).a;",
    " float hu=tex(tuv-vec2(0.,e),1.5).a,hd=tex(tuv+vec2(0.,e),1.5).a;",
    " vec3 nrm=normalize(vec3(hl-hr,hd-hu,.22));",
    " vec3 L=normalize(vec3(uLight,1.));",
    " float spec=pow(max(dot(reflect(-L,nrm),vec3(0.,0.,1.)),0.),28.);",
    " float caust=pow(1.-abs(cs*2.-1.),12.);",
    " float lum=dot(c.rgb,vec3(.3,.5,.2));",
    " c.rgb+=vec3(.9,1.,.93)*(caust*.5*lum+spec*.45*c.a);",
    " c.rgb=min(c.rgb,vec3(c.a));",
    " float m=smoothstep(1.,1.12,f);",
    " vec4 dcol=vec4(0.);",
    " if(m>0.){",
    "  float nz=best<1.?sqrt(1.-best*best):.35;",
    "  vec3 dn=best<1.?vec3(dv,nz):vec3(0.,0.,1.);",
    "  float ds=pow(max(dot(reflect(-L,dn),vec3(0.,0.,1.)),0.),80.)*2.;",
    // A glass drop: see-through core, dense green edge, bright fresnel rim, and a caustic
    // crescent on the side away from the light.
    "  float edge=1.-nz;",
    "  vec3 base=vec3(.08,.6,.26)+vec3(.85,1.,.9)*pow(edge,4.)*.9;",
    "  base+=vec3(.7,1.,.8)*pow(max(dot(dn.xy,-normalize(uLight)),0.),3.)*smoothstep(.45,.9,best)*.9;",
    "  float a=m*(.45+.5*edge*edge);",
    "  dcol=vec4(base*a+vec3(ds*m),min(1.,a+ds*m));",
    "  dcol.rgb=min(dcol.rgb,vec3(dcol.a));",
    " }",
    " gl_FragColor=(c+dcol*(1.-c.a))*uIntro;",
    "}",
  ].join("\n");

  // A software renderer would run this shader on the CPU every frame; it gets the static image instead.
  var gl = canvas.getContext("webgl", {
    premultipliedAlpha: true, alpha: true, antialias: false, depth: false, failIfMajorPerformanceCaveat: true,
  });
  if (!gl) return;
  // Chrome hands out SwiftShader (also what headless Chrome, Lighthouse and PageSpeed run on) despite the caveat flag.
  var dbg = gl.getExtension("WEBGL_debug_renderer_info");
  var renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  if (SOFTWARE_RENDERER.test(renderer)) return;

  function compile(type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    return sh;
  }

  // The GPU compiles while the image decodes. Querying the program (status, locations) blocks until it is
  // done, so start() does that.
  var prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);

  var buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

  var U = {};

  var box = [0, O_CENTER_Y, 0, 0];
  var alphaMap = null; // alpha samples for picking droplet launch points on the rim
  var ALPHA_RES = 128;

  function upload() {
    // WebGL1 only mipmaps power-of-two textures; the blur used for the metaball neck needs mips.
    var c = document.createElement("canvas");
    c.width = c.height = TEX_SIZE;
    var ctx = c.getContext("2d");
    ctx.drawImage(img, TEX_PAD, TEX_PAD, TEX_SIZE - 2 * TEX_PAD, TEX_SIZE - 2 * TEX_PAD);
    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.uniform1i(U.uTex, 0);

    // Drawn from the image, not from c: reading back a GPU-backed 1024px canvas stalls the main thread.
    var s = document.createElement("canvas");
    s.width = s.height = ALPHA_RES;
    var sctx = s.getContext("2d", { willReadFrequently: true });
    var k = ALPHA_RES / TEX_SIZE;
    sctx.drawImage(img, TEX_PAD * k, TEX_PAD * k, (TEX_SIZE - 2 * TEX_PAD) * k, (TEX_SIZE - 2 * TEX_PAD) * k);
    var data = sctx.getImageData(0, 0, ALPHA_RES, ALPHA_RES).data;
    alphaMap = new Float32Array(ALPHA_RES * ALPHA_RES);
    for (var i = 0; i < alphaMap.length; i++) alphaMap[i] = data[i * 4 + 3] / 255;

    var texH = O_HEIGHT * TEX_SIZE / (TEX_SIZE - 2 * TEX_PAD);
    box[3] = texH;
    box[2] = texH * (img.naturalWidth / img.naturalHeight);
  }

  function alphaAt(u, v) {
    var x = Math.min(ALPHA_RES - 1, Math.max(0, Math.round(u * ALPHA_RES)));
    var y = Math.min(ALPHA_RES - 1, Math.max(0, Math.round(v * ALPHA_RES)));
    return alphaMap[y * ALPHA_RES + x];
  }

  function texToCanvas(u, v) {
    return [box[0] + (u - 0.5) * box[2], box[1] + (0.5 - v) * box[3]];
  }

  /* March outward from the centre of the letter along angle a and return the last
     point that was inside the rim, or null if the ray never crossed it. */
  function rimPoint(a) {
    var inside = false, last = null, ca = Math.cos(a), sa = Math.sin(a);
    for (var r = 0; r < 0.5; r += 0.004) {
      var u = 0.5 + ca * r, v = 0.5 - sa * r;
      var al = alphaAt(u, v);
      if (al > 0.6) { inside = true; last = [u, v]; }
      else if (inside && al < 0.3) break;
    }
    return last;
  }

  var ripples = [];
  var drops = [];
  var ripArr = new Float32Array(MAX_RIPPLES * 4);
  var dropArr = new Float32Array(MAX_DROPS * 4);

  function addRipple(x, y, amp) {
    if (ripples.length >= MAX_RIPPLES) ripples.shift();
    ripples.push({ x: x, y: y, age: 0, amp: amp });
  }

  function spawnDrop() {
    // Upper arc only, so a drop launched mostly upward falls back onto the rim.
    var a = (0.2 + Math.random() * 0.6) * Math.PI;
    var rp = rimPoint(a);
    if (!rp) return;
    var anchor = texToCanvas(rp[0], rp[1]);
    var apex = 0.05 + Math.random() * 0.07;
    var vy = Math.sqrt(2 * GRAVITY * apex);
    drops.push({
      ax: anchor[0], ay: anchor[1],
      nx: Math.cos(a), ny: Math.sin(a),
      vx: Math.cos(a) * 0.05, vy: vy,
      flight: 2 * vy / GRAVITY,
      r: 0.014 + Math.random() * 0.01,
      age: 0,
    });
  }

  var EMERGE = 0.45, MERGE = 0.2;

  /* Writes a drop's x, y and radius into out at offset o; returns false once it has melted back in. */
  function dropState(d, out, o) {
    if (d.age < EMERGE) {
      var k = d.age / EMERGE;
      var e = k * k * (3 - 2 * k);
      out[o] = d.ax - d.nx * d.r * (1 - e) * 1.5;
      out[o + 1] = d.ay - d.ny * d.r * (1 - e) * 1.5;
      out[o + 2] = d.r * (0.5 + 0.5 * e);
      return true;
    }
    var tt = d.age - EMERGE;
    if (tt < d.flight) {
      out[o] = d.ax + d.vx * tt;
      out[o + 1] = d.ay + d.vy * tt - 0.5 * GRAVITY * tt * tt;
      out[o + 2] = d.r;
      return true;
    }
    var m = (tt - d.flight) / MERGE;
    if (m >= 1) return false;
    out[o] = d.ax + d.vx * d.flight;
    out[o + 1] = d.ay - d.vy * (tt - d.flight) * 0.6;
    out[o + 2] = d.r * (1 - m);
    return true;
  }

  var dpr = 1, aspect = 1, rawDpr = 0;
  function resize() {
    rawDpr = window.devicePixelRatio || 1;
    dpr = Math.min(rawDpr, MAX_DPR);
    var w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    var h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
    }
    gl.uniform2f(U.uRes, w, h);
    aspect = w / h;
  }

  // Pointer: tilts the stage, steers the light and trails small ripples across the letter.
  var target = { x: 0, y: 0 }, cur = { x: 0, y: 0 };
  var lastRip = null;
  var lastInput = 0;
  hero.addEventListener("pointermove", function (ev) {
    lastInput = performance.now();
    var hr = hero.getBoundingClientRect();
    target.x = ((ev.clientX - hr.left) / hr.width) * 2 - 1;
    target.y = ((ev.clientY - hr.top) / hr.height) * 2 - 1;
    if (!alphaMap) return;
    var cr = canvas.getBoundingClientRect();
    var x = ((ev.clientX - cr.left) / cr.width - 0.5) * aspect;
    var y = 0.5 - (ev.clientY - cr.top) / cr.height;
    var u = (x - box[0]) / box[2] + 0.5, v = 0.5 - (y - box[1]) / box[3];
    if (u < 0 || u > 1 || v < 0 || v > 1 || alphaAt(u, v) < RIPPLE_HIT_ALPHA) return;
    if (lastRip && Math.hypot(x - lastRip[0], y - lastRip[1]) < RIPPLE_SPACING) return;
    lastRip = [x, y];
    addRipple(x, y, RIPPLE_AMP_TRAIL);
  });
  hero.addEventListener("pointerleave", function () { target.x = target.y = 0; });
  canvas.addEventListener("pointerdown", function (ev) {
    lastInput = performance.now();
    var cr = canvas.getBoundingClientRect();
    addRipple(((ev.clientX - cr.left) / cr.width - 0.5) * aspect, 0.5 - (ev.clientY - cr.top) / cr.height, RIPPLE_AMP_CLICK);
  });

  var running = false, visible = true, onscreen = true, lost = false, raf = 0;
  var last = 0, clock = 0, nextDrop = FIRST_DROP_S, shown = false;
  var sized = false, tiltX = NaN, tiltY = NaN;
  new ResizeObserver(function () { sized = false; }).observe(canvas);

  function frame(now) {
    if (now - last < (now - lastInput > IDLE_AFTER_MS ? IDLE_FRAME_MS : ACTIVE_FRAME_MS)) {
      raf = requestAnimationFrame(frame);
      return;
    }
    // The first rAF timestamp can precede the performance.now() taken when the loop was started.
    var dt = Math.max(0, Math.min(MAX_DT_S, (now - last) / 1000 || 0));
    last = now;
    clock += dt;
    // A window moved to a monitor with another DPR keeps its CSS size, so the ResizeObserver stays silent.
    if (!sized || window.devicePixelRatio !== rawDpr) {
      resize();
      sized = true;
    }

    var ease = 1 - Math.pow(1 - TILT_EASE, dt * 60);
    cur.x += (target.x - cur.x) * ease;
    cur.y += (target.y - cur.y) * ease;
    // Rounded to 0.01deg so a settled pointer stops rewriting the transform.
    var rx = Math.round(-cur.y * MAX_TILT_X_DEG * 100) / 100, ry = Math.round(cur.x * MAX_TILT_Y_DEG * 100) / 100;
    if (rx !== tiltX || ry !== tiltY) {
      tiltX = rx;
      tiltY = ry;
      stage.style.transform = "perspective(" + TILT_PERSPECTIVE_PX + "px) rotateX(" + rx + "deg) rotateY(" + ry + "deg)";
    }

    nextDrop -= dt;
    if (nextDrop <= 0 && drops.length < MAX_DROPS) {
      spawnDrop();
      nextDrop = DROP_GAP_MIN_S + Math.random() * DROP_GAP_RANGE_S;
    }

    dropArr.fill(0);
    var n = 0;
    for (var i = 0; i < drops.length; i++) {
      var d = drops[i];
      var wasFlying = d.age - EMERGE < d.flight;
      d.age += dt;
      if (!dropState(d, dropArr, n * 4)) continue;
      if (wasFlying && d.age - EMERGE >= d.flight) addRipple(dropArr[n * 4], dropArr[n * 4 + 1], RIPPLE_AMP_LANDING);
      drops[n++] = d;
    }
    drops.length = n;

    ripArr.fill(0);
    n = 0;
    for (var j = 0; j < ripples.length; j++) {
      var r = ripples[j];
      r.age += dt;
      if (r.age >= RIPPLE_LIFE_S) continue;
      ripArr[n * 4] = r.x;
      ripArr[n * 4 + 1] = r.y;
      ripArr[n * 4 + 2] = r.age;
      ripArr[n * 4 + 3] = r.amp;
      ripples[n++] = r;
    }
    ripples.length = n;

    var intro = Math.min(1, clock / INTRO_S);
    intro = 1 - Math.pow(1 - intro, 3);

    gl.uniform1f(U.uTime, clock + CLOCK_OFFSET_S);
    gl.uniform1f(U.uIntro, intro);
    gl.uniform2f(U.uLight, -0.45 + cur.x * 0.8, 0.55 - cur.y * 0.8);
    gl.uniform4fv(U.uRip, ripArr);
    gl.uniform4fv(U.uDrop, dropArr);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    if (!shown) {
      shown = true;
      stage.classList.add("is-live");
    }
    raf = requestAnimationFrame(frame);
  }

  function update() {
    // The observers report before the image loads; a frame without alphaMap would throw in spawnDrop.
    var should = alphaMap !== null && !lost && visible && onscreen;
    if (should && !running) {
      running = true;
      last = performance.now();
      raf = requestAnimationFrame(frame);
    } else if (!should) {
      running = false;
      // A hidden tab keeps its pending callback, so without this each tab switch adds another loop.
      cancelAnimationFrame(raf);
    }
  }

  // No restore path: the static image takes over for good, so a later tab switch must not restart the loop.
  function stopForGood() {
    lost = true;
    running = false;
    cancelAnimationFrame(raf);
    stage.classList.remove("is-live");
    stage.style.transform = "";
  }
  canvas.addEventListener("webglcontextlost", stopForGood);
  reducedMotion.addEventListener("change", stopForGood);

  document.addEventListener("visibilitychange", function () {
    visible = document.visibilityState === "visible";
    update();
  });

  new IntersectionObserver(function (entries) {
    onscreen = entries[entries.length - 1].isIntersecting; // a delayed callback can batch several
    update();
  }).observe(stage);

  function start() {
    // Also false once the context is lost.
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return; // keep the static image
    gl.useProgram(prog);
    var aPos = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    ["uTex", "uTime", "uIntro", "uRes", "uBox", "uLight", "uRip", "uDrop"].forEach(function (n) {
      U[n] = gl.getUniformLocation(prog, n);
    });
    upload();
    gl.uniform4fv(U.uBox, box);
    addRipple(0, box[1] + box[3] * 0.3, RIPPLE_AMP_INTRO);
    lastInput = performance.now(); // otherwise a slow page load leaves the intro on the idle frame rate
    update();
  }

  // decode() also waits for the load, and spares upload()'s drawImage a synchronous decode. It rejects for a broken image.
  img.decode().then(start, function () {});
})();

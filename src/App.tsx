import React, { useState, useRef, useEffect, useCallback } from 'react';
import { computeHomography, invertHomography, defaultCorners, type Quad } from './homography';

// ─── Shaders ──────────────────────────────────────────────────────────────────

const VERT = `
attribute vec2 aPos;
attribute vec3 aUVW;
varying vec3 vUVW;
void main() { vUVW = aUVW; gl_Position = vec4(aPos, 0.0, 1.0); }
`;

// uEdge=1 → soft feather at UV edges; uOpacity < 1 for ghost/layer transparency
const FRAG_PLAIN = `
precision mediump float;
uniform sampler2D uTex;
uniform float uEdge;
uniform float uOpacity;
varying vec3 vUVW;
void main() {
  vec2 uv = vUVW.xy / vUVW.z;
  vec4 c = texture2D(uTex, uv);
  if (uEdge > 0.5) {
    float f = 0.05;
    float a = smoothstep(0.0, f, uv.x)   * smoothstep(0.0, f, 1.0-uv.x) *
              smoothstep(0.0, f, uv.y)   * smoothstep(0.0, f, 1.0-uv.y);
    c.a *= a;
  }
  float op = uOpacity > 0.001 ? uOpacity : 1.0;
  c.a *= op;
  gl_FragColor = c;
}
`;

// Chroma key v2: highp, saturation gate, luminance-preserving spill suppression
const FRAG_CHROMA = `
precision highp float;
uniform sampler2D uTex;
uniform vec3 uKey;
uniform float uThresh;
uniform float uSoft;
uniform float uSpill;
varying vec3 vUVW;
void main() {
  vec2 uv = vUVW.xy / vUVW.z;
  vec4 c = texture2D(uTex, uv);
  vec3 col = c.rgb;

  // Angle-weighted distance in RGB space
  float cLen = max(0.001, length(col));
  float kLen = max(0.001, length(uKey));
  float cosA = dot(col/cLen, uKey/kLen);
  float dist = distance(col, uKey);
  float combined = dist * (1.0 + max(0.0, 1.0 - cosA) * 0.5);

  // Saturation gate — protect desaturated/dark pixels (reflections, shadows, bezel)
  float maxC = max(col.r, max(col.g, col.b));
  float minC = min(col.r, min(col.g, col.b));
  float sat = maxC > 0.001 ? (maxC - minC) / maxC : 0.0;
  float satGate = smoothstep(0.06, 0.22, sat); // <6% sat = fully preserve

  float rawAlpha = smoothstep(uThresh - uSoft, uThresh + uSoft, combined);
  // Sharpen alpha: push partial-transparency pixels toward fully transparent
  // Eliminates soft green fringe on screen edges without affecting opaque bezel/body
  float sharpAlpha = rawAlpha * rawAlpha * (3.0 - 2.0 * rawAlpha);
  float alpha = mix(1.0, sharpAlpha, satGate);

  // Luminance-preserving spill suppression (uSpill = user despill strength)
  float spill = (1.0 - rawAlpha) * satGate * uSpill;
  if(spill > 0.001) {
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    if(uKey.g >= uKey.r && uKey.g >= uKey.b) {
      float excess = col.g - max(col.r, col.b);
      col.g -= max(0.0, excess) * spill * 0.95;
    } else if(uKey.b >= uKey.r) {
      float excess = col.b - max(col.r, col.g);
      col.b -= max(0.0, excess) * spill * 0.95;
    } else {
      float excess = col.r - max(col.g, col.b);
      col.r -= max(0.0, excess) * spill * 0.95;
    }
    // Restore original luminance to prevent darkening at edges
    float newLum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    if(newLum > 0.001) col *= lum / newLum;
  }

  gl_FragColor = vec4(clamp(col, 0.0, 1.0), alpha);
}
`;

// Single-pass two-texture blend — always fully opaque output, zero bleed possible.
// Outside pin quad → mockup. Inside → mix(mockup, recording, 1-luma): dark screen shows
// recording, bright pixels (hand/bezel) stay as mockup.
// uRec bound to TEXTURE1. uRecCrop = vec4(u0,u1,v0,v1) for cover-scale recording UV.
const FRAG_BLEND = `
precision mediump float;
uniform sampler2D uTex;
uniform sampler2D uRec;
uniform vec2 uQ0,uQ1,uQ2,uQ3;
uniform float uLuma;
uniform float uLumaSoft;
uniform float uChroma;
uniform vec3 uChromaKey;
uniform vec4 uRecCrop;
varying vec3 vUVW;
float cx(vec2 a,vec2 b){return a.x*b.y-a.y*b.x;}
void main(){
  vec2 uv=vUVW.xy/vUVW.z;
  vec4 mock=texture2D(uTex,uv);
  float d0=cx(uQ1-uQ0,uv-uQ0),d1=cx(uQ2-uQ1,uv-uQ1);
  float d2=cx(uQ3-uQ2,uv-uQ2),d3=cx(uQ0-uQ3,uv-uQ3);
  bool inside=(d0>=0.0&&d1>=0.0&&d2>=0.0&&d3>=0.0)||(d0<=0.0&&d1<=0.0&&d2<=0.0&&d3<=0.0);
  if(!inside){gl_FragColor=vec4(mock.rgb,1.0);return;}
  float t;
  if(uChroma>0.001){
    float kLen=max(0.001,length(uChromaKey));
    float cLen=max(0.001,length(mock.rgb));
    float cosA=dot(mock.rgb/cLen,uChromaKey/kLen);
    float dist=distance(mock.rgb,uChromaKey);
    float combined=dist*(1.0+max(0.0,1.0-cosA)*0.5);
    t=1.0-smoothstep(max(0.0,uChroma-uLumaSoft),uChroma+uLumaSoft,combined);
  } else {
    float lum=dot(mock.rgb,vec3(0.2126,0.7152,0.0722));
    t=1.0-smoothstep(max(0.0,uLuma-uLumaSoft),uLuma+uLumaSoft,lum);
  }
  vec2 ruv=vec2(uRecCrop.x+uv.x*(uRecCrop.y-uRecCrop.x),uRecCrop.z+uv.y*(uRecCrop.w-uRecCrop.z));
  vec4 rec=texture2D(uRec,ruv);
  gl_FragColor=vec4(mix(mock.rgb,rec.rgb,t),1.0);
}
`;

// Mockup with screen area punched out — lets recording show through, keeps foreground (hand) on top
const FRAG_CUTOUT = `
precision mediump float;
uniform sampler2D uTex;
uniform vec2 uQ0,uQ1,uQ2,uQ3;
uniform float uOpacity;
varying vec3 vUVW;
float cx(vec2 a,vec2 b){return a.x*b.y-a.y*b.x;}
void main(){
  vec2 uv=vUVW.xy/vUVW.z;
  float d0=cx(uQ1-uQ0,uv-uQ0),d1=cx(uQ2-uQ1,uv-uQ1);
  float d2=cx(uQ3-uQ2,uv-uQ2),d3=cx(uQ0-uQ3,uv-uQ3);
  if((d0>=0.0&&d1>=0.0&&d2>=0.0&&d3>=0.0)||(d0<=0.0&&d1<=0.0&&d2<=0.0&&d3<=0.0)) discard;
  vec4 col=texture2D(uTex,uv);
  float op = uOpacity > 0.001 ? uOpacity : 1.0;
  col.a *= op;
  gl_FragColor=col;
}
`;

// Screen recording layer: cool color-temp shift + micro contrast
// Supports optional rounded-corner clipping (uRx/uRy) and camera-hole punch (uCamPos/uCamR/uCamAsp)
const FRAG_SCREEN_REC = `
precision mediump float;
uniform sampler2D uTex;
uniform float uEdge;
uniform float uRx;
uniform float uRy;
uniform vec2  uCamPos;
uniform float uCamR;
uniform float uCamAsp;
varying vec3 vUVW;
void main() {
  vec2 uv = vUVW.xy / vUVW.z;

  // Rounded corner clipping — discard pixels in corner boxes outside the arc
  if(uRx > 0.0 && uRy > 0.0) {
    float cx = uv.x < uRx ? (uRx - uv.x)/uRx : uv.x > 1.0-uRx ? (uv.x - (1.0-uRx))/uRx : 0.0;
    float cy = uv.y < uRy ? (uRy - uv.y)/uRy : uv.y > 1.0-uRy ? (uv.y - (1.0-uRy))/uRy : 0.0;
    if(cx > 0.0 && cy > 0.0 && cx*cx + cy*cy > 1.0) discard;
  }

  // Camera-hole punch — circular cutout for punch-hole / notch cameras
  if(uCamR > 0.001) {
    vec2 d = uv - uCamPos;
    d.y *= uCamAsp;
    if(dot(d,d) < uCamR*uCamR) discard;
  }

  vec4 c = texture2D(uTex, uv);
  c.rgb = (c.rgb - 0.5) * 1.05 + 0.5;
  c.r -= 0.008; c.b += 0.010;
  if (uEdge > 0.5) {
    float f = 0.05;
    float a = smoothstep(0.0,f,uv.x)*smoothstep(0.0,f,1.0-uv.x)*
              smoothstep(0.0,f,uv.y)*smoothstep(0.0,f,1.0-uv.y);
    c.a *= a;
  }
  gl_FragColor = clamp(c,0.0,1.0);
}
`;

// Screen surface FX overlay: glare streak, corner vignette, LCD scanlines
const FRAG_SCREEN_FX = `
precision mediump float;
varying vec3 vUVW;
void main() {
  gl_FragColor = vec4(0.0);
  vec2 uv = vUVW.xy / vUVW.z;
  vec2 suv = vec2(uv.x, 1.0 - uv.y);
  float g1 = max(0.0, 1.0 - suv.x*2.0 - suv.y*2.6) * 0.10;
  float g2 = max(0.0, 1.0 - distance(suv, vec2(0.10,0.06))*5.0) * 0.05;
  float glare = clamp(g1+g2, 0.0, 0.13);
  vec2 vc = suv*2.0 - 1.0;
  float vignette = dot(vc,vc) * 0.09;
  float scan = (sin(suv.y*628.3)*0.5+0.5) * 0.010;
  float net = glare - vignette - scan;
  if(net > 0.0) gl_FragColor = vec4(1.0,1.0,1.0,net);
  else          gl_FragColor = vec4(0.0,0.0,0.0,-net);
}
`;

// Post: sharpen · contrast · sat · temp · bloom · vignette · grain
const FRAG_POST = `
precision mediump float;
uniform sampler2D uTex;
uniform float uInvW; uniform float uInvH;
uniform float uSharp;
uniform float uBright; uniform float uContrast;
uniform float uSat;
uniform float uTemp;
uniform float uVig;
uniform float uBloom;
uniform float uGrain; uniform float uTime;
varying vec3 vUVW;
void main() {
  vec2 uv = vUVW.xy / vUVW.z;
  vec2 px = vec2(uInvW, uInvH);
  vec4 c = texture2D(uTex, uv);

  if (uSharp > 0.001) {
    vec4 bl = (texture2D(uTex,uv+vec2(px.x,0.0))+texture2D(uTex,uv-vec2(px.x,0.0))+
               texture2D(uTex,uv+vec2(0.0,px.y))+texture2D(uTex,uv-vec2(0.0,px.y)))*0.25;
    c.rgb += (c.rgb - bl.rgb) * uSharp;
  }

  c.rgb = (c.rgb - 0.5) * uContrast + 0.5 + uBright;
  float lum = dot(c.rgb, vec3(0.2126,0.7152,0.0722));
  c.rgb = mix(vec3(lum), c.rgb, uSat);
  c.r += uTemp*0.15; c.b -= uTemp*0.2;

  if (uBloom > 0.001) {
    vec4 b4 = (texture2D(uTex,uv+vec2(px.x*5.0,0.0))+texture2D(uTex,uv-vec2(px.x*5.0,0.0))+
               texture2D(uTex,uv+vec2(0.0,px.y*5.0))+texture2D(uTex,uv-vec2(0.0,px.y*5.0)))*0.25;
    float bl2 = dot(b4.rgb, vec3(0.2126,0.7152,0.0722));
    c.rgb += b4.rgb * max(0.0, bl2 - 0.4) * uBloom * 3.5;
  }

  vec2 vc = uv - 0.5;
  c.rgb *= 1.0 - clamp(dot(vc,vc)/0.5,0.0,1.0) * uVig * 0.95;

  if (uGrain > 0.001) {
    vec2 sd = uv + fract(uTime * 0.1337);
    float gr = fract(sin(dot(sd,vec2(127.1,311.7)))*43758.5453) - 0.5;
    c.rgb += gr * uGrain * 0.1;
  }

  gl_FragColor = clamp(c, 0.0, 1.0);
}
`;

// ─── WebGL helpers ────────────────────────────────────────────────────────────

function mkShader(gl: WebGLRenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!; gl.shaderSource(s,src); gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) console.error('GL shader:',gl.getShaderInfoLog(s),src.slice(0,80));
  return s;
}
function mkProgram(gl: WebGLRenderingContext, frag: string, name='?') {
  const p = gl.createProgram()!;
  const vs=mkShader(gl,gl.VERTEX_SHADER,VERT);
  const fs=mkShader(gl,gl.FRAGMENT_SHADER,frag);
  gl.attachShader(p,vs); gl.attachShader(p,fs);
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))
    console.error(`GL link [${name}]:`,gl.getProgramInfoLog(p),'| frag compile:',gl.getShaderInfoLog(fs));
  return p;
}
function mkTex(gl: WebGLRenderingContext): WebGLTexture {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return t;
}
function uploadTex(gl: WebGLRenderingContext, tex: WebGLTexture, src: TexImageSource) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
}
function drawQuad(
  gl: WebGLRenderingContext, prog: WebGLProgram, tex: WebGLTexture,
  verts: Float32Array, uniforms?: Record<string, number | number[]>,
  tex2?: WebGLTexture | null,
) {
  gl.useProgram(prog);
  const buf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STREAM_DRAW);
  const aPos = gl.getAttribLocation(prog,'aPos'), aUVW = gl.getAttribLocation(prog,'aUVW');
  gl.enableVertexAttribArray(aPos); gl.enableVertexAttribArray(aUVW);
  gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,20,0);
  gl.vertexAttribPointer(aUVW,3,gl.FLOAT,false,20,8);
  gl.uniform1i(gl.getUniformLocation(prog,'uTex'),0);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
  if(tex2){
    gl.uniform1i(gl.getUniformLocation(prog,'uRec'),1);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,tex2);
    gl.activeTexture(gl.TEXTURE0);
  }
  if (uniforms) for (const [k,v] of Object.entries(uniforms)) {
    const loc = gl.getUniformLocation(prog,k);
    if (typeof v === 'number') gl.uniform1f(loc,v);
    else if (v.length===4) gl.uniform4fv(loc,v);
    else if (v.length===3) gl.uniform3fv(loc,v);
    else if (v.length===2) gl.uniform2fv(loc,v);
  }
  gl.drawArrays(gl.TRIANGLES, 0, verts.length/5);
  gl.deleteBuffer(buf);
}

type FBO = { fb: WebGLFramebuffer; tex: WebGLTexture };
function createFBO(gl: WebGLRenderingContext, w: number, h: number): FBO {
  const tex = mkTex(gl);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
  const fb = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER,fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  return {fb,tex};
}
function resizeFBO(gl: WebGLRenderingContext, fbo: FBO, w: number, h: number) {
  gl.bindTexture(gl.TEXTURE_2D,fbo.tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
}

// ─── Geometry ─────────────────────────────────────────────────────────────────

function bgVerts(): Float32Array {
  return new Float32Array([-1,-1,0,0,1, 1,-1,1,0,1, -1,1,0,1,1, 1,-1,1,0,1, 1,1,1,1,1, -1,1,0,1,1]);
}
function coverVerts(sw:number,sh:number,dw:number,dh:number): Float32Array {
  const sa=sw/sh,da=dw/dh; let u0=0,u1=1,v0=0,v1=1;
  if(sa>da){const m=(1-da/sa)/2;u0=m;u1=1-m;}
  else if(sa<da){const m=(1-sa/da)/2;v0=m;v1=1-m;}
  return new Float32Array([-1,-1,u0,v0,1, 1,-1,u1,v0,1, -1,1,u0,v1,1, 1,-1,u1,v0,1, 1,1,u1,v1,1, -1,1,u0,v1,1]);
}
// Scale recording to cover detected screen bounds — prevents oversized recording in auto mode
function boundsVerts(b:{x0:number;y0:number;x1:number;y1:number}, rW:number, rH:number, W:number, H:number): Float32Array {
  const bW=(b.x1-b.x0)*W, bH=(b.y1-b.y0)*H;
  const bAsp=bW/bH, rAsp=rW/rH;
  let u0=0,u1=1,v0=0,v1=1;
  if(rAsp>bAsp){const m=(1-bAsp/rAsp)/2;u0=m;u1=1-m;}
  else if(rAsp<bAsp){const m=(1-rAsp/bAsp)/2;v0=m;v1=1-m;}
  // NDC positions — x: 0→1 maps to -1→1; y: image-top(0)→NDC+1, image-bottom(1)→NDC-1
  const nx0=b.x0*2-1, nx1=b.x1*2-1;
  const nyT=1-b.y0*2, nyB=1-b.y1*2; // T=top of screen (higher NDC y), B=bottom
  // v0=recording bottom at screen bottom, v1=recording top at screen top (FLIP_Y=true)
  return new Float32Array([
    nx0,nyB,u0,v0,1, nx1,nyB,u1,v0,1, nx0,nyT,u0,v1,1,
    nx1,nyB,u1,v0,1, nx1,nyT,u1,v1,1, nx0,nyT,u0,v1,1,
  ]);
}
// Place the full recording (UV 0→1) at `scale` fraction of canvas height, centered at (cx,cy)
// scale=0.40 ≈ phone screen filling 40% of canvas height.  Full recording always visible.
// scaleW/scaleH are NDC half-extents: 0.20 = recording spans 40% of canvas dimension.
// Independent W and H let user match any phone screen aspect ratio (iPhone 14 Pro = ~46:100 vs 9:16).
function zoomVerts(cx:number,cy:number,scaleW:number,scaleH:number): Float32Array {
  const ndcCx=cx*2-1, ndcCy=1-cy*2;
  return new Float32Array([
    ndcCx-scaleW, ndcCy-scaleH, 0,0,1,
    ndcCx+scaleW, ndcCy-scaleH, 1,0,1,
    ndcCx-scaleW, ndcCy+scaleH, 0,1,1,
    ndcCx+scaleW, ndcCy-scaleH, 1,0,1,
    ndcCx+scaleW, ndcCy+scaleH, 1,1,1,
    ndcCx-scaleW, ndcCy+scaleH, 0,1,1,
  ]);
}
function coverUVBounds(sw:number,sh:number,dw:number,dh:number):[number,number,number,number]{
  const sa=sw/sh,da=dw/dh; let u0=0,u1=1,v0=0,v1=1;
  if(sa>da){const m=(1-da/sa)/2;u0=m;u1=1-m;}
  else if(sa<da){const m=(1-sa/da)/2;v0=m;v1=1-m;}
  return [u0,u1,v0,v1];
}

// 32×32 subdivided mesh — ultra-smooth perspective warp, zero corner artefacts
function pinVerts(pins: Quad, W: number, H: number, N=32): Float32Array | null {
  try {
    // V-flipped: UNPACK_FLIP_Y_WEBGL makes V=0 → image bottom, so map top corners to V=1
    const src: Quad = [{x:0,y:1},{x:1,y:1},{x:1,y:0},{x:0,y:0}];
    const hFwd = computeHomography(src, pins);
    const hInv = invertHomography(hFwd);
    function uv2v(u:number,v:number): number[]|null {
      const dw=hFwd[6]*u+hFwd[7]*v+hFwd[8]; if(Math.abs(dw)<1e-10) return null;
      const sx=(hFwd[0]*u+hFwd[1]*v+hFwd[2])/dw, sy=(hFwd[3]*u+hFwd[4]*v+hFwd[5])/dw;
      const w=hInv[6]*sx+hInv[7]*sy+hInv[8]; if(!isFinite(w)||Math.abs(w)<1e-10) return null;
      return [(sx/W)*2-1, 1-(sy/H)*2, u*w, v*w, w];
    }
    const out: number[]=[];
    for(let r=0;r<N;r++) for(let c=0;c<N;c++){
      const u0=c/N,u1=(c+1)/N,v0=r/N,v1=(r+1)/N;
      const p00=uv2v(u0,v0),p10=uv2v(u1,v0),p01=uv2v(u0,v1),p11=uv2v(u1,v1);
      if(!p00||!p10||!p01||!p11) return null;
      out.push(...p00,...p10,...p01, ...p10,...p11,...p01);
    }
    return new Float32Array(out);
  } catch { return null; }
}

// ─── Color utils ─────────────────────────────────────────────────────────────

// Find green-screen corners using three independent detection methods so any real
// screen is caught even when detectKeyColor returns a slightly wrong shade.
// Returns bounding-box quad [TL,TR,BR,BL] as 0-1 fractions, or null.
function detectScreenCorners(img: HTMLImageElement, keyHex: string): Quad | null {
  const W=Math.min(img.naturalWidth,480), H=Math.min(img.naturalHeight,480);
  const c=document.createElement('canvas'); c.width=W; c.height=H;
  const ctx=c.getContext('2d')!; ctx.drawImage(img,0,0,W,H);
  const d=ctx.getImageData(0,0,W,H).data;
  const [kr,kg,kb]=hexToRgb(keyHex);
  // Dominant channel of the detected key color
  const keyDom=kr>=kg&&kr>=kb?0:kg>=kb?1:2;
  const sm=(e0:number,e1:number,x:number)=>{const t=Math.max(0,Math.min(1,(x-e0)/(e1-e0)));return t*t*(3-2*t);};
  const pts:{x:number;y:number}[]=[];
  // Fixed generous thresholds — never use user's render sliders for spatial detection
  const T=0.50, S=0.10;
  for(let y=0;y<H;y++){
    for(let x=0;x<W;x++){
      const i=(y*W+x)*4;
      const r=d[i]/255,g=d[i+1]/255,b=d[i+2]/255;
      const mx=Math.max(r,g,b),mn=Math.min(r,g,b);
      const sat=mx>0.001?(mx-mn)/mx:0;
      // Method A: shader-sim with generous fixed thresholds
      const cLen=Math.max(0.001,Math.sqrt(r*r+g*g+b*b));
      const kLen=Math.max(0.001,Math.sqrt(kr*kr+kg*kg+kb*kb));
      const cosA=(r*kr+g*kg+b*kb)/(cLen*kLen);
      const dist=Math.sqrt((r-kr)**2+(g-kg)**2+(b-kb)**2);
      const combined=dist*(1+Math.max(0,1-cosA)*0.5);
      const satGate=sm(0.06,0.22,sat);
      const rawA=sm(T-S,T+S,combined);
      const alphaA=(1-satGate)+rawA*rawA*(3-2*rawA)*satGate;
      // Method B: raw distance — catches when detected key color is slightly off
      const isB=dist<0.52;
      // Method C: dominant-channel check — works even if key color is entirely wrong shade
      const ch=[r,g,b];
      const domVal=ch[keyDom];
      const oth1=ch[(keyDom+1)%3], oth2=ch[(keyDom+2)%3];
      const otherMax=Math.max(oth1,oth2);
      const isC=domVal>0.18&&domVal>otherMax*1.22&&sat>0.18;
      if(alphaA<0.5||isB||isC) pts.push({x,y});
    }
  }
  if(pts.length<100){console.warn('detectScreenCorners: only',pts.length,'matched px');return null;}
  // Percentile 2–98: removes stray outlier pixels before computing bounds
  const xs=pts.map(p=>p.x).sort((a,b)=>a-b);
  const ys=pts.map(p=>p.y).sort((a,b)=>a-b);
  const lo=Math.max(0,Math.floor(pts.length*0.02));
  const hi=Math.min(pts.length-1,Math.ceil(pts.length*0.98)-1);
  const x0=xs[lo],x1=xs[hi],y0=ys[lo],y1=ys[hi];
  // Reject if bounds span >90% of canvas in both dims — almost certainly noise/background
  if((x1-x0)/W>0.90&&(y1-y0)/H>0.90){
    console.warn('detectScreenCorners: matched region too large, discarding');
    return null;
  }
  // 1.5% outward expansion to cover anti-aliased / rounded screen corners
  const padX=W*0.015,padY=H*0.015;
  const q=[
    {x:Math.max(0,(x0-padX)/W), y:Math.max(0,(y0-padY)/H)},
    {x:Math.min(1,(x1+padX)/W), y:Math.max(0,(y0-padY)/H)},
    {x:Math.min(1,(x1+padX)/W), y:Math.min(1,(y1+padY)/H)},
    {x:Math.max(0,(x0-padX)/W), y:Math.min(1,(y1+padY)/H)},
  ] as Quad;
  console.log('Auto corners (%):', q.map(p=>`(${(p.x*100).toFixed(1)},${(p.y*100).toFixed(1)})`).join(' '));
  return q;
}

// Scan the whole mockup and return the bounding box of pixels that match the key color
function detectScreenBounds(img: HTMLImageElement, keyHex: string): {x0:number;y0:number;x1:number;y1:number}|null {
  const W=Math.min(img.naturalWidth,480), H=Math.min(img.naturalHeight,480);
  const c=document.createElement('canvas'); c.width=W; c.height=H;
  const ctx=c.getContext('2d')!; ctx.drawImage(img,0,0,W,H);
  const d=ctx.getImageData(0,0,W,H).data;
  const [kr,kg,kb]=hexToRgb(keyHex);
  let minX=W,minY=H,maxX=0,maxY=0,count=0;
  for(let y=0;y<H;y++){
    for(let x=0;x<W;x++){
      const i=(y*W+x)*4;
      const r=d[i]/255,g=d[i+1]/255,b=d[i+2]/255;
      const mx=Math.max(r,g,b),mn=Math.min(r,g,b);
      if(mx<0.06||mx-mn<mx*0.22) continue;
      const dist=Math.sqrt((r-kr)**2+(g-kg)**2+(b-kb)**2);
      if(dist<0.38){
        if(x<minX)minX=x; if(x>maxX)maxX=x;
        if(y<minY)minY=y; if(y>maxY)maxY=y;
        count++;
      }
    }
  }
  if(count<80) return null;
  return {x0:minX/W, y0:minY/H, x1:maxX/W, y1:maxY/H};
}

// Sample center 60% of mockup — avoids device frame; finer buckets + real average for accuracy
function detectKeyColor(img: HTMLImageElement): string {
  const W=Math.min(img.naturalWidth,400), H=Math.min(img.naturalHeight,400);
  const c=document.createElement('canvas'); c.width=W; c.height=H;
  const ctx=c.getContext('2d')!; ctx.drawImage(img,0,0,W,H);
  const x0=Math.floor(W*0.2), y0=Math.floor(H*0.2), sw=Math.floor(W*0.6), sh=Math.floor(H*0.6);
  const d=ctx.getImageData(x0,y0,sw,sh).data;
  const hist: Record<string,{count:number;sr:number;sg:number;sb:number}>={};
  for(let i=0;i<d.length;i+=4){
    const r=d[i],g=d[i+1],b=d[i+2];
    const mx=Math.max(r,g,b),mn=Math.min(r,g,b);
    const mid=r+g+b-mx-mn; // second-highest channel
    if(mx<55||mx>248) continue;
    if((mx-mn)/mx<0.35) continue; // require meaningful saturation — rejects greys and most skin
    if(mx<mid*1.35) continue; // require clear channel dominance — rejects warm mixed colors like skin
    const k=`${Math.round(r/16)},${Math.round(g/16)},${Math.round(b/16)}`;
    if(!hist[k]) hist[k]={count:0,sr:0,sg:0,sb:0};
    hist[k].count++; hist[k].sr+=r; hist[k].sg+=g; hist[k].sb+=b;
  }
  type Bin={count:number;sr:number;sg:number;sb:number};
  const score=(e:Bin)=>{
    const r=e.sr/e.count,g=e.sg/e.count,b=e.sb/e.count;
    const mx=Math.max(r,g,b),mn=Math.min(r,g,b);
    return e.count*1000+(mx>0?(mx-mn)/mx*255:0);
  };
  const best=Object.values(hist).sort((a,b)=>score(b)-score(a))[0];
  if(!best) return '#00ff00';
  const ar=Math.round(best.sr/best.count);
  const ag=Math.round(best.sg/best.count);
  const ab=Math.round(best.sb/best.count);
  return `#${ar.toString(16).padStart(2,'0')}${ag.toString(16).padStart(2,'0')}${ab.toString(16).padStart(2,'0')}`;
}
function hexToRgb(h:string):[number,number,number]{
  const v=parseInt(h.slice(1),16); return [(v>>16&255)/255,(v>>8&255)/255,(v&255)/255];
}
function dl(blob:Blob,name:string){
  const url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
}

// ─── Types & presets ──────────────────────────────────────────────────────────

type Enhance={sharp:number;bright:number;contrast:number;sat:number;vignette:number;temp:number;grain:number;bloom:number};
type GradeName='none'|'natural'|'cinematic'|'vivid';
type TextItem={id:string;text:string;x:number;y:number;size:number;color:string;family:string;bold:boolean;italic:boolean;};
const GRADES:Record<GradeName,Enhance>={
  none:      {sharp:0,    bright:0,     contrast:1.0,  sat:1.0,  vignette:0,    temp:0,     grain:0,    bloom:0   },
  natural:   {sharp:0.45, bright:0.018, contrast:1.06, sat:0.95, vignette:0.12, temp:0.05,  grain:0.08, bloom:0   },
  cinematic: {sharp:0.35, bright:-0.02, contrast:1.18, sat:0.72, vignette:0.42, temp:-0.08, grain:0.28, bloom:0.1 },
  vivid:     {sharp:0.65, bright:0.02,  contrast:1.24, sat:1.32, vignette:0.14, temp:0.04,  grain:0.04, bloom:0.08},
};

type Preset={name:string;mode:Mode;grade:GradeName;enhance:Enhance;keyColor:string;keyThresh:number;keySoft:number;pins?:Quad};
type Mode='manual'|'auto';
type ExportFmt='png'|'webm';
type Quality='high'|'ultra';
type ExportRatio='16:9'|'9:16'|'1:1';
type TemplateItem={id:string;name:string;thumb:string;mockupData:string;pins:Quad;grade:GradeName;enhance:Enhance;ratio:ExportRatio;borderWidth:number;borderRadius:number;topBorder:boolean;edgeBlend:boolean;mockupOp:number;};

// ─── Library ─────────────────────────────────────────────────────────────────

interface UPhoto{id:string;urls:{thumb:string;regular:string};alt_description:string;user:{name:string}}

const LIB_CATS=[
  {label:'💻 MacBook',   q:'macbook mockup clean desk'},
  {label:'📱 iPhone',    q:'iphone smartphone mockup minimal'},
  {label:'⬛ iPad',      q:'ipad tablet mockup flat lay'},
  {label:'🖥️ iMac',     q:'imac desktop workspace minimal'},
  {label:'📐 Workspace', q:'clean minimal desk setup laptop'},
  {label:'🎨 Gradient',  q:'gradient abstract tech background'},
];

function Library({onSelectPhoto}:{onSelectPhoto:(url:string,name:string)=>void}){
  const [apiKey,setApiKey]=useState(()=>localStorage.getItem('unsplash-key')||'');
  const [keyInput,setKeyInput]=useState(apiKey);
  const [catIdx,setCatIdx]=useState(0);
  const [query,setQuery]=useState('');
  const [results,setResults]=useState<UPhoto[]>([]);
  const [loading,setLoading]=useState(false);

  const search=useCallback(async(q:string)=>{
    const k=localStorage.getItem('unsplash-key'); if(!k) return;
    setLoading(true);
    try{
      const res=await fetch(`https://api.unsplash.com/search/photos?query=${encodeURIComponent(q)}&per_page=30&orientation=landscape&client_id=${k}`);
      const data=await res.json(); setResults(data.results||[]);
    }catch{setResults([]);}
    setLoading(false);
  },[]);

  useEffect(()=>{if(apiKey) search(query||LIB_CATS[catIdx].q);},[apiKey,catIdx]);

  const saveKey=(k:string)=>{localStorage.setItem('unsplash-key',k);setApiKey(k);};

  if(!apiKey) return(
    <div style={{padding:'16px 14px'}}>
      <div style={{fontSize:13,fontWeight:700,marginBottom:8,color:'var(--text)'}}>Connect Unsplash</div>
      <p style={{fontSize:10.5,color:'var(--muted)',lineHeight:1.75,marginBottom:10}}>
        Browse thousands of real device photos. Get a free key at{' '}
        <a href="https://unsplash.com/developers" target="_blank" rel="noreferrer"
           style={{color:'var(--blue)'}}>unsplash.com/developers</a>
        {' '}— takes 2 minutes.
      </p>
      <input style={{width:'100%',padding:'7px 10px',background:'var(--surface)',border:'1px solid var(--border)',
        borderRadius:7,fontSize:11,color:'var(--text)',outline:'none',marginBottom:8,boxSizing:'border-box'}}
        placeholder="Paste Unsplash Access Key…"
        value={keyInput} onChange={e=>setKeyInput(e.target.value)}
        onKeyDown={e=>{if(e.key==='Enter'&&keyInput.trim()) saveKey(keyInput.trim());}}/>
      <button style={{width:'100%',padding:9,background:'var(--accent)',color:'#fff',border:'none',
        borderRadius:7,fontWeight:700,fontSize:12,cursor:'pointer'}}
        onClick={()=>keyInput.trim()&&saveKey(keyInput.trim())}>Connect →</button>
      <div style={{marginTop:14,padding:'10px 12px',background:'var(--surface)',borderRadius:8,
        border:'1px solid var(--border)',fontSize:10,color:'var(--muted)',lineHeight:2}}>
        <div>① unsplash.com/developers → New Application</div>
        <div>② Accept terms, fill basic details</div>
        <div>③ Copy the Access Key and paste above</div>
      </div>
    </div>
  );

  return(<>
    <div style={{display:'flex',flexWrap:'wrap',gap:4,padding:'8px 14px',borderBottom:'1px solid var(--border)'}}>
      {LIB_CATS.map((c,i)=>(
        <div key={i} onClick={()=>{setCatIdx(i);search(c.q);}}
          style={{padding:'4px 9px',borderRadius:14,fontSize:10,fontWeight:600,cursor:'pointer',
            border:`1.5px solid ${catIdx===i?'var(--accent)':'var(--border)'}`,
            background:catIdx===i?'rgba(124,106,247,.12)':'var(--surface)',
            color:catIdx===i?'var(--text)':'var(--muted)',whiteSpace:'nowrap'}}>
          {c.label}
        </div>
      ))}
    </div>
    <div style={{display:'flex',gap:6,padding:'8px 14px',borderBottom:'1px solid var(--border)'}}>
      <input style={{flex:1,padding:'6px 9px',background:'var(--surface)',border:'1px solid var(--border)',
        borderRadius:7,fontSize:11,color:'var(--text)',outline:'none'}}
        placeholder="Search…" value={query} onChange={e=>setQuery(e.target.value)}
        onKeyDown={e=>{if(e.key==='Enter'&&query.trim()) search(query.trim());}}/>
      <button style={{padding:'6px 12px',background:'var(--accent)',color:'#fff',border:'none',
        borderRadius:7,fontSize:11,fontWeight:600,cursor:'pointer'}}
        onClick={()=>query.trim()&&search(query.trim())}>→</button>
    </div>
    {loading&&<div style={{padding:'20px',textAlign:'center',color:'var(--muted)',fontSize:11}}>Loading…</div>}
    {!loading&&results.length===0&&<div style={{padding:'20px',textAlign:'center',color:'var(--muted)',fontSize:11,lineHeight:1.7}}>
      No results — try a different search.
    </div>}
    {!loading&&results.length>0&&(
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:6,padding:'10px 14px',overflowY:'auto'}}>
        {results.map(p=>(
          <div key={p.id} onClick={()=>onSelectPhoto(p.urls.regular+'&w=2560&q=90',p.alt_description||'Unsplash')}
            style={{borderRadius:8,overflow:'hidden',cursor:'pointer',aspectRatio:'4/3',
              background:'var(--surface)',border:'1.5px solid var(--border)',
              transition:'all .15s',position:'relative'}}
            onMouseEnter={e=>{(e.currentTarget as HTMLElement).style.borderColor='var(--accent)';(e.currentTarget as HTMLElement).style.transform='scale(1.03)';}}
            onMouseLeave={e=>{(e.currentTarget as HTMLElement).style.borderColor='var(--border)';(e.currentTarget as HTMLElement).style.transform='scale(1)';}}>
            <img src={p.urls.thumb} alt={p.alt_description||''} loading="lazy"
              style={{width:'100%',height:'100%',objectFit:'cover',display:'block'}}/>
          </div>
        ))}
      </div>
    )}
    <div style={{padding:'6px 14px 10px',fontSize:9,color:'#383858',borderTop:'1px solid var(--border)',
      display:'flex',justifyContent:'space-between'}}>
      <span>Photos via Unsplash</span>
      <span style={{cursor:'pointer',textDecoration:'underline'}}
        onClick={()=>{localStorage.removeItem('unsplash-key');setApiKey('');setKeyInput('');}}>
        Disconnect
      </span>
    </div>
  </>);
}

// ─── CSS ─────────────────────────────────────────────────────────────────────

const CSS = `
:root{--bg:#07070F;--panel:#0D0D18;--surface:#12121C;--border:#1C1C2E;
  --accent:#7C6AF7;--blue:#5B9CF6;--text:#EEEEF5;--muted:#5A5A80}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,#root{height:100%;overflow:hidden}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:13px}
.app{display:flex;flex-direction:column;height:100vh}

.hdr{display:flex;align-items:center;justify-content:space-between;
  height:48px;padding:0 18px;background:var(--panel);
  border-bottom:1px solid var(--border);flex-shrink:0;
  box-shadow:0 1px 0 rgba(255,255,255,.04)}
.logo{display:flex;align-items:center;gap:9px;font-weight:700;font-size:13.5px;letter-spacing:-.4px}
.logo-gem{width:20px;height:20px;background:linear-gradient(135deg,#7C6AF7,#5B9CF6);border-radius:5px;
  box-shadow:0 2px 10px rgba(124,106,247,.4)}
.hdr-r{display:flex;align-items:center;gap:8px}

.btn{display:inline-flex;align-items:center;gap:5px;padding:6px 13px;
  border-radius:7px;font-size:11.5px;font-weight:500;border:none;cursor:pointer;
  transition:all .13s;white-space:nowrap}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-ghost:hover:not(:disabled){background:var(--surface);color:var(--text)}
.btn-danger{background:rgba(239,68,68,.1);color:#F87171;border:1px solid rgba(239,68,68,.25)}
.btn-danger:hover{background:rgba(239,68,68,.2)}
.btn-export{background:linear-gradient(130deg,#7C6AF7,#5B9CF6);color:#fff;font-weight:700;font-size:12px;
  box-shadow:0 4px 16px rgba(124,106,247,.35)}
.btn-export:hover:not(:disabled){opacity:.9;transform:translateY(-1px);box-shadow:0 6px 20px rgba(124,106,247,.5)}
.btn:disabled{opacity:.3;cursor:not-allowed}

.rec-badge{display:flex;align-items:center;gap:6px;padding:4px 11px;
  background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.35);
  border-radius:18px;font-size:11px;color:#F87171;font-weight:600}
.rec-dot{width:6px;height:6px;border-radius:50%;background:#EF4444;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}

.body{display:flex;flex:1;overflow:hidden;min-height:0}
.sidebar{width:260px;flex-shrink:0;overflow-y:auto;background:var(--panel);
  border-right:1px solid var(--border);display:flex;flex-direction:column}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

.top-tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0}
.top-tab{flex:1;padding:10px;font-size:11px;font-weight:600;border:none;
  background:transparent;color:var(--muted);cursor:pointer;transition:all .13s;
  border-bottom:2px solid transparent}
.top-tab.active{color:var(--text);border-bottom-color:var(--accent);background:rgba(124,106,247,.06)}

.sec{padding:11px 14px;border-bottom:1px solid var(--border)}
.sec-title{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted);margin-bottom:9px}

.mode-row{display:flex;gap:5px;padding:9px 14px;border-bottom:1px solid var(--border)}
.mode-btn{flex:1;padding:6px 8px;border-radius:7px;font-size:11px;font-weight:600;
  border:1.5px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;transition:all .13s}
.mode-btn.active{border-color:var(--accent);background:rgba(124,106,247,.1);color:var(--text)}

.drop{border:1.5px dashed var(--border);border-radius:9px;padding:11px 10px;
  text-align:center;cursor:pointer;transition:all .18s;background:var(--surface)}
.drop:hover,.drop.over{border-color:var(--accent);background:rgba(124,106,247,.05)}
.drop-ico{font-size:17px;margin-bottom:4px}
.drop-tx{font-size:10.5px;color:var(--muted);line-height:1.5}
.drop-tx strong{display:block;color:var(--text);font-size:11px;margin-bottom:2px}
.chip{margin-top:5px;padding:3px 8px;border-radius:5px;font-size:9px;font-weight:500;
  background:rgba(91,156,246,.1);color:#5B9CF6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

.grade-row{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:9px}
.grade-btn{padding:4px 10px;border-radius:16px;font-size:10px;font-weight:600;
  border:1.5px solid var(--border);background:var(--surface);color:var(--muted);cursor:pointer;transition:all .13s}
.grade-btn.active{border-color:var(--accent);background:rgba(124,106,247,.14);color:var(--text)}

.sl-row{display:flex;flex-direction:column;gap:4px;margin-bottom:7px}
.sl-lbl{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted)}
.sl-lbl span{color:var(--text);font-variant-numeric:tabular-nums}
input[type=range]{width:100%;accent-color:var(--accent);cursor:pointer;margin:1px 0}

.expand-row{display:flex;align-items:center;justify-content:space-between;cursor:pointer;margin-bottom:7px}
.expand-lbl{font-size:9px;font-weight:700;color:var(--muted);letter-spacing:.8px;text-transform:uppercase}
.expand-arrow{font-size:9px;color:var(--muted);transition:transform .15s;user-select:none}
.expand-arrow.open{transform:rotate(180deg)}

input[type=color]{width:32px;height:32px;border-radius:6px;border:1.5px solid var(--border);
  padding:2px;cursor:pointer;background:var(--surface);flex-shrink:0}

.trim-wrap{position:relative;height:18px;margin:6px 0}
.trim-track{position:absolute;top:50%;left:0;right:0;height:3px;
  background:var(--border);border-radius:2px;transform:translateY(-50%)}
.trim-fill{position:absolute;top:50%;height:3px;background:var(--accent);
  border-radius:2px;transform:translateY(-50%)}
.trim-wrap input[type=range]{position:absolute;width:100%;top:0;height:100%;
  opacity:0;cursor:pointer;margin:0;pointer-events:all}
.trim-wrap input[type=range]:first-of-type{z-index:2}

.preset-list{display:flex;flex-direction:column;gap:4px;margin-bottom:8px}
.preset-item{display:flex;align-items:center;justify-content:space-between;
  padding:5px 8px;border-radius:7px;background:var(--surface);
  border:1px solid var(--border);cursor:pointer;transition:border-color .13s}
.preset-item:hover{border-color:var(--accent)}
.preset-item span{font-size:11px;color:var(--text)}
.preset-item button{font-size:10px;color:var(--muted);background:none;border:none;cursor:pointer;padding:0 2px}
.preset-input{flex:1;padding:5px 8px;background:var(--surface);border:1px solid var(--border);
  border-radius:7px;font-size:11px;color:var(--text);outline:none}
.preset-input::placeholder{color:var(--muted)}
.preset-input:focus{border-color:var(--accent)}
.batch-drop{border:1.5px dashed var(--border);border-radius:9px;padding:8px 10px;
  text-align:center;cursor:pointer;transition:all .18s;background:var(--surface);margin-bottom:6px}
.batch-drop:hover{border-color:var(--accent);background:rgba(124,106,247,.05)}
.batch-item{display:flex;align-items:center;justify-content:space-between;
  padding:4px 8px;border-radius:6px;background:var(--surface);border:1px solid var(--border);
  margin-bottom:3px;font-size:10px;color:var(--text)}
.batch-item.active{border-color:var(--accent);background:rgba(124,106,247,.1);color:var(--accent)}
.batch-item.done{opacity:0.38}

.canvas-area{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;
  position:relative;background:radial-gradient(ellipse at center, #0E0E1C 0%, #07070F 100%)}
.zoom-pill{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);
  display:flex;align-items:center;gap:2px;
  background:rgba(10,10,20,0.92);border:1px solid var(--border);
  border-radius:20px;padding:3px 8px;box-shadow:0 4px 20px rgba(0,0,0,.6);z-index:50;user-select:none}
.zoom-btn{background:none;border:none;color:var(--muted);font-size:14px;cursor:pointer;
  padding:2px 7px;border-radius:12px;line-height:1;transition:all .1s}
.zoom-btn:hover{background:var(--surface);color:var(--text)}
.zoom-pct{font-size:11px;color:var(--text);cursor:pointer;min-width:38px;text-align:center;padding:0 4px}
.canvas-wrap{position:relative;
  box-shadow:0 32px 100px rgba(0,0,0,.9),0 12px 36px rgba(0,0,0,.7),0 0 0 1px rgba(255,255,255,.05)}
/* L-bracket corner pins — sit outside the corner, never obscure the screen edge */
.pin-corner{position:absolute;width:20px;height:20px;cursor:crosshair;z-index:10;
  user-select:none;touch-action:none;box-sizing:border-box;}
.pin-corner .arm-h,.pin-corner .arm-v{position:absolute;background:rgba(255,255,255,0.95);
  transition:background .1s;}
.pin-corner .arm-h{height:2px;width:12px;}
.pin-corner .arm-v{width:2px;height:12px;}
.pin-corner:hover .arm-h,.pin-corner:hover .arm-v{background:#5B9CF6;}
.pin-corner.active .arm-h,.pin-corner.active .arm-v{background:#7C6AF7;}
.empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:10px;pointer-events:none;color:var(--muted)}
.empty-ico{font-size:44px;opacity:.3}
.empty h3{font-size:15px;font-weight:600;color:rgba(238,238,245,.55)}
.empty p{font-size:11px;text-align:center;max-width:240px;line-height:1.75}

.modal-ov{position:fixed;inset:0;background:rgba(0,0,0,.8);
  display:flex;align-items:center;justify-content:center;z-index:200;animation:fin .14s}
.modal{background:var(--panel);border:1px solid var(--border);border-radius:14px;
  width:340px;overflow:hidden;box-shadow:0 32px 80px rgba(0,0,0,.75)}
.modal-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;border-bottom:1px solid var(--border)}
.modal-hdr h2{font-size:14px;font-weight:700}
.modal-x{background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;padding:2px 6px}
.modal-x:hover{color:var(--text)}
.modal-body{padding:18px}
.m-stat{display:flex;justify-content:space-between;margin-bottom:10px;font-size:11.5px;color:var(--muted)}
.m-stat strong{color:var(--text)}
.m-lbl{font-size:9px;color:var(--muted);letter-spacing:.8px;text-transform:uppercase;font-weight:700;margin-bottom:7px;margin-top:12px}
.fmt-tabs{display:flex;gap:5px;margin-bottom:4px}
.fmt-tab{flex:1;padding:9px;border-radius:7px;font-size:11px;font-weight:600;
  border:1.5px solid var(--border);background:var(--surface);color:var(--muted);cursor:pointer;transition:all .13s;text-align:center}
.fmt-tab.active{border-color:var(--accent);background:rgba(124,106,247,.1);color:var(--text)}
.q-row{display:flex;gap:5px;margin-bottom:9px}
.q-btn{flex:1;padding:8px;border-radius:7px;font-size:10.5px;font-weight:600;
  border:1.5px solid var(--border);background:var(--surface);color:var(--muted);cursor:pointer;transition:all .13s;text-align:center;line-height:1.7}
.q-btn.active{border-color:var(--accent);background:rgba(124,106,247,.1);color:var(--text)}
.q-note{font-size:10px;color:var(--muted);margin-bottom:12px;line-height:1.7}

.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
  background:#1A1A2E;border:1px solid var(--border);border-radius:9px;
  padding:9px 18px;font-size:12px;box-shadow:0 12px 40px rgba(0,0,0,.65);z-index:999;
  animation:fin .18s}
.toast.err{border-color:#F87171;color:#F87171}
@keyframes fin{from{opacity:0;transform:translateX(-50%) translateY(6px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}
`;

// ─── DropZone ────────────────────────────────────────────────────────────────

function DropZone({label,hint,file,onFile,accept,icon}:{
  label:string;hint:string;file:string;onFile:(f:File)=>void;accept:string;icon:string;
}){
  const [over,setOver]=useState(false);
  const open=()=>{const i=document.createElement('input');i.type='file';i.accept=accept;
    i.onchange=e=>{const f=(e.target as HTMLInputElement).files?.[0];if(f)onFile(f);};i.click();};
  return(
    <div className={`drop${over?' over':''}`}
      onDragOver={e=>{e.preventDefault();setOver(true)}} onDragLeave={()=>setOver(false)}
      onDrop={e=>{e.preventDefault();setOver(false);const f=e.dataTransfer.files[0];if(f)onFile(f)}}
      onClick={open}>
      <div className="drop-ico">{icon}</div>
      <div className="drop-tx"><strong>{file?'Replace file':label}</strong>{hint}</div>
      {file&&<div className="chip">{file}</div>}
    </div>
  );
}

function Slider({label,min,max,step,value,onChange}:{
  label:string;min:number;max:number;step:number;value:number;onChange:(v:number)=>void;
}){
  return(
    <div className="sl-row">
      <div className="sl-lbl">{label}<span>{value.toFixed(2)}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(Number(e.target.value))}/>
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App(){
  const [tab,       setTab]      = useState<'edit'|'library'>('edit');
  const [mode,      setMode]     = useState<Mode>('manual');
  const [mockupSrc, setMockupSrc]= useState<string|null>(null);
  const [mockupIsV, setMockupIsV]= useState(false);
  const [recSrc,    setRecSrc]   = useState<string|null>(null);
  const [recNative, setRecNative]= useState({w:1920,h:1080});
  const [csz,       setCsz]      = useState({w:840,h:520});
  const [native,    setNative]   = useState({w:840,h:520});
  const [pins,      setPins]     = useState<Quad>(()=>defaultCorners(840,520));
  const [keyColor,  setKeyColor] = useState('#00ff00');
  const [keyThresh, setKeyThresh]= useState(0.44);
  const [keySoft,   setKeySoft]  = useState(0.09);
  const [keySpill,  setKeySpill] = useState(0.90);
  const [grade,     setGrade]    = useState<GradeName>('natural');
  const [enhance,   setEnhance]  = useState<Enhance>(GRADES.natural);
  const [expandEnh, setExpandEnh]= useState(false);
  const [trimIn,    setTrimIn]   = useState(0);
  const [trimOut,   setTrimOut]  = useState(1);
  const [recDur,    setRecDur]   = useState(0);
  const [presets,   setPresets]  = useState<Preset[]>(()=>{try{return JSON.parse(localStorage.getItem('mockup-presets')||'[]');}catch{return [];}});
  const [presetName,setPresetName]=useState('');
  const [showPre,   setShowPre]  = useState(false);
  const [mockupFile,setMockupFile]=useState('');
  const [recFile,   setRecFile]  = useState('');
  const [activePin, setActivePin]= useState<number|null>(null);
  const [showExp,   setShowExp]  = useState(false);
  const [expFmt,    setExpFmt]   = useState<ExportFmt>('png');
  const [quality,   setQuality]  = useState<Quality>('high');
  const [exportRatio,setExportRatio]=useState<ExportRatio>('16:9');
  const [bgFill,    setBgFill]    = useState<'blur'|'black'>('blur');
  const [isRec,     setIsRec]    = useState(false);
  const [recTime,   setRecTime]  = useState(0);
  const [isExporting,   setIsExporting]   = useState(false);
  const [exportProgress,setExportProgress]= useState(0);
  const [toast,     setToast]    = useState<{msg:string;err?:boolean}|null>(null);
  const [zoom,      setZoom]     = useState(1.0);
  const [pan,       setPan]      = useState({x:0,y:0});
  const [edgeBlend,  setEdgeBlend]  = useState(true);
  const [borderWidth,  setBorderWidth]  = useState(2);
  const [borderRadius, setBorderRadius] = useState(0);
  const [topBorder,    setTopBorder]    = useState(false);
  const [templates,  setTemplates]  = useState<TemplateItem[]>(()=>{try{return JSON.parse(localStorage.getItem('mockup_templates')||'[]');}catch{return [];}});
  const [selTpls,    setSelTpls]    = useState<Set<string>>(()=>new Set());
  const [tplBatchIdx,setTplBatchIdx]= useState<number|null>(null);
  const [mockupOp,   setMockupOp]   = useState(1.0);
  const [lumaKey,    setLumaKey]    = useState(0.0);
  const [lumaSoft,   setLumaSoft]   = useState(0.08);
  const [chromaKey,  setChromaKey]  = useState(0.0);
  const [punchThrough, setPunchThrough] = useState(false);
  const [autoPlacement, setAutoPlacement]  = useState<'fit'|'corners'|'fill'>('fit');
  const [autoFitScaleW,  setAutoFitScaleW] = useState(0.20); // NDC half-width (×2 = % of canvas width)
  const [autoFitScaleH,  setAutoFitScaleH] = useState(0.38); // NDC half-height
  const [autoFitY,       setAutoFitY]      = useState(0.50); // vertical center (0=top, 1=bottom)
  const [autoRecScale,   setAutoRecScale]   = useState(1.0);
  const [camHole,    setCamHole]    = useState(false);
  const [camX,       setCamX]       = useState(0.50);
  const [camY,       setCamY]       = useState(0.04);
  const [camRadius,  setCamRadius]  = useState(22);
  const [textItems,  setTextItems]  = useState<TextItem[]>([]);
  const [selTextId,  setSelTextId]  = useState<string|null>(null);
  const [audioSrc,   setAudioSrc]   = useState<string|null>(null);
  const [audioName,  setAudioName]  = useState('');
  const [audioVol,   setAudioVol]   = useState(0.8);
  const [batchFiles, setBatchFiles] = useState<{name:string;url:string}[]>([]);
  const [batchIdx,   setBatchIdx]   = useState<number|null>(null);

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const reflRef     = useRef<HTMLCanvasElement>(null);
  const mouseXRef   = useRef(0.5);
  const zoomRef     = useRef(1.0);
  const panRef      = useRef({x:0,y:0});
  const panStartRef = useRef<{mx:number;my:number;px:number;py:number}|null>(null);
  const glRef       = useRef<WebGLRenderingContext|null>(null);
  const plainRef    = useRef<WebGLProgram|null>(null);
  const chromaRef   = useRef<WebGLProgram|null>(null);
  const postRef     = useRef<WebGLProgram|null>(null);
  const cutoutRef   = useRef<WebGLProgram|null>(null);
  const blendRef    = useRef<WebGLProgram|null>(null);
  const screenRecRef= useRef<WebGLProgram|null>(null);
  const screenFxRef = useRef<WebGLProgram|null>(null);
  const fboRef      = useRef<FBO|null>(null);
  const mTexRef     = useRef<WebGLTexture|null>(null);
  const rTexRef     = useRef<WebGLTexture|null>(null);
  const mockupVidRef= useRef<HTMLVideoElement>(null);
  const recVidRef   = useRef<HTMLVideoElement>(null);

  const mReadyRef   = useRef(false);
  const rReadyRef   = useRef(false);
  const mIsVRef     = useRef(false);
  const rStaticRef  = useRef(false);
  const pinsRef        = useRef<Quad>(pins);
  const cszRef         = useRef(csz);
  const activePinRef   = useRef<number|null>(null);
  const recNatRef   = useRef({w:1920,h:1080});
  const modeRef     = useRef<Mode>('manual');
  const keyClrRef   = useRef<[number,number,number]>([0,1,0]);
  const keyTRef     = useRef(0.44);
  const keySRef     = useRef(0.09);
  const keySpillRef = useRef(0.90);
  const screenBoundsRef  = useRef<{x0:number;y0:number;x1:number;y1:number}|null>(null);
  const screenCornersRef = useRef<Quad|null>(null);
  const enhRef      = useRef<Enhance>(GRADES.natural);
  const trimInRef   = useRef(0);
  const trimOutRef  = useRef(1);
  const rafRef      = useRef(0);
  const recorderRef = useRef<MediaRecorder|null>(null);
  const chunksRef   = useRef<Blob[]>([]);
  const timerRef    = useRef<ReturnType<typeof setInterval>|null>(null);
  const edgeBlendRef   = useRef(true);
  const borderWidthRef   = useRef(2);
  const borderRadiusRef  = useRef(0);
  const topBorderRef     = useRef(false);
  const mockupOpRef    = useRef(1.0);
  const lumaKeyRef   = useRef(0.0);
  const lumaSoftRef  = useRef(0.08);
  const chromaKeyRef    = useRef(0.0);
  const punchThroughRef   = useRef(false);
  const autoPlacementRef  = useRef<'fit'|'corners'|'fill'>('fit');
  const autoFitScaleWRef  = useRef(0.20);
  const autoFitScaleHRef  = useRef(0.38);
  const autoFitYRef       = useRef(0.50);
  const autoRecScaleRef   = useRef(1.0);
  const camHoleRef      = useRef(false);
  const camXRef         = useRef(0.50);
  const camYRef         = useRef(0.04);
  const camRadiusRef    = useRef(22);
  const textItemsRef    = useRef<TextItem[]>([]);
  const textDragRef   = useRef<{id:string;sx:number;sy:number;ox:number;oy:number}|null>(null);
  const textCanvasRef = useRef<HTMLCanvasElement|null>(null);
  const textTexRef    = useRef<WebGLTexture|null>(null);
  const audioElRef    = useRef<HTMLAudioElement|null>(null);
  const audioCtxRef   = useRef<AudioContext|null>(null);
  const audioVolRef   = useRef(0.8);
  const videoTrackRef  = useRef<{requestFrame():void}|null>(null);
  const outCanvasRef   = useRef<HTMLCanvasElement|null>(null);
  const exportRatioRef = useRef<ExportRatio>('16:9');
  const bgFillRef      = useRef<'blur'|'black'>('blur');
  const videoEncoderRef = useRef<any>(null);
  const muxerRef        = useRef<any>(null);
  const muxerTargetRef  = useRef<any>(null);
  const recStartTimeRef = useRef(0);
  const recFrameRef     = useRef(0);
  const useWebCodecsRef = useRef(false);
  const renderOneFrameRef = useRef<(()=>void)|null>(null);
  const isBatchingRef    = useRef(false);
  const batchOnDoneRef   = useRef<(()=>void)|null>(null);
  const batchFilenameRef = useRef<string|null>(null);
  const autoStopRef      = useRef<(()=>void)|null>(null);

  useEffect(()=>{pinsRef.current=pins},[pins]);
  useEffect(()=>{cszRef.current=csz},[csz]);
  useEffect(()=>{recNatRef.current=recNative},[recNative]);
  useEffect(()=>{
    modeRef.current=mode;
    // When switching to Auto with a static mockup already loaded, run detection now
    if(mode==='auto' && mockupSrc && !mockupIsV && screenCornersRef.current===null){
      const img=new Image(); img.crossOrigin='anonymous';
      img.onload=()=>{
        const kc=detectKeyColor(img);
        setKeyColor(kc);
        screenBoundsRef.current=detectScreenBounds(img,kc);
        screenCornersRef.current=detectScreenCorners(img,kc);
      };
      img.src=mockupSrc;
    }
  },[mode]);
  useEffect(()=>{keyClrRef.current=hexToRgb(keyColor)},[keyColor]);
  useEffect(()=>{keyTRef.current=keyThresh},[keyThresh]);
  useEffect(()=>{keySRef.current=keySoft},[keySoft]);
  useEffect(()=>{keySpillRef.current=keySpill},[keySpill]);
  useEffect(()=>{enhRef.current=enhance},[enhance]);
  useEffect(()=>{trimInRef.current=trimIn},[trimIn]);
  useEffect(()=>{trimOutRef.current=trimOut},[trimOut]);
  useEffect(()=>{zoomRef.current=zoom},[zoom]);
  useEffect(()=>{panRef.current=pan},[pan]);
  useEffect(()=>{edgeBlendRef.current=edgeBlend},[edgeBlend]);
  useEffect(()=>{borderWidthRef.current=borderWidth},[borderWidth]);
  useEffect(()=>{borderRadiusRef.current=borderRadius},[borderRadius]);
  useEffect(()=>{topBorderRef.current=topBorder},[topBorder]);
  useEffect(()=>{mockupOpRef.current=mockupOp},[mockupOp]);
  useEffect(()=>{lumaKeyRef.current=lumaKey},[lumaKey]);
  useEffect(()=>{lumaSoftRef.current=lumaSoft},[lumaSoft]);
  useEffect(()=>{chromaKeyRef.current=chromaKey},[chromaKey]);
  useEffect(()=>{punchThroughRef.current=punchThrough},[punchThrough]);
  useEffect(()=>{autoPlacementRef.current=autoPlacement},[autoPlacement]);
  useEffect(()=>{autoFitScaleWRef.current=autoFitScaleW},[autoFitScaleW]);
  useEffect(()=>{autoFitScaleHRef.current=autoFitScaleH},[autoFitScaleH]);
  useEffect(()=>{autoFitYRef.current=autoFitY},[autoFitY]);
  useEffect(()=>{autoRecScaleRef.current=autoRecScale},[autoRecScale]);
  useEffect(()=>{camHoleRef.current=camHole},[camHole]);
  useEffect(()=>{camXRef.current=camX},[camX]);
  useEffect(()=>{camYRef.current=camY},[camY]);
  useEffect(()=>{camRadiusRef.current=camRadius},[camRadius]);
  useEffect(()=>{textItemsRef.current=textItems},[textItems]);
  useEffect(()=>{audioVolRef.current=audioVol; if(audioElRef.current) audioElRef.current.volume=audioVol;},[audioVol]);
  useEffect(()=>{ exportRatioRef.current=exportRatio; },[exportRatio]);
  useEffect(()=>{ bgFillRef.current=bgFill; },[bgFill]);

  // ── Init WebGL ──────────────────────────────────────────────────────────────
  useEffect(()=>{
    if(glRef.current) return; // guard: React Strict Mode double-invoke
    const canvas=canvasRef.current!;
    // antialias:true — hardware MSAA eliminates mesh boundary jagging
    const gl=canvas.getContext('webgl',{preserveDrawingBuffer:true,alpha:false,antialias:true});
    if(!gl) return;
    glRef.current=gl;
    plainRef.current  =mkProgram(gl,FRAG_PLAIN,'plain');
    chromaRef.current =mkProgram(gl,FRAG_CHROMA,'chroma');
    postRef.current   =mkProgram(gl,FRAG_POST,'post');
    cutoutRef.current  =mkProgram(gl,FRAG_CUTOUT,'cutout');
    blendRef.current   =mkProgram(gl,FRAG_BLEND,'blend');
    screenRecRef.current=mkProgram(gl,FRAG_SCREEN_REC,'screenRec');
    screenFxRef.current =mkProgram(gl,FRAG_SCREEN_FX,'screenFx');
    mTexRef.current  =mkTex(gl); rTexRef.current=mkTex(gl); textTexRef.current=mkTex(gl);
    fboRef.current   =createFBO(gl,canvas.width,canvas.height);
    const tc=document.createElement('canvas'); tc.width=canvas.width; tc.height=canvas.height;
    textCanvasRef.current=tc;
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);

    let frameCount=0;
    const renderOnce=()=>{
      frameCount++;
      const gl=glRef.current!,plain=plainRef.current!,chroma=chromaRef.current!,
            post=postRef.current!,fbo=fboRef.current!,mt=mTexRef.current!,rt=rTexRef.current!;
      const mVid=mockupVidRef.current,rVid=recVidRef.current;
      const W=canvas.width,H=canvas.height;
      const {w:dW,h:dH}=cszRef.current,{w:rW,h:rH}=recNatRef.current;
      const e=enhRef.current,t=performance.now()*0.001;

      gl.bindFramebuffer(gl.FRAMEBUFFER,fbo.fb);
      gl.viewport(0,0,W,H); gl.clearColor(0.02,0.02,0.055,1); gl.clear(gl.COLOR_BUFFER_BIT);

      if(modeRef.current==='manual'){
        if(mReadyRef.current&&mIsVRef.current&&mVid&&mVid.readyState>=2) uploadTex(gl,mt,mVid);
        if(rReadyRef.current&&!rStaticRef.current&&rVid&&rVid.readyState>=2) uploadTex(gl,rt,rVid);
        const mop=mockupOpRef.current, ue=edgeBlendRef.current?1:0;
        const lk=lumaKeyRef.current, ls=lumaSoftRef.current, ck=chromaKeyRef.current;

        if(rReadyRef.current){
          const sx=dW>0?W/dW:1,sy=dH>0?H/dH:1;
          const np=pinsRef.current.map(p=>({x:p.x*sx,y:p.y*sy})) as Quad;
          const pinUV={
            uQ0:[np[0].x/W,1-np[0].y/H] as [number,number],
            uQ1:[np[1].x/W,1-np[1].y/H] as [number,number],
            uQ2:[np[2].x/W,1-np[2].y/H] as [number,number],
            uQ3:[np[3].x/W,1-np[3].y/H] as [number,number],
          };
          // Corner-rounding + camera-hole uniforms for FRAG_SCREEN_REC
          const qW=Math.sqrt((np[1].x-np[0].x)**2+(np[1].y-np[0].y)**2)||1;
          const qH=Math.sqrt((np[3].x-np[0].x)**2+(np[3].y-np[0].y)**2)||1;
          const rr=borderRadiusRef.current*(W/Math.max(1,cszRef.current.w||W));
          const uRx=rr/qW, uRy=rr/qH;
          const uCamAsp=qW/qH;
          const uCamR=camHoleRef.current?camRadiusRef.current*(W/Math.max(1,cszRef.current.w||W))/qW:0;
          const uCamPos:[number,number]=[camXRef.current,camYRef.current];
          const srecUni={uEdge:ue,uRx,uRy,uCamPos,uCamR,uCamAsp};

          if(punchThroughRef.current && cutoutRef.current){
            // Punch-through: recording below, mockup screen cut out on top → hand stays in front
            const vt=pinVerts(np,W,H);
            if(vt){
              const srec=screenRecRef.current||plain;
              drawQuad(gl,srec,rt,vt,srecUni);
              if(mReadyRef.current){
                drawQuad(gl,cutoutRef.current,mt,bgVerts(),{...pinUV,uOpacity:mop});
              }
              if(screenFxRef.current) drawQuad(gl,screenFxRef.current,rt,vt,{});
            } else if(mReadyRef.current){
              drawQuad(gl,plain,mt,bgVerts(),{uEdge:0,uOpacity:mop});
            }
          } else if((lk>0.001||ck>0.001) && mReadyRef.current && blendRef.current){
            // Luma/chroma blend: single-pass, always opaque — no bleed possible
            const crop=coverUVBounds(rW,rH,W,H);
            drawQuad(gl,blendRef.current,mt,bgVerts(),
              {...pinUV,uLuma:lk,uLumaSoft:ls,uChroma:ck,uChromaKey:keyClrRef.current,uRecCrop:crop},rt);
          } else {
            const vt=pinVerts(np,W,H);
            if(vt){
              // Mockup first (full), then recording on top
              if(mReadyRef.current){
                drawQuad(gl,plain,mt,bgVerts(),{uEdge:0,uOpacity:mop});
              }
              const srec=screenRecRef.current||plain;
              drawQuad(gl,srec,rt,vt,srecUni);
              if(screenFxRef.current) drawQuad(gl,screenFxRef.current,rt,vt,{});
            } else if(mReadyRef.current){
              drawQuad(gl,plain,mt,bgVerts(),{uEdge:0,uOpacity:mop});
            }
          }
        } else if(mReadyRef.current){
          drawQuad(gl,plain,mt,bgVerts(),{uEdge:0,uOpacity:mop});
        }
      } else {
        // AUTO mode: recording below, mockup WITH chroma key on top masks to screen only.
        const ac=screenCornersRef.current;
        const cUni={uKey:keyClrRef.current,uThresh:keyTRef.current,uSoft:keySRef.current,uSpill:keySpillRef.current};
        const placement=autoPlacementRef.current;
        const scl=autoRecScaleRef.current;

        // Upload mockup frame once (needed by all branches)
        if(mReadyRef.current&&mIsVRef.current&&mVid&&mVid.readyState>=2) uploadTex(gl,mt,mVid);
        // Upload recording frame once
        if(rReadyRef.current&&!rStaticRef.current&&rVid&&rVid.readyState>=2) uploadTex(gl,rt,rVid);

        if(placement==='fill'){
          // ── Fill-Canvas mode ──────────────────────────────────────────────
          // Recording fills the entire canvas — chroma key is the only mask.
          // Works for moving-phone videos: as the green screen moves each frame,
          // the chroma key follows it and reveals the recording in the right place.
          if(rReadyRef.current){
            drawQuad(gl,screenRecRef.current||plain,rt,coverVerts(rW,rH,W,H),
              {uEdge:0,uRx:0,uRy:0,uCamPos:[0.5,0.04] as [number,number],uCamR:0,uCamAsp:1});
          }
          if(mReadyRef.current) drawQuad(gl,chroma,mt,bgVerts(),cUni);
        } else if(placement==='fit'){
          // ── Scale-to-Fit mode (default) ───────────────────────────────────
          // Recording placed at user-set % of canvas height, centered on detected screen
          // (or canvas center if detection failed). Full recording always visible.
          if(rReadyRef.current){
            const sb=screenBoundsRef.current;
            const cx=sb?(sb.x0+sb.x1)/2:0.5;
            const cy=sb?(sb.y0+sb.y1)/2:0.5;
            const rvt=zoomVerts(cx,autoFitYRef.current,autoFitScaleWRef.current,autoFitScaleHRef.current);
            drawQuad(gl,screenRecRef.current||plain,rt,rvt,
              {uEdge:0,uRx:0,uRy:0,uCamPos:[0.5,0.04] as [number,number],uCamR:0,uCamAsp:1});
          }
          if(mReadyRef.current) drawQuad(gl,chroma,mt,bgVerts(),cUni);
        } else if(ac){
          // ── Fit-Corners mode (perspective warp with optional scale) ───────
          const cx=(ac[0].x+ac[1].x+ac[2].x+ac[3].x)/4;
          const cy=(ac[0].y+ac[1].y+ac[2].y+ac[3].y)/4;
          const sac=(scl===1?ac:ac.map(p=>({x:cx+(p.x-cx)*scl,y:cy+(p.y-cy)*scl}))) as Quad;
          const np=sac.map(p=>({x:p.x*W,y:p.y*H})) as Quad;
          const vt=pinVerts(np,W,H);
          if(rReadyRef.current&&vt){
            drawQuad(gl,screenRecRef.current||plain,rt,vt,
              {uEdge:0,uRx:0,uRy:0,uCamPos:[0.5,0.04] as [number,number],uCamR:0,uCamAsp:1});
          }
          if(mReadyRef.current) drawQuad(gl,chroma,mt,bgVerts(),cUni);
          if(rReadyRef.current&&screenFxRef.current&&vt) drawQuad(gl,screenFxRef.current,rt,vt,{});
        } else {
          // ── Fallback: no corners detected ─────────────────────────────────
          if(rReadyRef.current){
            const sb=screenBoundsRef.current;
            const rvt=sb?boundsVerts(sb,rW,rH,W,H):coverVerts(rW,rH,W,H);
            drawQuad(gl,plain,rt,rvt,{uEdge:0,uOpacity:1});
          }
          if(mReadyRef.current) drawQuad(gl,chroma,mt,bgVerts(),cUni);
        }
      }

      gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,W,H);
      drawQuad(gl,post,fbo.tex,bgVerts(),{
        uInvW:1/W,uInvH:1/H,uSharp:e.sharp,uBright:e.bright,uContrast:e.contrast,
        uSat:e.sat,uTemp:e.temp,uVig:e.vignette,uBloom:e.bloom,uGrain:e.grain,uTime:t,
      });

      // Text overlay + bezel border — composite on top of post-processed canvas
      const tc=textCanvasRef.current, ttex=textTexRef.current;
      const hasTextItems=textItemsRef.current.length>0;
      const hasRecForBorder=rReadyRef.current&&modeRef.current==='manual';
      if(tc && ttex && (hasTextItems||hasRecForBorder)){
        if(tc.width!==W||tc.height!==H){tc.width=W;tc.height=H;}
        const ctx2d=tc.getContext('2d');
        if(ctx2d){
          ctx2d.clearRect(0,0,W,H);
          const dw=cszRef.current.w||1;
          const scl=W/dw;
          ctx2d.textBaseline='top';
          if(hasTextItems) for(const item of textItemsRef.current){
            ctx2d.save();
            const sz=Math.round(item.size*scl);
            const base=item.family.replace(/,?\s*(sans-serif|serif|cursive|monospace)\s*$/,'');
            const mainFont=`${item.italic?'italic ':''}${item.bold?'bold ':''}${sz}px ${base},'Segoe UI Emoji','Apple Color Emoji','Noto Color Emoji',sans-serif`;
            const emojiFont=`${sz}px 'Segoe UI Emoji','Apple Color Emoji','Noto Color Emoji',sans-serif`;
            ctx2d.fillStyle=item.color;
            ctx2d.shadowColor='rgba(0,0,0,0.6)'; ctx2d.shadowBlur=Math.round(6*scl);
            const lines=item.text.split('\n');
            const lineH=sz*1.25;
            lines.forEach((txt,li)=>{
              const emojiRe=/\p{Extended_Pictographic}/gu;
              let last=0, cx=item.x*W;
              const cy=item.y*H+li*lineH;
              for(const m of txt.matchAll(emojiRe)){
                if(m.index!>last){
                  ctx2d.font=mainFont;
                  const seg=txt.slice(last,m.index);
                  ctx2d.fillText(seg,cx,cy);
                  cx+=ctx2d.measureText(seg).width;
                }
                ctx2d.font=emojiFont;
                ctx2d.fillText(m[0],cx,cy);
                cx+=ctx2d.measureText(m[0]).width;
                last=m.index!+m[0].length;
              }
              if(last<txt.length){ctx2d.font=mainFont;ctx2d.fillText(txt.slice(last),cx,cy);}
            });
            ctx2d.restore();
          }
          // Bezel border — rounded or sharp corners, top optional (white)
          if(hasRecForBorder){
            const {w:dW2,h:dH2}=cszRef.current;
            const sx2=dW2>0?W/dW2:1,sy2=dH2>0?H/dH2:1;
            const np2=pinsRef.current.map(p=>({x:p.x*sx2,y:p.y*sy2}));
            // np2[0]=TL, np2[1]=TR, np2[2]=BR, np2[3]=BL
            const TL=np2[0],TR=np2[1],BR=np2[2],BL=np2[3];
            const lw=Math.max(borderWidthRef.current,Math.round(borderWidthRef.current*scl));
            const r=borderRadiusRef.current*scl;
            // point at distance d from corner `c` toward neighbor `n`
            const pe=(c:{x:number;y:number},n:{x:number;y:number},d:number)=>{
              const dx=n.x-c.x,dy=n.y-c.y,len=Math.sqrt(dx*dx+dy*dy);
              if(len<0.001)return c;
              const t=Math.min(d,len*0.5)/len;
              return{x:c.x+dx*t,y:c.y+dy*t};
            };
            ctx2d.save();
            ctx2d.lineJoin='round';
            ctx2d.strokeStyle='rgba(0,0,0,0.92)';
            ctx2d.lineWidth=lw;
            ctx2d.beginPath();
            if(r>0){
              // Full perimeter with all 4 corners rounded (ideal for phones)
              const a0=pe(TL,TR,r);
              ctx2d.moveTo(a0.x,a0.y);
              ctx2d.arcTo(TL.x,TL.y,BL.x,BL.y,r);
              ctx2d.arcTo(BL.x,BL.y,BR.x,BR.y,r);
              ctx2d.arcTo(BR.x,BR.y,TR.x,TR.y,r);
              ctx2d.arcTo(TR.x,TR.y,TL.x,TL.y,r);
              ctx2d.closePath();
            } else {
              // Sharp — left, bottom, right only
              ctx2d.moveTo(TL.x,TL.y);
              ctx2d.lineTo(BL.x,BL.y);
              ctx2d.lineTo(BR.x,BR.y);
              ctx2d.lineTo(TR.x,TR.y);
            }
            ctx2d.stroke();
            // Top edge (white, optional — sharp corners mode only)
            if(topBorderRef.current&&r<=0){
              ctx2d.strokeStyle='rgba(255,255,255,0.90)';
              ctx2d.lineWidth=lw;
              ctx2d.beginPath();
              ctx2d.moveTo(TL.x,TL.y);
              ctx2d.lineTo(TR.x,TR.y);
              ctx2d.stroke();
            }
            ctx2d.restore();
          }

          gl.bindTexture(gl.TEXTURE_2D,ttex);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);
          gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,tc);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
          gl.bindFramebuffer(gl.FRAMEBUFFER,null);
          drawQuad(gl,plainRef.current!,ttex,bgVerts(),{uEdge:0,uOpacity:1});
        }
      }

      // Reflection — throttled to every 4 frames, skipped during recording
      const recording=recorderRef.current?.state==='recording';
      const refl = reflRef.current;
      if (refl && mReadyRef.current && !recording && frameCount%4===0) {
        gl.finish(); // block GPU only when needed for pixel readback
        const rfH = refl.height, rfW = Math.min(refl.width, W);
        const ctx = refl.getContext('2d');
        if (ctx) {
          // Read bottom rfH rows of the WebGL canvas (Y=0 is bottom in WebGL)
          const pixels = new Uint8Array(rfW * rfH * 4);
          gl.readPixels(0, 0, rfW, rfH, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          const imgData = ctx.createImageData(rfW, rfH);
          for (let row = 0; row < rfH; row++) {
            // Flip: row 0 of imgData ← row rfH-1 of pixels (bottom→top)
            const srcRow = rfH - 1 - row;
            const s = srcRow * rfW * 4, d = row * rfW * 4;
            imgData.data.set(pixels.subarray(s, s + rfW * 4), d);
            // Gradient alpha: strong at top (row 0), fades to transparent
            const fade = 1.0 - row / rfH;
            const alpha = Math.round(fade * fade * 230);
            for (let c = 0; c < rfW; c++) {
              const bi = d + c * 4;
              // 2× brightness boost + blue glass tint so dark pixels become visible
              imgData.data[bi]   = Math.min(255, imgData.data[bi]   * 2);
              imgData.data[bi+1] = Math.min(255, imgData.data[bi+1] * 2);
              imgData.data[bi+2] = Math.min(255, imgData.data[bi+2] * 2 + 35);
              imgData.data[bi+3] = alpha;
            }
          }
          ctx.clearRect(0, 0, rfW, rfH);
          // Subtle parallax: shift ImageData by mouse position
          const shift = Math.round((mouseXRef.current - 0.5) * rfW * 0.025);
          ctx.putImageData(imgData, shift, 0);
        }
      }

      // Composite to output canvas (ratio conversion for Reels / Square)
      const ratio=exportRatioRef.current;
      if(ratio!=='16:9'){
        if(!outCanvasRef.current) outCanvasRef.current=document.createElement('canvas');
        const oc=outCanvasRef.current;
        if(ratio==='9:16'&&(oc.width!==1080||oc.height!==1920)){oc.width=1080;oc.height=1920;}
        if(ratio==='1:1'&&(oc.width!==1080||oc.height!==1080)){oc.width=1080;oc.height=1080;}
        const octx=oc.getContext('2d');
        const glc=canvasRef.current;
        if(octx&&glc){
          const dw=oc.width,dh=oc.height,sa=W/H,da=dw/dh;
          // Background fill
          octx.save();
          if(bgFillRef.current==='blur'){
            octx.filter='blur(28px) brightness(0.28) saturate(1.6)';
            octx.drawImage(glc,-60,-60,dw+120,dh+120);
          } else {
            octx.fillStyle='#000';
            octx.fillRect(0,0,dw,dh);
          }
          octx.restore();
          // Main content centered, aspect-correct
          let mw:number,mh:number;
          if(sa>da){mw=dw;mh=Math.round(dw/sa);}else{mh=dh;mw=Math.round(dh*sa);}
          octx.drawImage(glc,Math.round((dw-mw)/2),Math.round((dh-mh)/2),mw,mh);
        }
      }

      // WebCodecs: feed VideoFrame to hardware H.264 encoder each rAF tick
      if(videoEncoderRef.current){
        let _vf:any=null;
        try{
          const VF=(window as any).VideoFrame;
          const srcCvs=(exportRatioRef.current!=='16:9'&&outCanvasRef.current)?outCanvasRef.current:canvas;
          const ts=Math.round((performance.now()-recStartTimeRef.current)*1000);
          _vf=new VF(srcCvs,{timestamp:ts});
          if(videoEncoderRef.current.state!=='closed'){
            videoEncoderRef.current.encode(_vf,{keyFrame:recFrameRef.current%120===0});
            recFrameRef.current++;
          }
        }catch{}finally{_vf?.close();}
      } else if(recorderRef.current?.state==='recording'){
        videoTrackRef.current?.requestFrame();
      }

    };
    renderOneFrameRef.current=renderOnce;
    function frame(){ renderOnce(); rafRef.current=requestAnimationFrame(frame); }
    rafRef.current=requestAnimationFrame(frame);
    return()=>{cancelAnimationFrame(rafRef.current);renderOneFrameRef.current=null;};
  },[]);

  useEffect(()=>{
    const c=canvasRef.current; if(!c) return;
    c.width=native.w; c.height=native.h;
    const gl=glRef.current,fbo=fboRef.current;
    if(gl&&fbo) resizeFBO(gl,fbo,native.w,native.h);
  },[native]);

  function loadMockupMedia(src:string,isVid:boolean){
    mReadyRef.current=false; mIsVRef.current=isVid;
    if(isVid){
      const vid=mockupVidRef.current; if(!vid) return;
      vid.src=src; vid.loop=true; vid.muted=true; vid.playsInline=true;
      vid.oncanplay=()=>{
        // Guard: applyMockupSize resets pins — only call once on initial load
        if(!mReadyRef.current) applyMockupSize(vid.videoWidth||1920,vid.videoHeight||1080);
        uploadTex(glRef.current!,mTexRef.current!,vid);
        mReadyRef.current=true; vid.play().catch(()=>{});
      };
      vid.load();
    } else {
      const img=new Image(); img.crossOrigin='anonymous';
      img.onload=()=>{
        const gl=glRef.current; if(!gl||!mTexRef.current) return;
        uploadTex(gl,mTexRef.current,img); mReadyRef.current=true;
        applyMockupSize(img.naturalWidth,img.naturalHeight);
        // Always detect — corners needed whenever user switches to auto mode
        const kc=detectKeyColor(img);
        if(modeRef.current==='auto') setKeyColor(kc);
        screenBoundsRef.current=detectScreenBounds(img,kc);
        screenCornersRef.current=detectScreenCorners(img,kc);
      };
      img.src=src;
    }
  }

  useEffect(()=>{if(mockupSrc) loadMockupMedia(mockupSrc,mockupIsV);},[mockupSrc,mockupIsV]);

  function applyMockupSize(nW:number,nH:number){
    const avW=window.innerWidth-280,avH=window.innerHeight-80;
    // Divide avH by 1.28 so canvas+reflection (22% extra) fits without overflow:hidden clipping
    const scale=Math.min(avW/nW,(avH/1.28)/nH,1);
    const dW=Math.round(nW*scale),dH=Math.round(nH*scale);
    setNative({w:nW,h:nH}); setCsz({w:dW,h:dH}); setPins(defaultCorners(dW,dH));
  }

  useEffect(()=>{
    const vid=recVidRef.current; if(!vid||!recSrc) return;
    rReadyRef.current=false; rStaticRef.current=false;
    vid.src=recSrc; vid.loop=false; vid.muted=true; vid.playsInline=true;
    vid.onloadedmetadata=()=>{setRecNative({w:vid.videoWidth||1920,h:vid.videoHeight||1080});setRecDur(vid.duration||0);};
    vid.oncanplay=()=>{rReadyRef.current=true;vid.play().catch(()=>{});};
    vid.ontimeupdate=()=>{
      const dur=vid.duration; if(!dur) return;
      const out=trimOutRef.current,inP=trimInRef.current;
      if(vid.currentTime>=dur*out) vid.currentTime=dur*inP;
      if(vid.currentTime<dur*inP)  vid.currentTime=dur*inP;
    };
    vid.load();
  },[recSrc]);

  const loadMockup=useCallback((file:File)=>{
    const isV=file.type.startsWith('video/')||/\.(mp4|webm|mov|mkv)$/i.test(file.name);
    setMockupIsV(isV); setMockupSrc(URL.createObjectURL(file)); setMockupFile(file.name);
  },[]);

  const loadRec=useCallback((file:File)=>{
    setRecFile(file.name);
    if(file.type.startsWith('image/')){
      rStaticRef.current=true; rReadyRef.current=false;
      const img=new Image(); img.onload=()=>{
        if(glRef.current&&rTexRef.current){uploadTex(glRef.current,rTexRef.current,img);rReadyRef.current=true;}
        setRecNative({w:img.naturalWidth,h:img.naturalHeight});
      };
      img.src=URL.createObjectURL(file);
    } else { setRecSrc(URL.createObjectURL(file)); }
  },[]);

  const loadMockupFromUrl=useCallback((url:string,name:string)=>{
    setMockupIsV(false); setMockupFile(name); setMockupSrc(url); setTab('edit');
  },[]);

  const addText=useCallback(()=>{
    const id=Date.now().toString();
    setTextItems(p=>[...p,{id,text:'Your text',x:0.5,y:0.15,size:48,color:'#ffffff',family:"'Bebas Neue', sans-serif",bold:false,italic:false}]);
    setSelTextId(id);
  },[]);
  const updateText=useCallback((id:string,key:keyof TextItem,val:unknown)=>{
    setTextItems(p=>p.map(it=>it.id===id?{...it,[key]:val}:it));
  },[]);
  const removeText=useCallback((id:string)=>{
    setTextItems(p=>p.filter(it=>it.id!==id));
    setSelTextId(null);
  },[]);

  const onPinDown=useCallback((i:number)=>(e:React.PointerEvent)=>{
    e.preventDefault();e.stopPropagation(); // stop bubbling so canvas-wrap doesn't start pan
    activePinRef.current=i; setActivePin(i);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  },[]);
  const onCanvasDown=useCallback((e:React.PointerEvent)=>{
    // No pin active → start pan drag
    panStartRef.current={mx:e.clientX,my:e.clientY,px:panRef.current.x,py:panRef.current.y};
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  },[]);
  const onMove=useCallback((e:React.PointerEvent)=>{
    const ap=activePinRef.current;
    const td=textDragRef.current;
    if(ap!==null){
      const rect=canvasRef.current!.getBoundingClientRect();
      const z=zoomRef.current,{w,h}=cszRef.current;
      setPins(prev=>{const n=[...prev] as Quad;n[ap]={x:Math.max(0,Math.min(w,(e.clientX-rect.left)/z)),y:Math.max(0,Math.min(h,(e.clientY-rect.top)/z))};return n;});
    } else if(td){
      const rect=canvasRef.current!.getBoundingClientRect();
      const z=zoomRef.current,{w,h}=cszRef.current;
      const cx=(e.clientX-rect.left)/z, cy=(e.clientY-rect.top)/z;
      setTextItems(prev=>prev.map(it=>it.id===td.id
        ?{...it,x:Math.max(0,Math.min(1,td.ox+(cx-td.sx)/w)),y:Math.max(0,Math.min(1,td.oy+(cy-td.sy)/h))}:it));
    } else if(panStartRef.current){
      const {mx,my,px,py}=panStartRef.current;
      setPan({x:px+e.clientX-mx,y:py+e.clientY-my});
    }
  },[]);
  const onUp=useCallback(()=>{activePinRef.current=null;setActivePin(null);panStartRef.current=null;textDragRef.current=null;},[]);
  const onAreaWheel=useCallback((e:React.WheelEvent)=>{
    e.preventDefault();
    const factor=e.deltaY>0?0.88:1.14;
    setZoom(z=>Math.max(0.2,Math.min(4,z*factor)));
  },[]);
  const fitZoom=useCallback(()=>{setZoom(1);setPan({x:0,y:0});},[]);

  const applyGrade=useCallback((g:GradeName)=>{setGrade(g);setEnhance(GRADES[g]);},[]);

  const savePreset=useCallback(()=>{
    if(!presetName.trim()) return;
    const p:Preset={name:presetName.trim(),mode,grade,enhance,keyColor,keyThresh,keySoft,pins:[...pins]};
    const updated=[...presets,p]; setPresets(updated);
    localStorage.setItem('mockup-presets',JSON.stringify(updated));
    setPresetName(''); showToast(`"${p.name}" saved`);
  },[presetName,mode,grade,enhance,keyColor,keyThresh,keySoft,presets]);

  const loadPreset=useCallback((p:Preset)=>{
    setMode(p.mode);setGrade(p.grade);setEnhance(p.enhance);
    setKeyColor(p.keyColor);setKeyThresh(p.keyThresh);setKeySoft(p.keySoft);
    if(p.pins) setPins(p.pins);
    showToast(`Loaded "${p.name}"${p.pins?' + pins':''}`);
  },[]);

  const deletePreset=useCallback((i:number)=>{
    const updated=presets.filter((_,idx)=>idx!==i);
    setPresets(updated); localStorage.setItem('mockup-presets',JSON.stringify(updated));
  },[presets]);

  const doExportPNG=useCallback(()=>{
    canvasRef.current?.toBlob(b=>{
      if(b){dl(b,'mockup.png');showToast('PNG saved!');setShowExp(false);}
      else showToast('Export failed.',true);
    },'image/png');
  },[]);

  const startRec=useCallback(()=>{
    const c=canvasRef.current; if(!c) return;
    const rVid=recVidRef.current;
    if(rVid&&rVid.duration) rVid.currentTime=rVid.duration*trimInRef.current;
    const ratio=exportRatioRef.current;
    let recCvs:HTMLCanvasElement=c, rW=c.width, rH=c.height;
    if(ratio!=='16:9'){
      if(!outCanvasRef.current) outCanvasRef.current=document.createElement('canvas');
      rW=1080; rH=ratio==='9:16'?1920:1080;
      outCanvasRef.current.width=rW; outCanvasRef.current.height=rH;
      recCvs=outCanvasRef.current;
    }

    // MediaRecorder fallback (Firefox / Safari / older Chrome)
    const doMR=()=>{
      const mime=MediaRecorder.isTypeSupported('video/mp4;codecs=avc1')?'video/mp4;codecs=avc1'
               :MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm';
      const ext=mime.startsWith('video/mp4')?'mp4':'webm';
      const cs=(recCvs as any).captureStream(60) as MediaStream;
      let rs=cs;
      const ae=audioElRef.current;
      if(ae&&ae.src){try{
        const a=new AudioContext();audioCtxRef.current=a;
        const s=a.createMediaElementSource(ae),d=a.createMediaStreamDestination();
        s.connect(d);s.connect(a.destination);
        ae.volume=audioVolRef.current;ae.currentTime=0;ae.loop=true;ae.play();
        rs=new MediaStream([...cs.getVideoTracks(),...d.stream.getAudioTracks()]);}catch{}}
      const rec=new MediaRecorder(rs,{mimeType:mime,videoBitsPerSecond:quality==='ultra'?80_000_000:40_000_000});
      chunksRef.current=[];
      rec.ondataavailable=e=>{if(e.data.size>0)chunksRef.current.push(e.data);};
      rec.onstop=()=>{dl(new Blob(chunksRef.current,{type:mime}),`mockup.${ext}`);
        setIsRec(false);setRecTime(0);if(timerRef.current)clearInterval(timerRef.current);showToast('Saved!');};
      recorderRef.current=rec; rec.start(500); setIsRec(true); setRecTime(0); setShowExp(false);
      timerRef.current=setInterval(()=>setRecTime(t=>t+1),1000);
      // Auto-stop when recording reaches trim out
      const rv=recVidRef.current;
      if(rv&&rv.duration){
        const onAutoStop=()=>{
          if(rv.currentTime>=rv.duration*trimOutRef.current-0.12){
            rv.removeEventListener('timeupdate',onAutoStop);autoStopRef.current=null;
            rec.stop();recorderRef.current=null;
            if(timerRef.current)clearInterval(timerRef.current);
          }
        };
        autoStopRef.current=()=>rv.removeEventListener('timeupdate',onAutoStop);
        rv.addEventListener('timeupdate',onAutoStop);
      }
    };

    // WebCodecs — hardware H.264 encoder, genuine premium quality (Chrome 94+)
    const wcOK=typeof (window as any).VideoEncoder!=='undefined'&&typeof (window as any).VideoFrame!=='undefined';
    useWebCodecsRef.current=wcOK;
    if(!wcOK){ doMR(); return; }

    // Cap resolution — hardware encoders typically max at 1920×1080 or 2560×1440
    const maxDim=1920;
    const scale=Math.min(1,maxDim/rW,maxDim/rH);
    const encW=Math.floor(rW*scale/2)*2, encH=Math.floor(rH*scale/2)*2;

    import('mp4-muxer').then(async ({Muxer,ArrayBufferTarget}:any)=>{
      const VE=(window as any).VideoEncoder;
      const br=quality==='ultra'?15_000_000:10_000_000;
      const codecList=['avc1.4D4028','avc1.42E028','avc1.42E01E'];
      let codec='';
      for(const c of codecList){
        try{const r=await VE.isConfigSupported({codec:c,width:encW,height:encH,bitrate:br,framerate:60});if(r.supported){codec=c;break;}}catch{}
      }
      if(!codec){showToast('No H.264 encoder — switching to MediaRecorder',true);useWebCodecsRef.current=false;doMR();return;}
      const tgt=new ArrayBufferTarget();
      const mux=new Muxer({target:tgt,video:{codec:'avc',width:encW,height:encH},fastStart:'in-memory'});
      const enc=new VE({
        output:(ch:any,mt:any)=>mux.addVideoChunk(ch,mt),
        error:(e:Error)=>{
          console.error('VideoEncoder:',e);videoEncoderRef.current=null;
          setIsRec(false);setRecTime(0);if(timerRef.current)clearInterval(timerRef.current);
          showToast('Encoder error — switching to MediaRecorder',true);
          useWebCodecsRef.current=false;doMR();
        },
      });
      enc.configure({
        codec,
        width:encW, height:encH,
        bitrate:br,
        framerate:60,
        hardwareAcceleration:'prefer-hardware',
        latencyMode:'realtime',
      });
      videoEncoderRef.current=enc; muxerRef.current=mux; muxerTargetRef.current=tgt;
      recStartTimeRef.current=performance.now(); recFrameRef.current=0;
      setIsRec(true); setRecTime(0); setShowExp(false);
      timerRef.current=setInterval(()=>setRecTime(t=>t+1),1000);
      // Auto-stop at trim out
      const rv=recVidRef.current;
      if(rv&&rv.duration){
        const onAutoStop=()=>{
          if(rv.currentTime>=rv.duration*trimOutRef.current-0.12){
            rv.removeEventListener('timeupdate',onAutoStop);autoStopRef.current=null;
            const e2=videoEncoderRef.current,m2=muxerRef.current,t2=muxerTargetRef.current;
            videoEncoderRef.current=null;
            setIsRec(false);setRecTime(0);if(timerRef.current)clearInterval(timerRef.current);
            showToast('Encoding…');
            if(e2&&e2.state!=='closed'){
              e2.flush().then(()=>{
                e2.close();m2?.finalize();
                if(t2?.buffer&&t2.buffer.byteLength>0){dl(new Blob([t2.buffer],{type:'video/mp4'}),'mockup.mp4');showToast('✓ Saved!');}
              }).catch((err:Error)=>showToast('Export failed: '+(err?.message||'unknown'),true));
            } else showToast('Encoder not ready',true);
          }
        };
        autoStopRef.current=()=>rv.removeEventListener('timeupdate',onAutoStop);
        rv.addEventListener('timeupdate',onAutoStop);
      }
      const ae=audioElRef.current;
      if(ae&&ae.src){try{const a=new AudioContext();audioCtxRef.current=a;
        const s=a.createMediaElementSource(ae);s.connect(a.destination);
        ae.volume=audioVolRef.current;ae.currentTime=0;ae.loop=true;ae.play();}catch{}}
    }).catch(()=>{ useWebCodecsRef.current=false; doMR(); });
  },[quality]);

  const stopRec=useCallback(()=>{
    autoStopRef.current?.(); autoStopRef.current=null;
    if(useWebCodecsRef.current&&videoEncoderRef.current){
      const enc=videoEncoderRef.current, mux=muxerRef.current, tgt=muxerTargetRef.current;
      videoEncoderRef.current=null;
      setIsRec(false); setRecTime(0); if(timerRef.current)clearInterval(timerRef.current);
      showToast('Encoding…');
      if(enc.state==='closed'){showToast('Encoder closed — nothing to save',true);return;}
      enc.flush().then(()=>{
        enc.close(); mux?.finalize();
        if(tgt?.buffer&&tgt.buffer.byteLength>0){const blob=new Blob([tgt.buffer],{type:'video/mp4'});dl(blob,'mockup.mp4');showToast('✓ Saved!');}
        else showToast('Export empty — record longer before stopping',true);
      }).catch((e:Error)=>{ console.error(e); showToast('Export failed: '+(e?.message||'unknown'),true); });
    } else {
      recorderRef.current?.stop(); if(timerRef.current)clearInterval(timerRef.current);
    }
    const ae=audioElRef.current; if(ae){ae.pause();ae.currentTime=0;}
    audioCtxRef.current?.close(); audioCtxRef.current=null;
  },[]);

  const startOfflineExport=useCallback(async()=>{
    const vid=recVidRef.current;
    if(!vid||!vid.src||!vid.duration){showToast('Load a screen recording first',true);return;}

    const inT=trimInRef.current*vid.duration;
    const outT=trimOutRef.current*vid.duration;
    const clipDur=Math.max(0.1,outT-inT);
    const glCvs=canvasRef.current!;
    const ratio=exportRatioRef.current;

    // ── Output dimensions ──────────────────────────────────────────────────
    let outW:number,outH:number;
    if(ratio==='9:16'){outW=1080;outH=1920;}
    else if(ratio==='1:1'){outW=1080;outH=1080;}
    else{const s=Math.min(1,1920/glCvs.width,1080/glCvs.height);outW=Math.floor(glCvs.width*s/2)*2;outH=Math.floor(glCvs.height*s/2)*2;}
    if(ratio!=='16:9'){
      if(!outCanvasRef.current)outCanvasRef.current=document.createElement('canvas');
      outCanvasRef.current.width=outW;outCanvasRef.current.height=outH;
    }

    setIsExporting(true);setExportProgress(0);setShowExp(false);

    // Seek to trim start before playback
    await new Promise<void>(res=>{
      const fn=()=>{vid.removeEventListener('seeked',fn);res();};
      vid.addEventListener('seeked',fn);vid.currentTime=inT;
    });
    // Also restart background video from its current position
    const bgVid=mockupVidRef.current;
    if(mIsVRef.current&&bgVid&&bgVid.duration) bgVid.play().catch(()=>{});

    const VE=(window as any).VideoEncoder;
    const VF=(window as any).VideoFrame;

    // ── MediaRecorder fallback (Safari / Firefox) ──────────────────────────
    if(!VE||!VF){
      try{
        const capCvs=(ratio!=='16:9'&&outCanvasRef.current)?outCanvasRef.current:glCvs;
        const mime=MediaRecorder.isTypeSupported('video/mp4;codecs=avc1')?'video/mp4;codecs=avc1'
          :MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm';
        const ext=mime.startsWith('video/mp4')?'mp4':'webm';
        const cs=(capCvs as any).captureStream(60) as MediaStream;
        const rec=new MediaRecorder(cs,{mimeType:mime,videoBitsPerSecond:quality==='ultra'?80_000_000:40_000_000});
        chunksRef.current=[];
        rec.ondataavailable=e=>{if(e.data.size>0)chunksRef.current.push(e.data);};
        await new Promise<void>((resolve,reject)=>{
          rec.onstop=()=>{
            const fname=batchFilenameRef.current||`mockup.${ext}`;batchFilenameRef.current=null;
            dl(new Blob(chunksRef.current,{type:mime}),fname);showToast('✓ Saved!');resolve();
          };
          const onTU=()=>{
            setExportProgress(Math.min(0.95,(vid.currentTime-inT)/clipDur));
            if(vid.currentTime>=outT-0.1||vid.ended){
              vid.removeEventListener('timeupdate',onTU);rec.stop();recorderRef.current=null;
            }
          };
          vid.addEventListener('timeupdate',onTU);
          rec.start(200);recorderRef.current=rec;
          vid.play().catch(reject);
        });
      }catch(e){showToast('Export failed: '+(e as Error).message,true);}
      finally{setIsExporting(false);setExportProgress(0);vid.pause();
        const done=batchOnDoneRef.current;batchOnDoneRef.current=null;done?.();}
      return;
    }

    // ── WebCodecs real-time path — let video play, rAF loop feeds encoder ──
    try{
      const {Muxer,ArrayBufferTarget}=await import('mp4-muxer') as any;
      const tgt=new ArrayBufferTarget();
      const mux=new Muxer({target:tgt,video:{codec:'avc',width:outW,height:outH},fastStart:'in-memory',firstTimestampBehavior:'offset'});
      const br=quality==='ultra'?15_000_000:10_000_000;

      const codecList=['avc1.4D4028','avc1.42E028','avc1.42E01E'];
      let codec='';
      for(const c of codecList){
        try{const r=await VE.isConfigSupported({codec:c,width:outW,height:outH,bitrate:br,framerate:60});
          if(r.supported){codec=c;break;}}catch{}
      }
      if(!codec)throw new Error('No H.264 encoder — update Chrome');

      const enc=new VE({
        output:(ch:any,mt:any)=>mux.addVideoChunk(ch,mt),
        error:(e:Error)=>console.error('VE:',e),
      });
      enc.configure({codec,width:outW,height:outH,bitrate:br,framerate:60,
        hardwareAcceleration:'prefer-hardware',latencyMode:'quality'});
      await new Promise(r=>setTimeout(r,80));
      if(enc.state==='closed'){
        // Hardware encoder refused — fall back to MediaRecorder silently
        try{enc.close();}catch{}
        const capCvs=(ratio!=='16:9'&&outCanvasRef.current)?outCanvasRef.current:glCvs;
        const mime=MediaRecorder.isTypeSupported('video/mp4;codecs=avc1')?'video/mp4;codecs=avc1'
          :MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm';
        const ext=mime.startsWith('video/mp4')?'mp4':'webm';
        const cs=(capCvs as any).captureStream(60) as MediaStream;
        const mrec=new MediaRecorder(cs,{mimeType:mime,videoBitsPerSecond:quality==='ultra'?80_000_000:40_000_000});
        chunksRef.current=[];
        mrec.ondataavailable=(ev:any)=>{if(ev.data.size>0)chunksRef.current.push(ev.data);};
        await new Promise<void>((resolve,reject)=>{
          mrec.onstop=()=>{
            const fname=batchFilenameRef.current||`mockup.${ext}`;batchFilenameRef.current=null;
            dl(new Blob(chunksRef.current,{type:mime}),fname);showToast('✓ Saved!');resolve();
          };
          const onTU=()=>{
            setExportProgress(Math.min(0.95,(vid.currentTime-inT)/clipDur));
            if(vid.currentTime>=outT-0.1||vid.ended){vid.removeEventListener('timeupdate',onTU);mrec.stop();recorderRef.current=null;}
          };
          vid.addEventListener('timeupdate',onTU);
          mrec.start(200);recorderRef.current=mrec;
          vid.play().catch(reject);
        });
        return;
      }

      // Wire encoder into rAF loop — renderOnce feeds VideoFrames automatically
      recStartTimeRef.current=performance.now();
      recFrameRef.current=0;
      videoEncoderRef.current=enc;

      // Play video — rAF loop encodes each rendered frame at natural speed
      await new Promise<void>((resolve,reject)=>{
        const onTU=()=>{
          setExportProgress(Math.min(0.95,(vid.currentTime-inT)/clipDur));
          if(vid.currentTime>=outT-0.05||vid.ended){
            vid.removeEventListener('timeupdate',onTU);
            vid.removeEventListener('error',onErr);
            vid.pause();resolve();
          }
        };
        const onErr=()=>{vid.removeEventListener('timeupdate',onTU);vid.removeEventListener('error',onErr);reject(new Error('Playback error'));};
        vid.addEventListener('timeupdate',onTU);
        vid.addEventListener('error',onErr);
        vid.play().catch(reject);
      });

      videoEncoderRef.current=null;
      setExportProgress(0.99);
      await enc.flush();enc.close();mux.finalize();
      const fname=batchFilenameRef.current||'mockup.mp4';batchFilenameRef.current=null;
      dl(new Blob([tgt.buffer],{type:'video/mp4'}),fname);
      showToast('✓ Export complete!');
    }catch(e){
      videoEncoderRef.current=null;
      console.error('Export error:',e);
      showToast('Export failed: '+(e as Error).message,true);
    }finally{
      setIsExporting(false);setExportProgress(0);
      videoEncoderRef.current=null;
      vid.pause();
      const done=batchOnDoneRef.current;batchOnDoneRef.current=null;done?.();
    }
  },[quality]);

  // ── Template system ───────────────────────────────────────────────────────────
  const saveCurrentTemplate=useCallback(async(name:string)=>{
    if(!mockupSrc){showToast('Load a mockup first',true);return;}
    const toBase64=(src:string,maxW:number,q:number)=>new Promise<string>((res,rej)=>{
      const img=new Image(); img.crossOrigin='anonymous';
      img.onload=()=>{
        const s=Math.min(1,maxW/img.naturalWidth);
        const cvs=document.createElement('canvas');
        cvs.width=Math.round(img.naturalWidth*s); cvs.height=Math.round(img.naturalHeight*s);
        cvs.getContext('2d')!.drawImage(img,0,0,cvs.width,cvs.height);
        res(cvs.toDataURL('image/jpeg',q));
      };
      img.onerror=rej; img.src=src;
    });
    try{
      const [mockupData,thumb]=await Promise.all([toBase64(mockupSrc,1280,0.82),toBase64(mockupSrc,200,0.70)]);
      const t:TemplateItem={id:Date.now().toString(),name,thumb,mockupData,pins:[...pins] as Quad,
        grade,enhance,ratio:exportRatio as ExportRatio,borderWidth,borderRadius,topBorder,edgeBlend,mockupOp};
      const updated=[...templates,t];
      setTemplates(updated);
      try{localStorage.setItem('mockup_templates',JSON.stringify(updated));}catch{showToast('Storage full — delete old templates',true);return;}
      showToast(`✓ Template "${name}" saved`);
    }catch{showToast('Failed to save template',true);}
  },[mockupSrc,pins,grade,enhance,exportRatio,borderWidth,borderRadius,topBorder,edgeBlend,mockupOp,templates]);

  const deleteTemplate=useCallback((id:string)=>{
    const updated=templates.filter(t=>t.id!==id);
    setTemplates(updated); setSelTpls(s=>{const n=new Set(s);n.delete(id);return n;});
    try{localStorage.setItem('mockup_templates',JSON.stringify(updated));}catch{}
  },[templates]);

  const runTemplateBatch=useCallback(async()=>{
    if(isBatchingRef.current||selTpls.size===0){showToast('Select templates first',true);return;}
    if(!rReadyRef.current){showToast('Load a recording first',true);return;}
    isBatchingRef.current=true;
    const selected=templates.filter(t=>selTpls.has(t.id));
    for(let i=0;i<selected.length;i++){
      const t=selected[i]; setTplBatchIdx(i);
      // Load template mockup
      mReadyRef.current=false;
      loadMockupMedia(t.mockupData,false);
      // Wait up to 5s for mockup
      await new Promise<void>(resolve=>{
        const chk=setInterval(()=>{if(mReadyRef.current){clearInterval(chk);resolve();}},50);
        setTimeout(()=>{clearInterval(chk);resolve();},5000);
      });
      // Apply template settings (direct ref + state)
      setPins(t.pins as Quad); pinsRef.current=t.pins as Quad;
      setGrade(t.grade); setEnhance(t.enhance); enhRef.current=t.enhance;
      setExportRatio(t.ratio); exportRatioRef.current=t.ratio;
      setBorderWidth(t.borderWidth); borderWidthRef.current=t.borderWidth;
      setBorderRadius(t.borderRadius||0); borderRadiusRef.current=t.borderRadius||0;
      setTopBorder(t.topBorder); topBorderRef.current=t.topBorder;
      setEdgeBlend(t.edgeBlend); edgeBlendRef.current=t.edgeBlend;
      setMockupOp(t.mockupOp); mockupOpRef.current=t.mockupOp;
      await new Promise(r=>setTimeout(r,250));
      batchFilenameRef.current=`${t.name}_mockup.mp4`;
      await new Promise<void>(resolve=>{batchOnDoneRef.current=resolve;startOfflineExport();});
      await new Promise(r=>setTimeout(r,400));
    }
    isBatchingRef.current=false; setTplBatchIdx(null);
    showToast(`✓ Template batch done — ${selected.length} exports`);
  },[selTpls,templates,startOfflineExport]);

  const runAllFormats=useCallback(async()=>{
    if(isBatchingRef.current){showToast('Already exporting',true);return;}
    if(!recSrc||!rReadyRef.current){showToast('Load a recording first',true);return;}
    isBatchingRef.current=true;
    const origRatio=exportRatioRef.current;
    const fmts:[ExportRatio,string][]=[['9:16','reels'],['1:1','square'],['16:9','landscape']];
    for(const [r,label] of fmts){
      exportRatioRef.current=r; setExportRatio(r);
      await new Promise(x=>setTimeout(x,120));
      batchFilenameRef.current=`mockup_${label}.mp4`;
      await new Promise<void>(resolve=>{batchOnDoneRef.current=resolve;startOfflineExport();});
      await new Promise(x=>setTimeout(x,350));
    }
    exportRatioRef.current=origRatio; setExportRatio(origRatio);
    isBatchingRef.current=false;
    showToast('✓ 3 formats exported — reels · square · landscape');
  },[recSrc,startOfflineExport]);

  const runBatch=useCallback(async()=>{
    if(isBatchingRef.current||batchFiles.length===0) return;
    const VE=(window as any).VideoEncoder;
    if(!VE){showToast('Batch export requires Chrome 94+',true);return;}
    isBatchingRef.current=true;
    const files=[...batchFiles];
    for(let i=0;i<files.length;i++){
      setBatchIdx(i);
      const {url,name}=files[i];
      await new Promise<void>(resolve=>{
        const vid=recVidRef.current; if(!vid){resolve();return;}
        rReadyRef.current=false; rStaticRef.current=false;
        setTrimIn(0); setTrimOut(1);
        trimInRef.current=0; trimOutRef.current=1;
        vid.src=url; vid.loop=false; vid.muted=true; vid.playsInline=true;
        vid.onloadedmetadata=()=>{setRecNative({w:vid.videoWidth||1920,h:vid.videoHeight||1080});setRecDur(vid.duration||0);};
        vid.oncanplay=()=>{rReadyRef.current=true;vid.play().catch(()=>{});resolve();};
        vid.onerror=()=>resolve();
        vid.load();
      });
      if(!rReadyRef.current){showToast(`Skipped: ${name}`,true);continue;}
      await new Promise(r=>setTimeout(r,350));
      batchFilenameRef.current=name.replace(/\.[^.]+$/,'')+'_mockup.mp4';
      await new Promise<void>(resolve=>{
        batchOnDoneRef.current=resolve;
        startOfflineExport();
      });
      await new Promise(r=>setTimeout(r,400));
    }
    setBatchIdx(null); isBatchingRef.current=false; setBatchFiles([]);
    showToast(`✓ Batch done — ${files.length} exports saved`);
  },[batchFiles,startOfflineExport]);

  function showToast(msg:string,err=false){setToast({msg,err});setTimeout(()=>setToast(null),3000);}
  const fmt=(s:number)=>`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  const trimInPct=trimIn*100,trimOutPct=trimOut*100;

  return(
    <>
      <style>{CSS}</style>
      <video ref={mockupVidRef} style={{display:'none'}}/>
      <video ref={recVidRef}    style={{display:'none'}}/>

      <div className="app">
        <header className="hdr">
          <div className="logo"><div className="logo-gem"/>Mockup Studio</div>
          <div className="hdr-r">
            {isRec&&<div className="rec-badge"><span className="rec-dot"/>REC {fmt(recTime)}</div>}
            {isRec
              ?<button className="btn btn-danger" onClick={stopRec}>■ Stop &amp; Save</button>
              :<button className="btn btn-export" onClick={()=>setShowExp(true)} disabled={!mockupSrc}>↑ Export</button>}
          </div>
        </header>

        <div className="body">
          <aside className="sidebar">
            <div className="top-tabs">
              <button className={`top-tab${tab==='edit'?' active':''}`}    onClick={()=>setTab('edit')}>✦ Edit</button>
              <button className={`top-tab${tab==='library'?' active':''}`} onClick={()=>setTab('library')}>⊞ Library</button>
            </div>

            {tab==='library'&&<Library onSelectPhoto={loadMockupFromUrl}/>}

            {tab==='edit'&&<>
              <div className="mode-row">
                <button className={`mode-btn${mode==='manual'?' active':''}`} onClick={()=>setMode('manual')}>✦ Manual</button>
                <button className={`mode-btn${mode==='auto'?' active':''}`}   onClick={()=>setMode('auto')}>⚡ Auto</button>
              </div>

              <div className="sec">
                <div className="sec-title">{mode==='auto'?'Mockup (solid-colour screen)':'Mockup / Background'}</div>
                <DropZone label="Upload mockup" hint="PNG · JPG · MP4 · WebM"
                  file={mockupFile} onFile={loadMockup} accept="image/*,video/*" icon="🖼️"/>
                {mode==='auto'&&!mockupSrc&&(
                  <div style={{marginTop:8,padding:'8px 10px',background:'rgba(91,156,246,.07)',borderRadius:7,
                    border:'1px solid rgba(91,156,246,.15)',fontSize:10,color:'#5B9CF6',lineHeight:1.8}}>
                    Auto mode keys out the screen colour. Best with mockup templates that have a flat green, white, or any solid-colour screen.
                  </div>
                )}
              </div>

              <div className="sec">
                <div className="sec-title">Screen Recording</div>
                <DropZone label="Upload recording" hint="MP4 · WebM · MOV · PNG"
                  file={recFile} onFile={loadRec} accept="video/*,image/*" icon="🎬"/>
                {recDur>0&&<>
                  <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'var(--muted)',marginTop:8,marginBottom:2}}>
                    <span>Trim</span>
                    <span>{fmt(Math.round(trimIn*recDur))} – {fmt(Math.round(trimOut*recDur))}</span>
                  </div>
                  <div className="trim-wrap">
                    <div className="trim-track"/>
                    <div className="trim-fill" style={{left:`${trimInPct}%`,right:`${100-trimOutPct}%`}}/>
                    <input type="range" min={0} max={1} step={0.01} value={trimIn}
                      onChange={e=>setTrimIn(Math.min(Number(e.target.value),trimOut-0.01))}/>
                    <input type="range" min={0} max={1} step={0.01} value={trimOut}
                      onChange={e=>setTrimOut(Math.max(Number(e.target.value),trimIn+0.01))}/>
                  </div>
                </>}
              </div>

              {mode==='manual'&&mockupSrc&&(
                <div className="sec">
                  <div className="sec-title">Corner Pins</div>
                  <p style={{fontSize:10.5,color:'var(--muted)',lineHeight:1.6,marginBottom:8}}>
                    Drag 4 handles onto the device screen corners.
                  </p>
                  <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:11,marginBottom:12}}
                    onClick={()=>setPins(defaultCorners(csz.w,csz.h))}>Reset pins</button>

                  <div className="sec-title" style={{marginBottom:5}}>Edge Style</div>
                  <div className="grade-row" style={{marginBottom:12}}>
                    <button className={`grade-btn${!edgeBlend?' active':''}`} onClick={()=>setEdgeBlend(false)}>Sharp</button>
                    <button className={`grade-btn${edgeBlend?' active':''}`}  onClick={()=>setEdgeBlend(true)}>Soft Blend</button>
                  </div>

                  <div className="sec-title" style={{marginBottom:5}}>Border Width</div>
                  <div className="grade-row" style={{marginBottom:8}}>
                    {[1,2,3].map(w=>(
                      <button key={w} className={`grade-btn${borderWidth===w?' active':''}`} onClick={()=>setBorderWidth(w)}>{w}px</button>
                    ))}
                  </div>

                  <div className="sec-title" style={{marginBottom:5}}>Corner Radius</div>
                  <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:4}}>
                    <input type="range" min={0} max={60} step={1} value={borderRadius}
                      onChange={e=>setBorderRadius(+e.target.value)} style={{flex:1}}/>
                    <div style={{display:'flex',alignItems:'center',gap:3}}>
                      <input type="number" min={0} max={60} step={1} value={borderRadius}
                        onChange={e=>setBorderRadius(Math.max(0,Math.min(60,+e.target.value)))}
                        style={{width:40,fontSize:11,padding:'2px 4px',textAlign:'center',
                          background:'var(--bg)',border:'1px solid var(--border)',color:'var(--text)',borderRadius:4}}/>
                      <span style={{fontSize:10,color:'var(--muted)'}}>px</span>
                    </div>
                  </div>
                  {borderRadius>0&&(
                    <p style={{fontSize:9.5,color:'var(--muted)',lineHeight:1.5,marginBottom:8}}>
                      Rounds all 4 corners — ideal for phones. Replaces left/bottom/right with a full closed border.
                    </p>
                  )}

                  <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:12}}>
                    <span style={{fontSize:11,color:'var(--muted)',flex:1}}>Top border {borderRadius>0?'(sharp mode only)':''}</span>
                    <button className={`grade-btn${topBorder?' active':''}`}
                      onClick={()=>setTopBorder(v=>!v)}
                      style={{opacity:borderRadius>0?0.4:1}}>
                      {topBorder?'● White':'○ Off'}
                    </button>
                  </div>

                  <Slider label="Mockup Opacity" min={0.1} max={1} step={0.02} value={mockupOp} onChange={setMockupOp}/>
                  {mockupOp<0.98&&(
                    <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:10.5,marginTop:5}}
                      onClick={()=>setMockupOp(1.0)}>Restore full opacity</button>
                  )}

                  <div className="sec-title" style={{marginTop:12,marginBottom:5}}>Recording Behind Hand</div>
                  <p style={{fontSize:10.5,color:'var(--muted)',lineHeight:1.6,marginBottom:8}}>
                    Hand visible in your mockup? Choose how to keep it in front of the recording.
                  </p>

                  <div className="sec-title" style={{marginBottom:4,fontSize:9.5}}>METHOD</div>
                  <div className="grade-row" style={{marginBottom:4,flexWrap:'wrap' as const,gap:4}}>
                    <button className={`grade-btn${!punchThrough&&chromaKey<=0&&lumaKey<=0?' active':''}`}
                      style={{flex:'1 1 auto',fontSize:10}}
                      onClick={()=>{setPunchThrough(false);setLumaKey(0);setChromaKey(0);}}>Off</button>
                    <button className={`grade-btn${punchThrough?' active':''}`}
                      style={{flex:'1 1 auto',fontSize:10}}
                      onClick={()=>{setPunchThrough(true);setLumaKey(0);setChromaKey(0);}}>✦ Punch-Through</button>
                    <button className={`grade-btn${!punchThrough&&lumaKey>0&&chromaKey<=0?' active':''}`}
                      style={{flex:'1 1 auto',fontSize:10}}
                      onClick={()=>{setPunchThrough(false);setLumaKey(0.08);setChromaKey(0);}}>Dark Screen</button>
                    <button className={`grade-btn${!punchThrough&&chromaKey>0?' active':''}`}
                      style={{flex:'1 1 auto',fontSize:10}}
                      onClick={()=>{setPunchThrough(false);setChromaKey(0.12);setLumaKey(0);}}>Color Key</button>
                  </div>

                  {punchThrough&&(
                    <p style={{fontSize:10,color:'rgba(91,156,246,0.9)',lineHeight:1.5,marginTop:6,marginBottom:6,
                      padding:'6px 8px',borderRadius:6,background:'rgba(91,156,246,0.08)',border:'1px solid rgba(91,156,246,0.15)'}}>
                      ✦ Screen area is cut out and recording plays below — no keying needed. Works with any mockup.
                    </p>
                  )}

                  {!punchThrough&&lumaKey>0&&chromaKey<=0&&(
                    <>
                      <Slider label="Luma Threshold" min={0.01} max={0.5} step={0.01} value={lumaKey} onChange={setLumaKey}/>
                      <Slider label="Softness" min={0.01} max={0.2} step={0.01} value={lumaSoft} onChange={setLumaSoft}/>
                      <p style={{fontSize:10,color:'var(--muted)',lineHeight:1.5,marginTop:5}}>
                        Start around 0.08. Raise until the screen area clears. Use a mockup with a dark/black screen.
                      </p>
                    </>
                  )}
                  {!punchThrough&&chromaKey>0&&(
                    <>
                      <div style={{marginBottom:6,marginTop:6}}>
                        <div style={{fontSize:9,fontWeight:700,letterSpacing:'.8px',textTransform:'uppercase',color:'var(--muted)',marginBottom:5}}>Screen Color — pick manually</div>
                        <div style={{display:'flex',alignItems:'center',gap:8}}>
                          <div style={{position:'relative',flexShrink:0}}>
                            <div style={{width:36,height:36,borderRadius:6,background:keyColor,border:'2px solid var(--border)',cursor:'pointer'}}
                              onClick={()=>document.getElementById('manual-chroma-pick')?.click()}/>
                            <input id="manual-chroma-pick" type="color" value={keyColor}
                              onChange={e=>setKeyColor(e.target.value)}
                              style={{position:'absolute',opacity:0,width:0,height:0,pointerEvents:'none'}}/>
                          </div>
                          <div style={{flex:1}}>
                            <input value={keyColor} onChange={e=>{if(/^#[0-9a-fA-F]{6}$/.test(e.target.value))setKeyColor(e.target.value);}}
                              style={{width:'100%',fontSize:11,padding:'4px 6px',
                                background:'var(--bg)',border:'1px solid var(--border)',color:'var(--text)',
                                borderRadius:4,fontFamily:'monospace',letterSpacing:'.05em',boxSizing:'border-box' as const}}/>
                            <div style={{fontSize:9,color:'var(--muted)',marginTop:2}}>Click swatch or type hex</div>
                          </div>
                        </div>
                      </div>
                      <Slider label="Key Threshold" min={0.02} max={0.5} step={0.01} value={chromaKey} onChange={setChromaKey}/>
                      <Slider label="Softness" min={0.01} max={0.2} step={0.01} value={lumaSoft} onChange={setLumaSoft}/>
                      <p style={{fontSize:10,color:'var(--muted)',lineHeight:1.5,marginTop:5}}>
                        Pick the exact screen colour. Raise threshold until screen clears — stop before the device body bleeds through.
                      </p>
                    </>
                  )}

                  {/* Camera hole / punch-hole cutout */}
                  <div className="sec-title" style={{marginTop:14,marginBottom:5}}>Camera Hole</div>
                  <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:8}}>
                    <span style={{fontSize:11,color:'var(--muted)',flex:1}}>Punch-hole / notch</span>
                    <button className={`grade-btn${camHole?' active':''}`} onClick={()=>setCamHole(v=>!v)}>
                      {camHole?'● On':'○ Off'}
                    </button>
                  </div>
                  {camHole&&(
                    <>
                      <div style={{display:'flex',gap:5,marginBottom:4}}>
                        <div style={{flex:1}}>
                          <div className="sl-lbl" style={{marginBottom:2}}><span>X position</span><span>{Math.round(camX*100)}%</span></div>
                          <input type="range" min={0} max={1} step={0.01} value={camX}
                            onChange={e=>setCamX(+e.target.value)} style={{width:'100%'}}/>
                        </div>
                        <div style={{flex:1}}>
                          <div className="sl-lbl" style={{marginBottom:2}}><span>Y position</span><span>{Math.round(camY*100)}%</span></div>
                          <input type="range" min={0} max={0.2} step={0.005} value={camY}
                            onChange={e=>setCamY(+e.target.value)} style={{width:'100%'}}/>
                        </div>
                      </div>
                      <div className="sl-lbl" style={{marginBottom:2}}><span>Hole size</span><span>{camRadius}px</span></div>
                      <input type="range" min={6} max={60} step={1} value={camRadius}
                        onChange={e=>setCamRadius(+e.target.value)} style={{width:'100%',marginBottom:6}}/>
                      <p style={{fontSize:9.5,color:'var(--muted)',lineHeight:1.5}}>
                        Cuts a circular hole so the mockup's camera shows through the recording.
                      </p>
                    </>
                  )}

                  {/* ── Audio Track ── */}
                  <div className="sec-title" style={{marginTop:16,marginBottom:6}}>Audio Track</div>
                  <label style={{display:'block',cursor:'pointer',marginBottom:6}}>
                    <div className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:11}}
                      onClick={()=>document.getElementById('audio-pick')?.click()}>
                      {audioName ? `🎵 ${audioName.slice(0,22)}${audioName.length>22?'…':''}` : '+ Add Music / Audio'}
                    </div>
                    <input id="audio-pick" type="file" accept="audio/*" style={{display:'none'}}
                      onChange={e=>{
                        const f=e.target.files?.[0]; if(!f) return;
                        const url=URL.createObjectURL(f);
                        if(!audioElRef.current){audioElRef.current=new Audio();}
                        audioElRef.current.src=url; audioElRef.current.volume=audioVolRef.current;
                        setAudioSrc(url); setAudioName(f.name);
                      }}/>
                  </label>
                  {audioSrc&&(
                    <div style={{marginBottom:8}}>
                      <div className="sl-lbl" style={{marginBottom:3}}>
                        <span>Volume</span><span>{Math.round(audioVol*100)}%</span>
                      </div>
                      <input type="range" min={0} max={1} step={0.01} value={audioVol}
                        onChange={e=>setAudioVol(+e.target.value)} style={{width:'100%'}}/>
                      <div style={{display:'flex',gap:5,marginTop:5}}>
                        <button className="btn btn-ghost" style={{flex:1,fontSize:10,justifyContent:'center'}}
                          onClick={()=>{const ae=audioElRef.current;if(ae){ae.currentTime=0;ae.play();}}}>▶ Preview</button>
                        <button className="btn btn-ghost" style={{flex:1,fontSize:10,justifyContent:'center'}}
                          onClick={()=>{audioElRef.current?.pause();}}>⏹ Stop</button>
                        <button className="btn btn-ghost" style={{flex:1,fontSize:10,justifyContent:'center',color:'#e05050'}}
                          onClick={()=>{
                            audioElRef.current?.pause();
                            if(audioElRef.current) audioElRef.current.src='';
                            setAudioSrc(null); setAudioName('');
                          }}>✕</button>
                      </div>
                      <p style={{fontSize:9,color:'var(--muted)',marginTop:4,lineHeight:1.5}}>
                        Audio will be mixed into your recording. Loops automatically.
                      </p>
                    </div>
                  )}

                  {/* ── Text Overlay ── */}
                  <div className="sec-title" style={{marginTop:16,marginBottom:6}}>Text Overlay</div>
                  <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:11,marginBottom:8}}
                    onClick={addText}>+ Add Text</button>
                  {textItems.map(item=>(
                    <div key={item.id} style={{marginBottom:4,padding:'6px 8px',borderRadius:6,
                      background:selTextId===item.id?'var(--surface)':'transparent',
                      border:`1px solid ${selTextId===item.id?'var(--accent)':'var(--border)'}`,cursor:'pointer'}}
                      onClick={()=>setSelTextId(selTextId===item.id?null:item.id)}>
                      <div style={{fontSize:11,color:'var(--text)',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>
                        {item.text||'(empty)'}
                      </div>
                      {selTextId===item.id&&(
                        <div onClick={e=>e.stopPropagation()} style={{marginTop:6}}>
                          <textarea value={item.text} placeholder="Enter text… (Enter = new line)"
                            rows={3}
                            onChange={e=>updateText(item.id,'text',e.target.value)}
                            style={{width:'100%',marginBottom:5,fontSize:12,padding:'4px 6px',boxSizing:'border-box',
                              background:'var(--bg)',border:'1px solid var(--border)',color:'var(--text)',borderRadius:4,
                              resize:'none',fontFamily:'inherit',lineHeight:1.5}}/>
                          <div style={{marginBottom:7}}>
                            <div style={{fontSize:9,fontWeight:700,letterSpacing:'.8px',textTransform:'uppercase',color:'var(--muted)',marginBottom:4}}>Emojis — click to insert</div>
                            <div style={{display:'flex',flexWrap:'wrap',gap:2,background:'var(--bg)',border:'1px solid var(--border)',borderRadius:5,padding:'5px'}}>
                              {['😀','😂','😍','😎','🥰','😭','🤩','😤','🥳','🔥','✨','💯','👏','🙌','👍','❤️','💕','🎉','🚀','⭐','💪','🏆','💎','⚡','🎯','💡','📱','💻','🎬','🎵'].map(em=>(
                                <span key={em}
                                  onClick={()=>updateText(item.id,'text',item.text+em)}
                                  style={{fontSize:17,cursor:'pointer',padding:'2px 3px',borderRadius:4,lineHeight:1,
                                    transition:'transform .1s'}}
                                  onMouseEnter={e=>(e.currentTarget as HTMLElement).style.transform='scale(1.3)'}
                                  onMouseLeave={e=>(e.currentTarget as HTMLElement).style.transform='scale(1)'}>
                                  {em}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div style={{display:'flex',gap:5,alignItems:'center',marginBottom:6}}>
                            <input type="number" value={item.size} min={8} max={400}
                              onChange={e=>updateText(item.id,'size',Math.max(8,+e.target.value))}
                              style={{width:56,fontSize:11,padding:'3px 4px',
                                background:'var(--bg)',border:'1px solid var(--border)',color:'var(--text)',borderRadius:4}}/>
                            <span style={{fontSize:9,color:'var(--muted)'}}>px</span>
                            <input type="color" value={item.color}
                              onChange={e=>updateText(item.id,'color',e.target.value)}
                              style={{width:26,height:22,border:'none',background:'none',cursor:'pointer',padding:0}}/>
                            <button className={`grade-btn${item.bold?' active':''}`}
                              onClick={()=>updateText(item.id,'bold',!item.bold)}
                              style={{padding:'2px 8px',fontWeight:'bold',fontSize:12,minWidth:28}}>B</button>
                            <button className={`grade-btn${item.italic?' active':''}`}
                              onClick={()=>updateText(item.id,'italic',!item.italic)}
                              style={{padding:'2px 8px',fontStyle:'italic',fontSize:12,minWidth:28}}>I</button>
                          </div>
                          <select value={item.family} onChange={e=>updateText(item.id,'family',e.target.value)}
                            style={{width:'100%',marginBottom:6,fontSize:11,padding:'3px 6px',
                              background:'var(--bg)',border:'1px solid var(--border)',color:'var(--text)',borderRadius:4}}>
                            <optgroup label="── Reel / Bold ──">
                              <option value="'Bebas Neue', sans-serif">Bebas Neue</option>
                              <option value="'Anton', sans-serif">Anton</option>
                              <option value="'Bungee', sans-serif">Bungee</option>
                              <option value="'Bangers', cursive">Bangers</option>
                              <option value="'Racing Sans One', sans-serif">Racing Sans One</option>
                              <option value="'Oswald', sans-serif">Oswald Heavy</option>
                              <option value="'Montserrat', sans-serif">Montserrat Black</option>
                              <option value="'Black Han Sans', sans-serif">Black Han Sans</option>
                            </optgroup>
                            <optgroup label="── Handwriting ──">
                              <option value="'Permanent Marker', cursive">Permanent Marker</option>
                              <option value="'Pacifico', cursive">Pacifico</option>
                              <option value="'Satisfy', cursive">Satisfy</option>
                              <option value="'Righteous', cursive">Righteous</option>
                            </optgroup>
                            <optgroup label="── Elegant ──">
                              <option value="'Cinzel', serif">Cinzel</option>
                              <option value="Georgia, serif">Georgia</option>
                            </optgroup>
                            <optgroup label="── Clean ──">
                              <option value="'Inter', sans-serif">Inter</option>
                              <option value="'Roboto', sans-serif">Roboto</option>
                              <option value="'Arial Black', sans-serif">Arial Black</option>
                            </optgroup>
                          </select>
                          <button className="btn btn-ghost"
                            style={{width:'100%',justifyContent:'center',fontSize:10.5,color:'#e05050'}}
                            onClick={()=>removeText(item.id)}>Remove</button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {mode==='auto'&&(
                <div className="sec">
                  <div className="sec-title">Chroma Key — Screen Color</div>
                  <div style={{marginBottom:10}}>
                    <div style={{fontSize:9,fontWeight:700,letterSpacing:'.8px',textTransform:'uppercase',color:'var(--muted)',marginBottom:5}}>
                      Manual setup — pick screen color
                    </div>
                    <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:6}}>
                      <div style={{position:'relative',flexShrink:0}}>
                        <div style={{width:40,height:40,borderRadius:7,background:keyColor,
                          border:'2px solid var(--border)',cursor:'pointer',boxShadow:'0 2px 8px rgba(0,0,0,.3)'}}
                          onClick={()=>document.getElementById('auto-chroma-pick')?.click()}/>
                        <input id="auto-chroma-pick" type="color" value={keyColor}
                          onChange={e=>setKeyColor(e.target.value)}
                          style={{position:'absolute',opacity:0,width:0,height:0,pointerEvents:'none'}}/>
                      </div>
                      <div style={{flex:1}}>
                        <input value={keyColor} onChange={e=>{if(/^#[0-9a-fA-F]{6}$/.test(e.target.value))setKeyColor(e.target.value);}}
                          style={{width:'100%',fontSize:12,padding:'5px 8px',
                            background:'var(--bg)',border:'1px solid var(--border)',color:'var(--text)',
                            borderRadius:5,fontFamily:'monospace',letterSpacing:'.08em',
                            boxSizing:'border-box' as const,marginBottom:3}}/>
                        <div style={{fontSize:9,color:'var(--muted)'}}>Click swatch or type hex</div>
                      </div>
                    </div>
                    <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:10.5,padding:'5px 0'}} onClick={()=>{
                      if(!mockupSrc||mockupIsV) return;
                      const img=new Image();img.crossOrigin='anonymous';
                      img.onload=()=>{
                        const kc=detectKeyColor(img);
                        setKeyColor(kc);
                        screenBoundsRef.current=detectScreenBounds(img,kc);
                        screenCornersRef.current=detectScreenCorners(img,kc);
                      };img.src=mockupSrc;
                    }}>⚡ Auto-detect from mockup</button>
                  </div>
                  <Slider label="Threshold" min={0.05} max={0.8} step={0.01} value={keyThresh} onChange={setKeyThresh}/>
                  <Slider label="Softness"  min={0.01} max={0.4} step={0.01} value={keySoft}   onChange={setKeySoft}/>
                  <Slider label="Despill"   min={0}    max={1}   step={0.01} value={keySpill}  onChange={setKeySpill}/>

                  {/* Recording placement */}
                  <div className="sec-title" style={{marginTop:14,marginBottom:5}}>Recording Placement</div>
                  <div style={{display:'flex',gap:5,marginBottom:8}}>
                    <button className={`grade-btn${autoPlacement==='fit'?' active':''}`} style={{flex:1,fontSize:9.5}}
                      onClick={()=>setAutoPlacement('fit')}>✦ Scale Fit</button>
                    <button className={`grade-btn${autoPlacement==='corners'?' active':''}`} style={{flex:1,fontSize:9.5}}
                      onClick={()=>setAutoPlacement('corners')}>Corners</button>
                    <button className={`grade-btn${autoPlacement==='fill'?' active':''}`} style={{flex:1,fontSize:9.5}}
                      onClick={()=>setAutoPlacement('fill')}>Fill Canvas</button>
                  </div>
                  {autoPlacement==='fit'&&(
                    <>
                      <p style={{fontSize:9.5,color:'rgba(91,156,246,0.9)',lineHeight:1.5,marginBottom:6}}>
                        Adjust Width + Height independently to fill the phone screen exactly.
                      </p>
                      <div className="sl-lbl" style={{marginBottom:2}}>
                        <span>Width</span><span>{Math.round(autoFitScaleW*200)}%</span>
                      </div>
                      <input type="range" min={0.02} max={0.80} step={0.01} value={autoFitScaleW}
                        onChange={e=>setAutoFitScaleW(+e.target.value)} style={{width:'100%',marginBottom:6}}/>
                      <div className="sl-lbl" style={{marginBottom:2}}>
                        <span>Height</span><span>{Math.round(autoFitScaleH*200)}%</span>
                      </div>
                      <input type="range" min={0.02} max={0.80} step={0.01} value={autoFitScaleH}
                        onChange={e=>setAutoFitScaleH(+e.target.value)} style={{width:'100%',marginBottom:6}}/>
                      <div className="sl-lbl" style={{marginBottom:2}}>
                        <span>Vertical Pos</span><span>{Math.round(autoFitY*100)}%</span>
                      </div>
                      <input type="range" min={0.1} max={0.9} step={0.01} value={autoFitY}
                        onChange={e=>setAutoFitY(+e.target.value)} style={{width:'100%',marginBottom:4}}/>
                      <p style={{fontSize:9,color:'var(--muted)',lineHeight:1.4,marginBottom:4}}>
                        Width/Height: dial each until recording exactly covers screen edges. Vertical Pos: shift up/down to center on phone screen.
                      </p>
                    </>
                  )}
                  {autoPlacement==='corners'&&(
                    <>
                      <p style={{fontSize:9.5,color:'var(--muted)',lineHeight:1.5,marginBottom:4}}>
                        Perspective-warps recording to exact detected corners. Use Screen Scale if the recording still overflows.
                      </p>
                      <div className="sl-lbl" style={{marginBottom:3}}><span>Screen Scale</span><span>{Math.round(autoRecScale*100)}%</span></div>
                      <input type="range" min={0.5} max={1.2} step={0.01} value={autoRecScale}
                        onChange={e=>setAutoRecScale(+e.target.value)} style={{width:'100%',marginBottom:4}}/>
                    </>
                  )}
                  {autoPlacement==='fill'&&(
                    <p style={{fontSize:10,color:'rgba(91,156,246,0.9)',lineHeight:1.5,
                      padding:'6px 8px',borderRadius:6,background:'rgba(91,156,246,0.08)',border:'1px solid rgba(91,156,246,0.15)',marginBottom:6}}>
                      Recording fills the entire canvas — chroma key is the only mask. Best for moving-phone videos where screen position changes each frame.
                    </p>
                  )}
                </div>
              )}

              <div className="sec">
                <div className="sec-title">Color Grade</div>
                <div className="grade-row">
                  {(['none','natural','cinematic','vivid'] as GradeName[]).map(g=>(
                    <button key={g} className={`grade-btn${grade===g?' active':''}`} onClick={()=>applyGrade(g)}>
                      {g==='none'?'Off':g.charAt(0).toUpperCase()+g.slice(1)}
                    </button>
                  ))}
                </div>
                <div className="expand-row" onClick={()=>setExpandEnh(v=>!v)}>
                  <span className="expand-lbl">Custom controls</span>
                  <span className={`expand-arrow${expandEnh?' open':''}`}>▾</span>
                </div>
                {expandEnh&&<>
                  <Slider label="Sharpen"     min={0}    max={2}   step={0.05} value={enhance.sharp}    onChange={v=>{setGrade('none');setEnhance(p=>({...p,sharp:v}));}}/>
                  <Slider label="Brightness"  min={-.25} max={.25} step={0.01} value={enhance.bright}   onChange={v=>{setGrade('none');setEnhance(p=>({...p,bright:v}));}}/>
                  <Slider label="Contrast"    min={0.5}  max={2}   step={0.01} value={enhance.contrast} onChange={v=>{setGrade('none');setEnhance(p=>({...p,contrast:v}));}}/>
                  <Slider label="Saturation"  min={0}    max={2}   step={0.01} value={enhance.sat}      onChange={v=>{setGrade('none');setEnhance(p=>({...p,sat:v}));}}/>
                  <Slider label="Vignette"    min={0}    max={1}   step={0.01} value={enhance.vignette} onChange={v=>{setGrade('none');setEnhance(p=>({...p,vignette:v}));}}/>
                  <Slider label="Temperature" min={-.3}  max={.3}  step={0.01} value={enhance.temp}     onChange={v=>{setGrade('none');setEnhance(p=>({...p,temp:v}));}}/>
                  <Slider label="Bloom"       min={0}    max={1}   step={0.01} value={enhance.bloom}    onChange={v=>{setGrade('none');setEnhance(p=>({...p,bloom:v}));}}/>
                  <Slider label="Film Grain"  min={0}    max={1}   step={0.01} value={enhance.grain}    onChange={v=>{setGrade('none');setEnhance(p=>({...p,grain:v}));}}/>
                </>}
              </div>

              <div className="sec">
                <div className="expand-row" onClick={()=>setShowPre(v=>!v)}>
                  <span className="expand-lbl">Presets</span>
                  <span className={`expand-arrow${showPre?' open':''}`}>▾</span>
                </div>
                {showPre&&<>
                  {presets.length>0&&(
                    <div className="preset-list">
                      {presets.map((p,i)=>(
                        <div key={i} className="preset-item" onClick={()=>loadPreset(p)}>
                          <span>{p.name}</span>
                          <button onClick={e=>{e.stopPropagation();deletePreset(i);}}>✕</button>
                        </div>
                      ))}
                    </div>
                  )}
                  <div style={{display:'flex',gap:5}}>
                    <input className="preset-input" placeholder="Preset name…"
                      value={presetName} onChange={e=>setPresetName(e.target.value)}
                      onKeyDown={e=>{if(e.key==='Enter') savePreset();}}/>
                    <button className="btn btn-ghost" style={{padding:'5px 10px',fontSize:11}} onClick={savePreset}>Save</button>
                  </div>
                </>}
              </div>

              {mockupSrc&&(
                <div className="sec">
                  <div className="sec-title">Batch Export</div>
                  <p style={{fontSize:10,color:'var(--muted)',lineHeight:1.65,marginBottom:8}}>
                    Drop multiple recordings — each gets exported with the current mockup, pins &amp; grade.
                  </p>
                  <div className="batch-drop"
                    onDragOver={e=>e.preventDefault()}
                    onDrop={e=>{
                      e.preventDefault();
                      const files=Array.from(e.dataTransfer.files).filter(f=>f.type.startsWith('video/')||/\.(mp4|webm|mov)$/i.test(f.name));
                      setBatchFiles(prev=>[...prev,...files.map(f=>({name:f.name,url:URL.createObjectURL(f)}))]);
                    }}
                    onClick={()=>{
                      const inp=document.createElement('input');inp.type='file';inp.accept='video/*';inp.multiple=true;
                      inp.onchange=e=>{
                        const files=Array.from((e.target as HTMLInputElement).files||[]);
                        setBatchFiles(prev=>[...prev,...files.map(f=>({name:f.name,url:URL.createObjectURL(f)}))]);
                      };inp.click();
                    }}>
                    <div className="drop-ico">🎬</div>
                    <div className="drop-tx"><strong>{batchFiles.length?`${batchFiles.length} queued — add more`:'Add recordings'}</strong>MP4 · WebM · MOV</div>
                  </div>
                  {batchFiles.length>0&&(
                    <>
                      <div style={{maxHeight:130,overflowY:'auto',marginBottom:6}}>
                        {batchFiles.map((f,i)=>(
                          <div key={i} className={`batch-item${batchIdx===i?' active':batchIdx!==null&&i<batchIdx?' done':''}`}>
                            <span style={{overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',flex:1,marginRight:4}}>
                              {batchIdx!==null&&i<batchIdx?'✓ ':batchIdx===i?'⚙ ':''}{f.name}
                            </span>
                            {batchIdx===null&&(
                              <button onClick={()=>setBatchFiles(p=>p.filter((_,j)=>j!==i))}
                                style={{background:'none',border:'none',color:'var(--muted)',cursor:'pointer',fontSize:11,flexShrink:0}}>✕</button>
                            )}
                          </div>
                        ))}
                      </div>
                      {batchIdx!==null
                        ?<div style={{fontSize:10,color:'var(--accent)',textAlign:'center',padding:'5px 0',fontWeight:600}}>
                          Processing {batchIdx+1} / {batchFiles.length}…
                        </div>
                        :<>
                          <button className="btn btn-export" style={{width:'100%',justifyContent:'center',fontSize:11,marginBottom:4}}
                            onClick={runBatch}>
                            ▶ Run Batch ({batchFiles.length})
                          </button>
                          <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:10}}
                            onClick={()=>setBatchFiles([])}>Clear queue</button>
                        </>
                      }
                    </>
                  )}
                </div>
              )}

              {/* ── Template Batch ─────────────────────────────────────── */}
              {mockupSrc&&(
                <div className="sec">
                  <div className="sec-title" style={{marginBottom:6}}>Templates</div>
                  <p style={{fontSize:10,color:'var(--muted)',lineHeight:1.65,marginBottom:8}}>
                    Save the current mockup+pins+grade as a template. One recording → export through all selected templates.
                  </p>
                  {/* Save button */}
                  <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:11,marginBottom:8}}
                    onClick={()=>{
                      const name=prompt('Template name (e.g. MacBook, iPhone 15):');
                      if(name?.trim()) saveCurrentTemplate(name.trim());
                    }}>+ Save Current as Template</button>

                  {/* Template list */}
                  {templates.length>0&&<>
                    <div style={{display:'flex',flexDirection:'column',gap:4,marginBottom:8}}>
                      {templates.map((t,i)=>(
                        <div key={t.id} style={{display:'flex',alignItems:'center',gap:6,
                          padding:'5px 7px',borderRadius:7,background:'var(--surface)',
                          border:`1.5px solid ${selTpls.has(t.id)?'var(--accent)':'var(--border)'}`,
                          cursor:'pointer',transition:'all .15s'}}
                          onClick={()=>setSelTpls(s=>{const n=new Set(s);s.has(t.id)?n.delete(t.id):n.add(t.id);return n;})}>
                          <img src={t.thumb} alt="" style={{width:40,height:26,objectFit:'cover',borderRadius:4,flexShrink:0}}/>
                          <span style={{flex:1,fontSize:11,color:'var(--text)',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{t.name}</span>
                          {tplBatchIdx!==null&&templates.filter(x=>selTpls.has(x.id))[tplBatchIdx]?.id===t.id&&
                            <span style={{fontSize:9,color:'var(--accent)'}}>●</span>}
                          <button style={{background:'none',border:'none',color:'var(--muted)',cursor:'pointer',fontSize:13,padding:'0 2px'}}
                            onClick={e=>{e.stopPropagation();deleteTemplate(t.id);}}>✕</button>
                        </div>
                      ))}
                    </div>
                    {selTpls.size>0&&(
                      tplBatchIdx!==null
                      ?<div style={{textAlign:'center',fontSize:11,color:'var(--accent)',padding:'6px 0'}}>
                        Exporting template {tplBatchIdx+1}/{selTpls.size}…
                      </div>
                      :<button className="btn btn-export" style={{width:'100%',justifyContent:'center',fontSize:11}}
                        onClick={runTemplateBatch}>
                        ▶ Export {selTpls.size} Template{selTpls.size>1?'s':''}
                      </button>
                    )}
                  </>}
                </div>
              )}

              <div className="sec">
                <div className="sec-title">Quick guide</div>
                {mode==='manual'
                  ?<div style={{fontSize:10.5,color:'var(--muted)',lineHeight:2.1}}>
                    <div>① Library → pick a device photo</div>
                    <div>② Upload your screen recording</div>
                    <div>③ Drag 4 pins onto the screen</div>
                    <div>④ Choose Natural · Cinematic · Vivid</div>
                    <div>⑤ Export PNG or record WebM</div>
                  </div>
                  :<div style={{fontSize:10.5,color:'var(--muted)',lineHeight:2.1}}>
                    <div>① Upload mockup with solid-colour screen</div>
                    <div>② Upload screen recording</div>
                    <div>③ Hit Auto-detect key colour</div>
                    <div>④ Tune Threshold to clean edges</div>
                    <div>⑤ Export</div>
                  </div>}
              </div>
            </>}
          </aside>

          <main className="canvas-area"
            onMouseMove={e=>{
              const r=(e.currentTarget as HTMLElement).getBoundingClientRect();
              mouseXRef.current=(e.clientX-r.left)/r.width;
            }}
            onWheel={onAreaWheel}>
            <div style={{display:'flex',flexDirection:'column',alignItems:'center',
              transform:`translate(${pan.x}px,${pan.y}px)`,transition:'none'}}>
            <div className="canvas-wrap" style={{width:csz.w,height:csz.h,
              transform:`scale(${zoom})`,transformOrigin:'center top',
              cursor:activePin!==null?'grabbing':'grab'}}
              onPointerDown={onCanvasDown}
              onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}>
              <canvas ref={canvasRef} width={native.w} height={native.h}
                style={{display:'block',width:csz.w,height:csz.h}}/>
              {mode==='manual'&&mockupSrc&&(
                <svg style={{position:'absolute',inset:0,pointerEvents:'none'}} width={csz.w} height={csz.h}>
                  {/* Thin outline of the mapped screen rect */}
                  <polyline points={[...pins,pins[0]].map(p=>`${p.x},${p.y}`).join(' ')}
                    fill="none" stroke="rgba(255,255,255,.18)" strokeWidth="1"/>
                </svg>
              )}
              {mode==='manual'&&mockupSrc&&pins.map((p,i)=>{
                // L-bracket sits OUTSIDE the screen corner — never covers the actual edge point
                // Each bracket extends 12px along the two edges meeting at this corner, offset 1px outward
                const O=1; // outward offset in px
                const A=12; // arm length
                const offsets=[
                  {left:p.x-O-A, top:p.y-O-A}, // TL: bracket opens bottom-right
                  {left:p.x+O,   top:p.y-O-A}, // TR: bracket opens bottom-left
                  {left:p.x+O,   top:p.y+O},   // BR: bracket opens top-left
                  {left:p.x-O-A, top:p.y+O},   // BL: bracket opens top-right
                ];
                // Arm positions inside the 20×20 drag zone for each corner
                const arms=[
                  [{bottom:0,right:0},{bottom:0,right:0}],  // TL: H at bottom-right, V at bottom-right
                  [{bottom:0,left:0},{bottom:0,left:0}],    // TR
                  [{top:0,left:0},{top:0,left:0}],          // BR
                  [{top:0,right:0},{top:0,right:0}],        // BL
                ];
                const hStyle=[
                  {bottom:0,right:0,width:A,height:2},
                  {bottom:0,left:0,width:A,height:2},
                  {top:0,left:0,width:A,height:2},
                  {top:0,right:0,width:A,height:2},
                ];
                const vStyle=[
                  {bottom:0,right:0,width:2,height:A},
                  {bottom:0,left:0,width:2,height:A},
                  {top:0,left:0,width:2,height:A},
                  {top:0,right:0,width:2,height:A},
                ];
                return(
                  <div key={i} className={`pin-corner${activePin===i?' active':''}`}
                    style={{...offsets[i]}} onPointerDown={onPinDown(i)}>
                    <div className="arm-h" style={hStyle[i]}/>
                    <div className="arm-v" style={vStyle[i]}/>
                  </div>
                );
              })}
              {mockupSrc&&textItems.map(item=>(
                <div key={item.id}
                  style={{position:'absolute',left:item.x*csz.w,top:item.y*csz.h,
                    fontSize:item.size,fontFamily:item.family,
                    fontWeight:item.bold?'bold':'normal',fontStyle:item.italic?'italic':'normal',
                    color:item.color,cursor:'move',userSelect:'none',pointerEvents:'all',
                    outline:selTextId===item.id?'1px dashed var(--accent)':'1px dashed transparent',
                    padding:'2px 4px',whiteSpace:'pre',
                    textShadow:'0 2px 8px rgba(0,0,0,0.6)',lineHeight:1.25}}
                  onPointerDown={e=>{
                    e.stopPropagation();
                    setSelTextId(item.id);
                    const rect=canvasRef.current!.getBoundingClientRect();
                    const z=zoomRef.current;
                    textDragRef.current={id:item.id,sx:(e.clientX-rect.left)/z,sy:(e.clientY-rect.top)/z,ox:item.x,oy:item.y};
                    (e.target as HTMLElement).setPointerCapture(e.pointerId);
                  }}>
                  {item.text||'✦ Text'}
                </div>
              ))}
              {!mockupSrc&&(
                <div className="empty" style={{width:csz.w,height:csz.h}}>
                  <div className="empty-ico">🖼️</div>
                  <h3>Upload a mockup or browse Library</h3>
                  <p>Grab a real device photo from the Library tab, or drag your own mockup file here.</p>
                </div>
              )}
            </div>
            {/* Glass-table reflection — only when mockup loaded */}
            {mockupSrc&&(
              <canvas ref={reflRef}
                width={native.w} height={Math.round(native.h*0.26)}
                style={{
                  display:'block',
                  width:csz.w,
                  height:Math.round(csz.h*0.26),
                  opacity:0.7,
                  pointerEvents:'none',
                  marginTop:0,
                  transform:`scale(${zoom})`,transformOrigin:'center top',
                  filter:'blur(1.5px) brightness(1.2)',
                  mixBlendMode:'screen' as React.CSSProperties['mixBlendMode'],
                }}/>
            )}
            </div>
            {/* Zoom pill */}
            <div className="zoom-pill">
              <button className="zoom-btn" onClick={()=>setZoom(z=>Math.max(0.2,z*0.8))}>−</button>
              <span className="zoom-pct" title="Click to reset" onClick={fitZoom}>{Math.round(zoom*100)}%</span>
              <button className="zoom-btn" onClick={()=>setZoom(z=>Math.min(4,z*1.25))}>+</button>
              <div style={{width:1,height:14,background:'var(--border)',margin:'0 3px'}}/>
              <button className="zoom-btn" style={{fontSize:10,padding:'2px 6px'}} onClick={fitZoom} title="Fit to screen">⊡</button>
            </div>
          </main>
        </div>
      </div>

      {showExp&&(
        <div className="modal-ov" onClick={()=>setShowExp(false)}>
          <div className="modal" onClick={e=>e.stopPropagation()}>
            <div className="modal-hdr">
              <h2>Export</h2>
              <button className="modal-x" onClick={()=>setShowExp(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="m-stat"><span>Resolution</span><strong>
                {exportRatio==='9:16'?'1080 × 1920':exportRatio==='1:1'?'1080 × 1080':`${native.w} × ${native.h}`}
              </strong></div>
              <div className="m-stat"><span>Grade</span><strong>{grade==='none'?'None':grade.charAt(0).toUpperCase()+grade.slice(1)}</strong></div>
              {recDur>0&&<div className="m-stat"><span>Clip</span><strong>{fmt(Math.round(trimIn*recDur))} – {fmt(Math.round(trimOut*recDur))}</strong></div>}
              <div className="m-lbl">Format</div>
              <div className="fmt-tabs">
                <div className={`fmt-tab${expFmt==='png'?' active':''}`} onClick={()=>setExpFmt('png')}>📷 PNG Still</div>
                <div className={`fmt-tab${expFmt==='webm'?' active':''}`} onClick={()=>setExpFmt('webm')}>🎬 WebM Video</div>
              </div>
              {expFmt==='webm'&&<>
                <div className="m-lbl">Aspect Ratio</div>
                <div className="q-row">
                  {(['16:9','9:16','1:1'] as ExportRatio[]).map(r=>(
                    <div key={r} className={`q-btn${exportRatio===r?' active':''}`} onClick={()=>setExportRatio(r)}
                      style={{flexDirection:'column',alignItems:'center'}}>
                      <span style={{fontSize:12,fontWeight:700}}>{r}</span>
                      <span style={{fontSize:8,opacity:.6,marginTop:2}}>
                        {r==='16:9'?'LinkedIn · X':r==='9:16'?'Reels · TikTok':'Instagram'}
                      </span>
                    </div>
                  ))}
                </div>
                {exportRatio!=='16:9'&&(
                  <>
                    <div className="m-lbl" style={{marginTop:10}}>Background Fill</div>
                    <div className="q-row">
                      <div className={`q-btn${bgFill==='blur'?' active':''}`} onClick={()=>setBgFill('blur')}
                        style={{flexDirection:'column',alignItems:'center'}}>
                        <span style={{fontSize:11,fontWeight:700}}>Blurred</span>
                        <span style={{fontSize:8,opacity:.6,marginTop:2}}>cinematic look</span>
                      </div>
                      <div className={`q-btn${bgFill==='black'?' active':''}`} onClick={()=>setBgFill('black')}
                        style={{flexDirection:'column',alignItems:'center'}}>
                        <span style={{fontSize:11,fontWeight:700}}>Black</span>
                        <span style={{fontSize:8,opacity:.6,marginTop:2}}>clean bars</span>
                      </div>
                    </div>
                  </>
                )}
                <div className="m-lbl">Quality</div>
                <div className="q-row">
                  <div className={`q-btn${quality==='high'?' active':''}`} onClick={()=>setQuality('high')}>
                    High<br/><span style={{fontSize:9,opacity:.6}}>40 Mbps</span>
                  </div>
                  <div className={`q-btn${quality==='ultra'?' active':''}`} onClick={()=>setQuality('ultra')}>
                    Ultra<br/><span style={{fontSize:9,opacity:.6}}>80 Mbps</span>
                  </div>
                </div>
                <div className="q-note">60 fps · grade baked in · starts from trim point</div>
              </>}
              {expFmt==='png'
                ?<button className="btn btn-export" style={{width:'100%',justifyContent:'center',padding:10}} onClick={doExportPNG}>Export PNG</button>
                :(recSrc&&!rStaticRef.current&&typeof (window as any).VideoEncoder!=='undefined'
                  ?<div style={{display:'flex',flexDirection:'column',gap:6}}>
                    <button className="btn btn-export" style={{width:'100%',justifyContent:'center',padding:10}} onClick={startOfflineExport}>
                      ↓ Export MP4 — {exportRatio}
                    </button>
                    <button className="btn btn-export" style={{width:'100%',justifyContent:'center',padding:10,
                      background:'linear-gradient(135deg,#7C6AF7,#5B9CF6)'}}
                      onClick={()=>{setShowExp(false);runAllFormats();}}>
                      ↓ Export All 3 Formats
                      <span style={{fontSize:9,opacity:.7,marginLeft:6}}>reels · square · landscape</span>
                    </button>
                  </div>
                  :<button className="btn btn-export" style={{width:'100%',justifyContent:'center',padding:10}} onClick={startRec}>● Start Recording</button>
                )}
            </div>
          </div>
        </div>
      )}

      {isExporting&&(
        <div style={{position:'fixed',inset:0,background:'rgba(4,4,14,.93)',display:'flex',
          flexDirection:'column',alignItems:'center',justifyContent:'center',zIndex:600}}>
          <div style={{fontSize:13,fontWeight:700,color:'var(--text)',marginBottom:18,letterSpacing:'0.05em'}}>
            Exporting — real-time H.264
          </div>
          <div style={{width:300,height:6,background:'var(--surface)',borderRadius:3,overflow:'hidden'}}>
            <div style={{width:`${exportProgress*100}%`,height:'100%',background:'linear-gradient(90deg,#7C6AF7,#5B9CF6)',
              transition:'width .25s linear',borderRadius:3}}/>
          </div>
          <div style={{color:'var(--muted)',fontSize:11,marginTop:10}}>
            {Math.round(exportProgress*100)}% — keep this tab active &amp; visible
          </div>
        </div>
      )}

      {toast&&<div className={`toast${toast.err?' err':''}`}>{toast.msg}</div>}
    </>
  );
}

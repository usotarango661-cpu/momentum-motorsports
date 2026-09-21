// Headless three.js renderer: node render.mjs model.glb out_prefix [--views front,quarter,side,back] [--size 1024] [--bg 0x0b0b0c] [--turntable N]
// Renders a GLB with studio lighting (3-point + environment) and writes PNGs.
import { chromium } from 'playwright'
import fs from 'node:fs'
import path from 'node:path'

const [,, glbPath, outPrefix, ...rest] = process.argv
const opt = { views: 'front,quarter,side,back', size: '1024', bg: '0x0b0b0c', turntable: '0', hdr: '1' }
for (let i = 0; i < rest.length; i += 2) opt[rest[i].replace(/^--/, '')] = rest[i + 1]
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname))
const glb = fs.readFileSync(glbPath).toString('base64')

const html = `<!doctype html><html><body style="margin:0;background:#000">
<script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js","three/addons/":"/node_modules/three/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
const SIZE = ${Number(opt.size)}
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: true })
renderer.setSize(SIZE, SIZE); renderer.setPixelRatio(1)
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.15
renderer.outputColorSpace = THREE.SRGBColorSpace
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap
document.body.appendChild(renderer.domElement)
const scene = new THREE.Scene()
scene.background = new THREE.Color(${opt.bg})
const pmrem = new THREE.PMREMGenerator(renderer)
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
const key = new THREE.DirectionalLight(0xfff2dc, 2.2); key.position.set(2, 3, 4); key.castShadow = true
const fill = new THREE.DirectionalLight(0xdde6ff, 0.8); fill.position.set(-3, 1, 2)
const rim = new THREE.DirectionalLight(0xffffff, 1.6); rim.position.set(0, 2, -4)
scene.add(key, fill, rim, new THREE.AmbientLight(0xffffff, 0.15))
const camera = new THREE.PerspectiveCamera(30, 1, 0.001, 100)
const bin = Uint8Array.from(atob('${glb}'), c => c.charCodeAt(0))
const gltf = await new GLTFLoader().parseAsync(bin.buffer, '')
const model = gltf.scene
model.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true } })
scene.add(model)
const box = new THREE.Box3().setFromObject(model)
const c = box.getCenter(new THREE.Vector3()); const s = box.getSize(new THREE.Vector3())
const r = Math.max(s.x, s.y, s.z)
const dist = r / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 0.62
window.__stats = { center: [c.x, c.y, c.z], size: [s.x, s.y, s.z], tris: 0 }
model.traverse(o => { if (o.isMesh) window.__stats.tris += o.geometry.index ? o.geometry.index.count / 3 : o.geometry.attributes.position.count / 3 })
window.__render = (az, el) => {
  const a = THREE.MathUtils.degToRad(az), e = THREE.MathUtils.degToRad(el)
  camera.position.set(c.x + dist * Math.sin(a) * Math.cos(e), c.y + dist * Math.sin(e), c.z + dist * Math.cos(a) * Math.cos(e))
  camera.lookAt(c); camera.updateProjectionMatrix()
  renderer.render(scene, camera)
  return renderer.domElement.toDataURL('image/png')
}
window.__ready = true
</script></body></html>`

const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: Number(opt.size), height: Number(opt.size) } })
page.on('console', m => console.log('[page]', m.text()))
page.on('pageerror', e => console.log('[pageerror]', e.message))
await page.route('**/*', async route => {
  const u = new URL(route.request().url())
  if (u.pathname === '/index.html') return route.fulfill({ contentType: 'text/html', body: html })
  const f = path.join(ROOT, u.pathname)
  if (fs.existsSync(f)) return route.fulfill({ contentType: 'text/javascript', body: fs.readFileSync(f) })
  return route.fulfill({ status: 404, body: '' })
})
await page.goto('http://render.local/index.html')
await page.waitForFunction(() => window.__ready === true, null, { timeout: 120000 })
const stats = await page.evaluate(() => window.__stats)
console.log('model stats', JSON.stringify(stats))
const VIEWS = { front: [0, 8], quarter: [35, 18], side: [90, 5], back: [180, 8], top: [0, 70], low: [-30, -15] }
for (const v of opt.views.split(',')) {
  const [az, el] = VIEWS[v] || [0, 8]
  const data = await page.evaluate(([a, e]) => window.__render(a, e), [az, el])
  fs.writeFileSync(`${outPrefix}_${v}.png`, Buffer.from(data.split(',')[1], 'base64'))
  console.log('wrote', `${outPrefix}_${v}.png`)
}
const n = Number(opt.turntable)
for (let i = 0; i < n; i++) {
  const data = await page.evaluate(([a, e]) => window.__render(a, e), [i * 360 / n, 12])
  fs.writeFileSync(`${outPrefix}_turn${String(i).padStart(2, '0')}.png`, Buffer.from(data.split(',')[1], 'base64'))
}
await browser.close()
